

import torch
import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_names = ['O',
'B-Natural_Resource',
'I-Natural_Resource',
'B-Duration',
'I-Duration',
'B-Humidity',
'I-Humidity',
'B-Other',
'I-Other',
'B-Organism',
'I-Organism',
'B-Soil',
'I-Soil',
'B-Agri_Process',
'I-Agri_Process',
'B-Season',
'I-Season',
'B-Money',
'I-Money',
'B-Agri_Method',
'I-Agri_Method',
'B-ML_Model',
'I-ML_Model',
'B-Fruit',
'I-Fruit',
'B-Agri_Pollution',
'I-Agri_Pollution',
'B-Person',
'I-Person',
'B-Agri_Waste',
'I-Agri_Waste',
'B-Location',
'I-Location',
'B-Technology',
'I-Technology',
'B-Natural_Disaster',
'I-Natural_Disaster',
'B-Disease',
'I-Disease',
'B-Crop',
'I-Crop',
'B-Treatment',
'I-Treatment',
'B-Rainfall',
'I-Rainfall',
'B-Quantity',
'I-Quantity',
'B-Vegetable',
'I-Vegetable',
'B-Chemical',
'I-Chemical',
'B-Policy',
'I-Policy',
'B-Nutrient',
'I-Nutrient',
'B-Field_Area',
'I-Field_Area',
'B-Temp',
'I-Temp',
'B-Date_and_Time',
'I-Date_and_Time',
'B-Food_Item',
'I-Food_Item',
'B-Weather',
'I-Weather',
'B-Other_Quantity',
'I-Other_Quantity',
'B-Organization',
'I-Organization',
'B-Citation',
'I-Citation',
'B-Event',
'I-Event']


# Load SpanBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased", num_labels=len(label_names))
model.to(device)

class NERDataset(Dataset):
    def __init__(self, filepath, tokenizer, label_map, max_len=128):
        self.data = pd.read_csv(filepath)
        self.data['Tokens'] = self.data['Tokens'].apply(literal_eval)
        self.data['Entities'] = self.data['Entities'].apply(literal_eval)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx]['Tokens']
        labels = self.data.iloc[idx]['Entities']

        # Tokenize tokens with word alignment
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")

        # Map labels to tokenized output
        label_ids = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens (CLS, SEP, PAD)
                label_ids.append(-100)  # Ignore these during loss computation
            else:
                label_ids.append(self.label_map[labels[word_idx]])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids)
        }




from sklearn.model_selection import train_test_split

# Filepaths for train and test data
train_file = "D:/Sayan/Augment_Roberta/New_train.csv"  # Replace with actual path
test_file = "D:/Sayan/Augment_Roberta/New_test.csv"    # Replace with actual path

# Load datasets
label_map = {label: idx for idx, label in enumerate(label_names)}

# Create Dataset objects
train_dataset = NERDataset(filepath=train_file, tokenizer=tokenizer, label_map=label_map)
test_dataset = NERDataset(filepath=test_file, tokenizer=tokenizer, label_map=label_map)

# Create DataLoaders
data_collator = DataCollatorForTokenClassification(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)


import optuna
# from transformers import AdamW
from torch.optim import AdamW
# from seqeval.metrics import precision_score, recall_score, f1_score

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, get_scheduler

from seqeval.metrics import f1_score as seqeval_f1_score, classification_report as seqeval_classification_report

def evaluate_model(model, val_loader, label_names):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2).cpu().numpy()

            # Prepare predictions and true labels
            for pred, true in zip(pred_ids, labels.numpy()):
                pred_labels = [label_names[p] for p, t in zip(pred, true) if t != -100]  # Ignore padding
                true_labels_seq = [label_names[t] for t in true if t != -100]
                predictions.append(pred_labels)
                true_labels.append(true_labels_seq)

    # Use seqeval metrics
    f1 = seqeval_f1_score(true_labels, predictions)
    print(seqeval_classification_report(true_labels, predictions))
    return f1



def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    max_epochs = trial.suggest_int("max_epochs", 5, 15)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    warmup_steps_ratio = trial.suggest_float("warmup_steps_ratio", 0.0, 0.2)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)

    # Update DataLoader with batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    # Define model
    config = AutoConfig.from_pretrained("SpanBERT/spanbert-base-cased", num_labels=len(label_map))
    config.hidden_dropout_prob = dropout_rate  # Update dropout
    model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased", config=config)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_steps_ratio)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(max_epochs):
        model.train()

        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {total_loss:.4f}")

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2).cpu().numpy()

            for pred, true in zip(pred_ids, labels.numpy()):
                pred_labels = [label_names[p] for p, t in zip(pred, true) if t != -100]
                true_labels_seq = [label_names[t] for t in true if t != -100]
                predictions.append(pred_labels)
                true_labels.append(true_labels_seq)

    print(classification_report(true_labels, predictions))

    # Calculate F1 Score
    f1 = seqeval_f1_score(true_labels, predictions, average="weighted")
    return f1
if __name__ == "__main__":
    # Define the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(study.best_params)

    best_params = study.best_params
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    max_epochs = best_params["max_epochs"]
    dropout_rate = best_params["dropout_rate"]
    weight_decay = best_params["weight_decay"]
    warmup_steps_ratio = best_params["warmup_steps_ratio"]
    max_grad_norm = best_params["max_grad_norm"]

    # Update DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    # Train final model
    final_model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased",
                                                                  num_labels=len(label_map))
    final_model.to(device)
    optimizer = AdamW(final_model.parameters(), lr=learning_rate)

    # Training loop with best hyperparameters
    for epoch in range(max_epochs):
        final_model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = final_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {total_loss:.4f}")

    # Evaluate the final model
    evaluate_model(final_model, val_loader, label_names)
