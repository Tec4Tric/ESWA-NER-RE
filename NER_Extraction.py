import pandas as pd
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import optuna
from optuna.integration import HuggingFacePruningCallback
from sklearn.model_selection import train_test_split

# Load CSV and preprocess
raw_df = pd.read_csv("D:/Sayan/Augment_Roberta/NER_data.csv")
raw_df['Tokens'] = raw_df['Tokens'].apply(eval)
raw_df['Entities'] = raw_df['Entities'].apply(eval)

# Prepare Hugging Face Dataset
dataset = Dataset.from_pandas(raw_df[['Tokens', 'Entities']].rename(columns={'Tokens': 'tokens', 'Entities': 'ner_tags'}))
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Label mappings
label_list = sorted({label for labels in raw_df['Entities'] for label in labels})
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

# Tokenization and label alignment
def tokenize_and_align_labels_batch(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, padding='max_length', truncation=True, max_length=128)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('I-') else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels_batch, batched=True)

# Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    metric = evaluate.load("seqeval")
    return metric.compute(predictions=true_predictions, references=true_labels)

# Model init
def model_init():
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    return model

# Hyperparameter space
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 5.0),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./agner-ner-results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

# Hyperparameter tuning with Optuna
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=20,
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["eval_f1"]
)

print("Best Trial:", best_trial)

# Save final model
trainer.save_model("./AgNER-BERTa")
tokenizer.save_pretrained("./AgNER-BERTa")
