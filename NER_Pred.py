import re

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the model and fast tokenizer
model_name = "./NEW_ner-roberta-base_512"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name)
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

id2label = {id : label for id, label in enumerate(label_names)}
label2id = {label: id for id, label in enumerate(label_names)}

def NER_Pred(text):
    def predict_entities(text):
        # Tokenize the input text with offset mapping
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get the predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted token class indices
        predictions = torch.argmax(outputs.logits, dim=2)

        # Convert indices to labels
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        predictions = predictions.squeeze().tolist()
        # offset_mapping = inputs["offset_mapping"].squeeze().tolist()

        # Post-process to align subwords with their entities
        word_ids = inputs.word_ids()


        previous_word_idx = None
        label_ids = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append("O")
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(id2label[predictions[idx]])
            else:
                label_ids.append(id2label[predictions[idx]].replace('B-', 'I-'))  # Continuation of the word
            previous_word_idx = word_idx

        # Combine tokens and labels, skipping special tokens
        final_tokens = []
        final_labels = []
        for token, label in zip(tokens, label_ids):
            if token not in tokenizer.all_special_tokens:
                final_tokens.append(token)
                final_labels.append(label)

        # Reconstruct the original words and align the labels
        words = tokenizer.convert_tokens_to_string(final_tokens).split()
        final_word_labels = []
        current_word = ""
        current_label = "O"
        # numerical_labels = list(model.config.label2id.values())
        #
        # # Convert numerical predictions to original labels
        # final_labels = [label_names[label] for label in numerical_labels]
        for token, label in zip(final_tokens, final_labels):

            if not token.startswith("Ġ"):
                current_word += token.replace("Ġ", "")
            else:
                if current_word:
                    final_word_labels.append((current_word, current_label))
                current_word = token.replace("Ġ", "")
                current_label = label
        if current_word:
            final_word_labels.append((current_word, current_label))

        return final_word_labels


    def convert_to_bio(predictions):
        bio_predictions = []
        current_entity_type = None

        for token, entity_type in predictions:
            if entity_type.startswith("B-"):
                # Beginning of a new entity
                entity_type = entity_type[2:]  # Extract entity type
                if entity_type != current_entity_type:
                    current_entity_type = entity_type
                    # continue
                    bio_predictions.append((token, "B-" + current_entity_type))
                elif current_entity_type == entity_type:
                    bio_predictions.append((token, "I-" + current_entity_type))
                    # continue

            elif entity_type.startswith("I-"):
                # Inside an entity
                if current_entity_type == entity_type[2:]:
                    # Inside the same entity type
                    bio_predictions.append((token, "I-" + current_entity_type))
                else:
                    # Inside a different entity type, treat as beginning of a new entity
                    current_entity_type = entity_type[2:]
                    bio_predictions.append((token, "B-" + current_entity_type))
            else:
                # Outside an entity
                current_entity_type = None
                bio_predictions.append((token, "O"))

        return bio_predictions


    def join_bio_tags(predictions):
        joined_predictions = []
        current_entity = None
        current_word = ""

        for token, entity_type in predictions:
            if entity_type.startswith("B-"):
                # Beginning of a new entity
                if current_entity:
                    joined_predictions.append((current_word, current_entity))
                    current_word = ""
                current_entity = entity_type[2:]
                current_word += token
            elif entity_type.startswith("I-"):
                # Inside an entity
                current_word += " " + token
            else:
                # Outside an entity
                if current_entity:
                    joined_predictions.append((current_word, current_entity))
                    current_entity = None
                    current_word = ""
                else:
                    joined_predictions.append((token, "O"))

        # Add the last entity if it exists
        if current_entity:
            joined_predictions.append((current_word, current_entity))

        return joined_predictions


    # Example usage
    # text = '''Major crops grown in India are rice, wheat, millets, pulses, tea, coffee, sugarcane, oil seeds, cotton and jute, etc. of canal irrigation and tubewells have made it possible to grow rice in areas of less rainfall such as Punjab, Haryana and western Uttar Pradesh and parts of Rajasthan.'''
    predictions = predict_entities(text)
    bio_predictions = convert_to_bio(predictions)
    joined_predictions = join_bio_tags(bio_predictions)

    # print()
    # Display the results
    data = []
    for word, label in joined_predictions:
        if label != 'O':  # Only display tokens that are labeled with named entities
            # print(f"{word}: {label}")
            data.append([word, label])

    new_data = []
    for data in data:
        name = data[0]
        if not name[-1].isalpha():
            # name = re.findall(r'[a-zA-Z0-9 ]', name)
            pattern = re.compile(r'[^a-zA-Z0-9 ]')
            # Substitute the matched special characters with an empty string
            name = pattern.sub('', name)

        new_data.append([name, data[1]])

    return new_data
