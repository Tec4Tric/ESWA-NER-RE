# Example input for GPT-3

import openai, os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
# Load the CSV dataset
data = pd.read_csv("Raw_Data.csv")  # Ensure the dataset includes 'sentence', 'entity1', 'entity2', and 'relation'

openai.api_key = "your_api_key"


#input_sentence = "The prospective climate change is global warming induced by greenhouse gases."
#entity1 = "climate change"
#entity2 = "global warming"

label_names = ['Coreference', 'Helps_In', 'Includes', 'Used_For', 'Synonym_Of', 'Seasonal', 'Origin_Of', 'Caused_By', 'Conjunction']
# Entities
all_entities = [
    "Agri_Pollution", "Agri_Process", "Agri_Waste", "Agri_Method",
    "Chemical", "Citation", "Crop", "Date_and_Time", "Disease",
    "Duration", "Event", "Field_Area", "Food_Item", "Fruit", "Humidity",
    "Location", "ML_Model", "Money", "Natural_Disaster", "Natural_Resource",
    "Nutrient", "Organism", "Organization", "Other", "Other_Quantity",
    "Person", "Policy", "Quantity", "Rainfall", "Season", "Soil", "Technology",
    "Temp", "Treatment", "Vegetable", "Weather"
]
'''You are an advanced NLP model trained to extract structured information from text. Based on the input paragraph, perform the following tasks:

1. *Named Entity Recognition (NER):* Identify and classify entities present in the text based on the given list of entity types.
   - *Entity Types:* {', '.join(entity_types)}

2. *Relation Extraction (RE):* Identify relationships between the extracted entities, from the predefined list of relationship types.
   - *Relationship Types:* {', '.join(relation_types)}

3. Output the results in a structured format:
   - For NER: Provide a list of identified entities with their types.
   - For RE: Provide triplets in the format <head entity, relation, tail entity>.

*Input Paragraph:*
{paragraph}

*Output Format:*

*Named Entities:*
- Entity: [Entity name], Type: [Entity type]

*Relations:*
- <Head entity, Relation, Tail entity>'''


def  generate_sentence(sentence, entities):
    prompt = f"""
        You are an advanced NLP model trained to extract structured information from text. Based on the input sentence, perform the following tasks:
        *Named Entity Recognition (NER):* Identify and classify entities present in the text based on the given list of [Entity Types]. Provide a list of identified entities with their types in the given [Output] format only. Strictly give only the answer.
         Entity Types: {', '.join(entities)}
    Input:
    Sentence: "{sentence}"
    Outout:
    Entity name, Entity type
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Replace with the model you're using
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response['choices'][0]['message']['content']



true_entities = []
pred_entities = []
for index, row in data.iterrows():
    sentence = row['generated_sentence']
    entity1 = row['entity1'].strip()
    entity1_type = row['head_label'].strip()
    entity2 = row['entity2'].strip()
    entity2_type = row['tail_label'].strip()
    # true_entities.append(entity1)
    true_entities.append(entity1_type)
    # true_entities.append(entity2)
    true_entities.append(entity2_type)
    f = generate_sentence(row,all_entities)
    # Append the true and predicted labels
    for item in  f.split('\n'):
        if item.isalpha():
            try:
                predicted_entities = item.strip()
                pred_entities.append(predicted_entities.split(",")[1].strip())
            except Exception:
                print(f)
                print(item)
        # a = [item.strip() for item in predicted_entities.split(',')]
        # pred_entities.append(predicted_entities)
    # predicted_entities = [item.strip() for item in f.split('\n')]

if len(true_entities) > len(pred_entities):
    pred_entities = pred_entities + ['O']* (len(true_entities)- len(pred_entities))
elif len(pred_entities) > len(true_entities):
    pred_entities = pred_entities[:len(true_entities)]

# pred_entities=pred_entities[:len(true_entities)]
print(len(pred_entities))
print(len(true_entities))
print(input("HI:"))
# Compute classification metrics
accuracy = accuracy_score(true_entities, pred_entities)
precision, recall, f1, _ = precision_recall_fscore_support(true_entities, pred_entities, average='weighted')

# Classification report
report = classification_report(true_entities, pred_entities)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n", report)
# Save classification report to a file
output_dir = "./metrics/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "ner_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

print(f"Metrics saved to {output_dir}classification_report.txt")
