# ner_extraction.py

"""
NER Extraction using spaCy
--------------------------
This script demonstrates:
1. Loading a dataset (unstructured text like job postings or scientific articles).
2. Using spaCy's pre-trained Named Entity Recognition (NER) model.
3. Extracting structured information (e.g., Person-Organization table).
"""

import spacy
import pandas as pd

# -------------------------------
# Load pre-trained spaCy model
# -------------------------------
# Download model first if not installed: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Example dataset (replace with your dataset)
# -------------------------------
documents = [
    "Google is hiring Data Scientists in Bangalore. Contact John Doe for more details.",
    "Dr. Alice Johnson published a paper on Machine Learning at Stanford University.",
    "Microsoft appointed Satya Nadella as the CEO.",
    "OpenAI collaborated with Elon Musk and Sam Altman in the early years."
]

# -------------------------------
# Perform Named Entity Recognition
# -------------------------------
entities = []

for doc in documents:
    parsed_doc = nlp(doc)
    for ent in parsed_doc.ents:
        entities.append([doc, ent.text, ent.label_])

# Store results in DataFrame
df_entities = pd.DataFrame(entities, columns=["Document", "Entity", "Label"])

print("===== Named Entities Detected =====")
print(df_entities)

# -------------------------------
# Extract structured information: Person ↔ Organization
# -------------------------------
person_org_relations = []

for doc in documents:
    parsed_doc = nlp(doc)
    persons = [ent.text for ent in parsed_doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in parsed_doc.ents if ent.label_ == "ORG"]

    for person in persons:
        for org in orgs:
            person_org_relations.append([person, org])

df_relations = pd.DataFrame(person_org_relations, columns=["Person", "Organization"])

print("\n===== Person ↔ Organization Table =====")
print(df_relations)
