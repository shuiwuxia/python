import pandas as pd
import numpy as np
import re
import nltk
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

TEXT_COL = 'case_text'
LABEL_COL = 'case_outcome'
TARGET_CLASSES = ['cited', 'followed', 'referred to']

print("--- Phase 1: Multi-Class Labeling ---")
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))
print("--- Phase 1: Data Preparation ---")
try:
    df = pd.read_csv('gyaniproject.csv', low_memory=False) 
except FileNotFoundError:
    print("ERROR: File not found. Check your file name or directory.")
    exit()

df['HAS_CITATION'] = df[LABEL_COL].apply(
    lambda x: 1 if str(x).lower() in TARGET_CLASSES else 0
)
df[TEXT_COL] = df[TEXT_COL].fillna('')


def create_multi_class_label(outcome):
    outcome_lower = str(outcome).lower()
    if outcome_lower in TARGET_CLASSES:
        return outcome_lower
    else:
        return 'other' 

df['CITATION_TYPE'] = df[LABEL_COL].apply(create_multi_class_label)
X_text_raw = df[TEXT_COL]
y_multi = df['CITATION_TYPE']

print(f"Total documents: {len(df)}")
print(f"Multi-Class distribution:\n{y_multi.value_counts().to_markdown()}")
print("\n--- Phase 2 & 3: Cleaning & Feature Engineering ---")

X_text_cleaned = X_text_raw.apply(lambda x: x.lower()) 
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_features = tfidf_vectorizer.fit_transform(X_text_cleaned)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, 
    y_multi, 
    test_size=0.2, 
    random_state=42,
    stratify=y_multi 
)
print("Data Split using Multi-Class Labels.")
print("\n--- Phase 4: Multi-Class Model Training and Evaluation ---")

model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nFinal Multi-Class Model Evaluation (Classification Report):")
print(classification_report(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred, labels=y_multi.unique())
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual: {c}' for c in y_multi.unique()], 
                     columns=[f'Predicted: {c}' for c in y_multi.unique()])

print("\n--- Multi-Class Confusion Matrix ---")
print(cm_df.to_markdown(numalign="left", stralign="left"))
def demonstrate_model(text_input):
    """Takes raw text and predicts the citation type."""
    cleaned_input = tfidf_vectorizer.transform([text_input])
    prediction = model.predict(cleaned_input)[0]
    return prediction

def demonstrate_proba(text_input):
    """Return class probabilities for a given input text.

    Output is a dict mapping class label -> probability.
    """
    cleaned_input = tfidf_vectorizer.transform([text_input])
    proba = model.predict_proba(cleaned_input)[0]
    return dict(zip(model.classes_, proba))

example_text = "The court strictly followed the principle established in the previous case."
print(f"\n--- Model Demonstration ---")
print(f"Input: {example_text[:60]}...")
print(f"Predicted Citation Type: **{demonstrate_model(example_text).upper()}**")
