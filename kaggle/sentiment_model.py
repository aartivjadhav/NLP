# sentiment_model.py

import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Path to dataset
CSV_PATH = "kaggle/data/1429_1.csv"  # adjust if file name is different

# Load the data
def load_data():
    df = pd.read_csv(CSV_PATH, encoding='latin1')
    # print(df.columns)
    df = df[['reviews.text', 'reviews.rating']].dropna()
    df.columns = ['review', 'rating']
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

    # Balance the classes by downsampling positives
    pos = df[df['sentiment'] == 1]
    neg = df[df['sentiment'] == 0]

    pos_downsampled = pos.sample(len(neg), random_state=42)
    df_balanced = pd.concat([pos_downsampled, neg]).sample(frac=1, random_state=42)  # Shuffle

    print("✅ Balanced dataset:")
    # print(df_balanced['sentiment'].value_counts())   

    return df_balanced


# Clean the review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    # print(text)
    return text

# Prepare text and labels
def preprocess(df):
    df['cleaned'] = df['review'].apply(clean_text)
    X = df['cleaned']
    y = df['sentiment']
    # pd.set_option('display.max_colwidth', None)
    # print(df[['review','cleaned','sentiment']].sample(5))
    return X, y

# Train and evaluate the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, tfidf

# Save the model and vectorizer
def save_model(model, vectorizer):
    joblib.dump(model, "kaggle/sentiment_model.pkl")
    joblib.dump(vectorizer, "kaggle/tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer saved.")

def main():
    print("📥 Loading data...")
    df = load_data()

    print("🧹 Cleaning text...")
    X, y = preprocess(df)
    # print(X,y)

    print("🤖 Training model...")
    model, tfidf = train_model(X, y)

    print("💾 Saving artifacts...")
    save_model(model, tfidf)

    print("✅ All done. Your model is ready!")

if __name__ == "__main__":
    main()
