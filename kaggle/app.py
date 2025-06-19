# app.py

import streamlit as st
import joblib
import os
import re

# Load model and vectorizer
# MODEL_PATH = "NLP/kaggle/sentiment_model.pkl"
# VECTORIZER_PATH = "NLP/kaggle/tfidf_vectorizer.pkl"

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
st.write(os.path.exists(VECTORIZER_PATH))

@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    else:
        st.error("Model files not found. Please run sentiment_model.py first.")
        return None, None
    

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.title("📝 Sentiment Analyzer")
st.write("Enter a product review below and see if it's positive or negative.")

user_input = st.text_area("✍️ Write your review here:")

if st.button("Analyze"):
    st.write('Hello')
    model, vectorizer = load_artifacts()

    if model and vectorizer and user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][prediction]

        label = "✅ Positive" if prediction == 1 else "⚠️ Negative"
        st.subheader(f"Prediction: {label}")
        st.caption(f"Confidence: {prob:.2%}")
    elif not user_input.strip():
        st.warning("Please enter a review to analyze.")
