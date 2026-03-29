import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data.csv")

# Train model automatically
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# UI
st.title("📊 AI-Powered Financial Sentiment Analyzer")

st.write("Analyze whether financial news is bullish or bearish")

user_input = st.text_area("Enter financial news:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        prob = model.predict_proba(input_vec)
        confidence = max(prob[0]) * 100

        if prediction == "positive":
            st.success(f"📈 Bullish (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"📉 Bearish (Confidence: {confidence:.2f}%)")
    else:
        st.warning("Please enter text")
