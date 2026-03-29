import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📊 AI-Powered Financial Sentiment Analyzer")

st.write("Analyze whether financial news is bullish or bearish")

# Input
user_input = st.text_area("Enter financial news:")

# Button
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(input_vector)[0]

        # Confidence score
        probabilities = model.predict_proba(input_vector)
        confidence = max(probabilities[0]) * 100

        if prediction == "positive":
            st.success(f"📈 Positive Sentiment (Bullish)\nConfidence: {confidence:.2f}%")
        else:
            st.error(f"📉 Negative Sentiment (Bearish)\nConfidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text")