import pandas as pd
import random
import numpy as np
import streamlit as st
from textblob import TextBlob
from difflib import get_close_matches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("/content/india_recommendation_dataset (3).csv")

# Train ML model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Location"] + " " + df["Climate"] + " " + df["Age Group"])
y = df["Recommendation"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

def chatbot(location, age_group, sentiment_text, preference):
    if location not in df["Location"].values:
        closest_match = get_close_matches(location, df["City"].unique(), n=1)
        if closest_match:
            location = closest_match[0]
        else:
            return "Sorry, we don't have data for your location."
    
    climate = df[df["Location"] == location]["Climate"].values[0]
    sentiment = TextBlob(sentiment_text).sentiment.polarity
    
    # If preference is not provided, determine based on sentiment
    if not preference:
        preference = "wellness" if sentiment < -0.2 else "food" if sentiment < 0.2 else "travel"
    
    query_vector = vectorizer.transform([f"{location} {climate} {age_group}"])
    prediction = clf.predict(query_vector)[0]
    
    return f"For age group {age_group} in {location} ({climate} climate), we recommend: {prediction}."

# Streamlit UI
st.title("Personalized Recommendation Chatbot")

location = st.text_input("Enter your location:")
age_group = st.selectbox("Select your age group:", df["Age Group"].unique())
sentiment_text = st.text_area("Describe how you're feeling:")
preference = st.selectbox("Select your preference:", ["wellness", "food", "travel", "Auto Detect (Based on Sentiment)"])

if st.button("Get Recommendation"):
    if preference == "Auto Detect (Based on Sentiment)":
        preference = ""
    response = chatbot(location, age_group, sentiment_text, preference)
    st.write(response)
