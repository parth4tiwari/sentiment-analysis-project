from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
import tf_keras

print("tf-keras version:", tf_keras.__version__)


from transformer_model import (
    predict_sentiment,
)  # Import the function for transformer model

app = Flask(__name__)

# Paths to the trained model and TF-IDF vectorizer
MODEL_PATH = "sentiment_analysis_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Load the trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
with open(VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bnot\b", "not_", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def preprocess_reviews(reviews):
    """Preprocess the reviews using the preprocess_text function."""
    return [preprocess_text(review) for review in reviews]


def extract_features(reviews):
    """Extract features from the reviews using the TF-IDF vectorizer."""
    return tfidf_vectorizer.transform(reviews).toarray()


def predict_sentiments(reviews):
    """Predict sentiments for the given reviews using the trained model."""
    clean_reviews = preprocess_reviews(reviews)
    features = extract_features(clean_reviews)
    return model.predict(features)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    # Use the transformer model for prediction
    sentiment = predict_sentiment(text)

    return jsonify({"prediction": sentiment})


if __name__ == "__main__":
    app.run(debug=True)

print("Model file size:", os.path.getsize("sentiment_analysis_model.pkl"))
print("Vectorizer file size:", os.path.getsize("tfidf_vectorizer.pkl"))
