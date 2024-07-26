import joblib
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Sample training data
train_reviews = [
    "This movie was fantastic! I loved every moment of it.",
    "The acting was terrible and the plot was boring.",
    "I'm not sure how I feel about this movie. It was okay, I guess.",
    "An absolute masterpiece. A must-watch for all movie lovers.",
]
train_labels = [1, 0, 0, 1]  # 1 for positive, 0 for negative

# Preprocess function
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


train_reviews = [preprocess_text(review) for review in train_reviews]

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Pipeline
pipeline = Pipeline([("tfidf", tfidf_vectorizer), ("clf", RandomForestClassifier())])

# Hyperparameter tuning
param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
grid_search.fit(train_reviews, train_labels)

best_model = grid_search.best_estimator_

# Save the model and vectorizer
joblib.dump(best_model, "sentiment_analysis_model.pkl")

print("Model and vectorizer saved successfully.")
