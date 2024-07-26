import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec


# Example function to train Word2Vec
def train_word2vec(reviews):
    tokenized_reviews = [review.split() for review in reviews]
    word2vec_model = Word2Vec(
        tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4
    )
    return word2vec_model


# Sample training data
train_reviews = [
    "This movie was fantastic! I loved every moment of it.",
    "The acting was terrible and the plot was boring.",
    "I'm not sure how I feel about this movie. It was okay, I guess.",
    "An absolute masterpiece. A must-watch for all movie lovers.",
]
train_labels = [1, 0, 0, 1]  # 1 for positive, 0 for negative

# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
train_features = tfidf_vectorizer.fit_transform(train_reviews)

# Train a logistic regression model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Save the model and vectorizer
joblib.dump(model, "sentiment_analysis_model.pkl")
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
