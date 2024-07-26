from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline
nlp = pipeline("sentiment-analysis")


def predict_sentiment(text):
    result = nlp(text)
    return result[0]["label"]


# Test the new function
if __name__ == "__main__":
    sample_text = "In my opinion, the heroine was not good"
    print(predict_sentiment(sample_text))
