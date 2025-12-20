import joblib
import pandas as pd
from preprocess import clean_text 
import nltk
# Ensure NLTK resources are available to prevent silent hangs
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# Print statement to confirm the script starts execution
print("--- Starting Sentiment Prediction Script ---", flush=True)

# Define file paths
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

def predict_sentiment(text_list):
    """
    Loads the trained model and vectorizer, cleans the input text,
    and returns sentiment predictions.
    """
    try:
        # Load the model and vectorizer
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model and Vectorizer loaded successfully.")
    except FileNotFoundError:
        print("⚠️ File Not Found Error: Ensure 'models' directory and files are correct.")
        return None
    except Exception as e:
        # This will catch corruption, version incompatibility, or other loading errors
        print(f"❌ CRITICAL LOADING ERROR! Saved file is incompatible or corrupted.")
        print(f"Details: {e}")
        return None 

    # --- MISSING CODE BLOCK START (Text Processing & Prediction) ---

    # 1. Clean the input text
    # Apply the same cleaning function used during training
    cleaned_text = [clean_text(str(text)) for text in text_list]

    # 2. Transform text to features
    # Use the *saved* vectorizer to transform the new text
    text_features = vectorizer.transform(cleaned_text)

    # 3. Make predictions
    predictions = model.predict(text_features)
    
    return predictions

    # --- MISSING CODE BLOCK END ---

# --- CRITICAL MISSING BLOCK START (Main Execution) ---
if __name__ == "__main__":
    # Example feedback to test your model
    new_feedback = [
        "The company's new policy is absolutely brilliant and will drive growth.",
        "The product arrived damaged and the customer service was awful.",
         "This computer has amazing performance and fast processing speed",
        "Management announced no major changes today, keeping everything status quo.",
        "I am extremely happy with the performance of the stock this quarter."
    ]

    results = predict_sentiment(new_feedback)
    
    if results is not None:
        print("\n--- Sentiment Prediction Results ---")
        for text, prediction in zip(new_feedback, results):
            print(f"Text: '{text[:50]}...'")
            print(f"Sentiment: {prediction}\n")

# --- CRITICAL MISSING BLOCK END ---