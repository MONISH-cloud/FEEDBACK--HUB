# train_simple_aspect_model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re

# Load your data
df = pd.read_csv("Laptop_Train_v2.csv", encoding='utf-8')
print("📂 Data loaded successfully.")

# Prepare training data - Convert to binary classification
# We'll create a model that identifies if a word/phrase is an aspect or not
def prepare_training_data(df):
    sentences = []
    labels = []
    
    for _, row in df.iterrows():
        sentence = str(row['Sentence']).lower()
        aspect = str(row['Aspect Term']).lower()
        
        # Create positive example (aspect)
        sentences.append(aspect)
        labels.append(1)  # 1 means "is aspect"
        
        # Create negative examples (non-aspect words from the sentence)
        words = re.findall(r'\b\w+\b', sentence)
        non_aspect_words = [w for w in words if w not in aspect and len(w) > 2]
        
        # Add some non-aspect words as negative examples
        for word in non_aspect_words[:3]:  # Take up to 3 non-aspect words
            sentences.append(word)
            labels.append(0)  # 0 means "not aspect"
    
    return sentences, labels

print("🛠️ Preparing training data...")
X, y = prepare_training_data(df)
print(f"Training samples: {len(X)}")
print(f"Aspect examples: {sum(y)}, Non-aspect examples: {len(y)-sum(y)}")

# Create and train a simple classifier
print("🚀 Training aspect detection model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])

model.fit(X, y)
print("✅ Model training complete!")

# Save the model
joblib.dump(model, 'aspect_detector_model.pkl')
print("💾 Model saved as 'aspect_detector_model.pkl'")

# Function to extract aspects from new text
def extract_aspects(text, model, threshold=0.7):
    words = re.findall(r'\b\w+\b', text.lower())
    aspects = []
    
    # Check each word and combinations
    for i, word in enumerate(words):
        if len(word) < 3:  # Skip very short words
            continue
            
        # Check single words
        prob = model.predict_proba([word])[0][1]
        if prob > threshold:
            aspects.append(word)
        
        # Check 2-word phrases
        if i < len(words) - 1:
            phrase = f"{word} {words[i+1]}"
            prob_phrase = model.predict_proba([phrase])[0][1]
            if prob_phrase > threshold:
                aspects.append(phrase)
    
    return list(set(aspects))  # Remove duplicates

# Test the model
print("\n🔍 Testing the model:")
test_sentences = [
    "The super computer has low performance",
    "This computer has great performance and fast processing speed",
    "The display quality is poor and the speakers are too quiet"
]

for sentence in test_sentences:
    aspects = extract_aspects(sentence, model)
    print(f"📝 Sentence: {sentence}")
    print(f"✅ Extracted aspects: {aspects}")
    print("-" * 60)