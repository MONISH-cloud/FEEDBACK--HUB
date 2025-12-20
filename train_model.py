import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.svm import SVC
from preprocess import clean_text
from imblearn.over_sampling import RandomOverSampler # Import necessary for balancing

# 1. Load dataset
data = pd.read_csv('feedback.csv', encoding='latin-1', header=None, sep=',')

# Rename columns for clarity
data = data.rename(columns={0: 'label', 1: 'text'})
print("Actual Column Names:", data.columns.tolist())

# 2. Clean feedback text
data['text'] = data['text'].astype(str).apply(clean_text)

# 3. Split data 
# X is the text, y is the label
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], 
    data['label'], 
    test_size=0.2, 
    random_state=42
)

# 4. Convert text to numeric using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4b. BALANCE THE TRAINING DATA (The Improvement Step)
sampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_tfidf, y_train)

# 5. Train Support Vector Classifier (using balanced data)
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 6. Evaluate model (using the original, unbiased test data)
y_pred = model.predict(X_test_tfidf)
print("✅ Balanced Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report (After Balancing):\n", classification_report(y_test, y_pred))

# 7. Save model and vectorizer
try:
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("\n🎉 Model trained, saved, and balanced successfully!")
except Exception as e:
    print(f"\n⚠️ Error saving files. Did you create the 'models' directory? Error: {e}")