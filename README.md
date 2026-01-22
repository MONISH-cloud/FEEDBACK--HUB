🧠 AI-Driven Sentiment & Aspect Analysis System

An end-to-end Machine Learning–based sentiment analysis system that classifies user text, extracts key aspects, and continuously improves using user feedback.
The project simulates a real-world ML pipeline — from data preprocessing and training to deployment, monitoring, and retraining.

🚀 Features

🔍 Sentiment Classification (Positive / Negative / Neutral) with confidence scores

🧩 Aspect-Based Analysis to identify important topics within text

🌐 Flask REST API for real-time predictions

🔁 User Feedback Loop to track model accuracy

🔄 Automated Model Retraining and logging

📊 Performance Monitoring using stored predictions and feedback

🏗️ Project Architecture
```text
AI-Sentiment-Analysis/
│
├── server.py                 # Flask application (API endpoints)
├── predict.py                # Loads trained model & handles predictions
├── train_model.py            # Model training and evaluation script
├── pipeline.py               # Core ML pipeline (TF-IDF + SVM)
├── preprocess.py             # Text preprocessing utilities
├── feedback_processor.py     # User feedback handling and storage
├── automated_processor.py    # Scheduled / automated retraining
├── read.py                   # Data reading utilities
│
├── templates/
│   └── index.html            # Frontend UI
│
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # Frontend JavaScript
│   └── assets/               # Images / icons (if any)
│
├── models/
│   └── sentiment_model.pkl   # Trained ML model
│
├── data/
│   ├── Laptop_Train_v2.csv   # Training dataset
│   └── feedback.csv          # Stored user feedback
│
├── logs/
│   ├── prediction_logs.jsonl # Prediction logs
│   └── training.log          # Model training logs
│
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```


🔄 Data Flow Overview

User enters text on the frontend

Text is sent to /predict API

Model returns sentiment + confidence

User submits feedback

Feedback is stored and analyzed

Model is periodically retrained using new data

⚙️ Tech Stack

Language: Python

ML: Scikit-learn (TF-IDF, Linear SVM)

Backend: Flask

Frontend: HTML, CSS, JavaScript

Data Handling: Pandas, CSV, JSON

Model Persistence: Joblib


📈 Future Improvements

Add deep learning models (LSTM / BERT)

Improve aspect extraction accuracy

Add dashboard for model performance

Dockerize the application

Deploy on cloud (AWS / GCP)

