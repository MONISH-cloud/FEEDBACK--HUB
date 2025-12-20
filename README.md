# FEEDBACK--HUB
FeedHub is an NLP-based Aspect-Based Sentiment Analysis system that extracts service aspects from citizen feedback and classifies sentiment for each aspect. Using TF-IDF, Logistic Regression, and SVC, it converts unstructured text into actionable insights beyond traditional sentiment analysis.

FeedHub is a machine learning–based Aspect-Based Sentiment Analysis (ABSA) system designed to extract actionable insights from citizen feedback. Unlike traditional sentiment analysis models that assign a single sentiment label to an entire feedback entry, FeedHub identifies what users are talking about (aspects) and how they feel about each aspect.

The system processes unstructured textual feedback and produces structured outputs in the form of Aspect : Sentiment pairs, along with an overall sentiment summary. This enables organizations and public service authorities to pinpoint specific problem areas such as staff behaviour, waiting time, or service quality, rather than relying on coarse-grained sentiment labels.

🔍 Key Features

-Aspect-Based Sentiment Analysis (ABSA) for fine-grained feedback interpretation
-TF–IDF vectorization for robust text feature extraction
-Logistic Regression for aspect detection
-Support Vector Classifier (SVC) for sentiment classification
-Random Oversampling to handle sentiment class imbalance
-Handles multi-aspect and mixed-sentiment feedback effectively
-Outputs structured, decision-ready results
