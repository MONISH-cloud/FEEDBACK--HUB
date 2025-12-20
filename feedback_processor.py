#!/usr/bin/env python3
"""
Python script to process raw feedbacks and classify them
Input: raw_feedbacks.txt
Output: classified_results.txt
"""

import re
from typing import List, Tuple
import spacy

# Load spaCy model (you might need to install it: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

def classify_sentiment(text: str) -> str:
    """Classify text sentiment using rule-based approach"""
    text_lower = text.lower()
    
    positive_words = {'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic', 'perfect'}
    negative_words = {'broken', 'poor', 'bad', 'terrible', 'awful', 'horrible', 'slow'}
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

def extract_aspects(text: str) -> List[str]:
    """Extract key aspects from text using spaCy NER and rule-based matching"""
    aspects = set()
    
    # Rule-based aspect detection
    aspect_keywords = {
        'Infrastructure': ['road', 'pipe', 'building', 'bridge', 'street', 'highway'],
        'Customer Service': ['service', 'staff', 'employee', 'representative', 'support'],
        'Environment': ['recycling', 'garbage', 'trash', 'pollution', 'clean', 'green'],
        'Transportation': ['bus', 'train', 'transit', 'transport', 'traffic', 'parking'],
        'Utilities': ['water', 'power', 'electricity', 'gas', 'internet', 'utility']
    }
    
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            aspects.add(aspect)
    
    # Use spaCy for more advanced aspect extraction if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "FAC", "GPE", "PRODUCT"]:
                aspects.add(ent.text)
    
    return list(aspects) if aspects else ["General"]

def process_feedbacks(input_file: str = "raw_feedbacks.txt", output_file: str = "classified_results.txt"):
    """Main function to process feedbacks and generate classified results"""
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please export data from the web app first.")
        return
    
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse the input format: ID|Organization|Urgency|Text
        parts = line.split('|')
        if len(parts) < 4:
            print(f"Warning: Skipping malformed line: {line}")
            continue
            
        feedback_id = parts[0]
        organization = parts[1]
        urgency = parts[2]
        text = '|'.join(parts[3:])  # Rejoin in case text contains pipes
        
        # Classify the feedback
        sentiment = classify_sentiment(text)
        aspects = extract_aspects(text)
        
        # Format: ID|Sentiment|Aspects (comma-separated)
        result_line = f"{feedback_id}|{sentiment}|{','.join(aspects)}"
        results.append(result_line)
        
        print(f"Processed: {feedback_id} -> {sentiment} [{', '.join(aspects)}]")
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"\n✅ Successfully processed {len(results)} feedbacks")
    print(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    process_feedbacks()