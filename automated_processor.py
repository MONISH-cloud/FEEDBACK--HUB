#!/usr/bin/env python3
"""
Automated Feedback Processor
Reads: accumulated_feedbacks.txt
Writes: classification_results.txt
"""

def process_feedbacks():
    try:
        with open('accumulated_feedbacks.txt', 'r') as f:
            lines = f.readlines()
        
        results = []
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                feedback_id = parts[0]
                # Your classification logic here
                sentiment = "Positive"  # Replace with actual ML
                aspects = "General"     # Replace with actual extraction
                results.append(f"{feedback_id}|{sentiment}|{aspects}")
        
        with open('classification_results.txt', 'w') as f:
            f.write('\n'.join(results))
            
        print(f"Processed {len(results)} feedbacks")
        
    except FileNotFoundError:
        print("No accumulated_feedbacks.txt found - export data from web app first")

if __name__ == "__main__":
    process_feedbacks()