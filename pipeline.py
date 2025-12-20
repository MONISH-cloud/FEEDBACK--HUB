# pipeline.py
import os
import re
import joblib
import pandas as pd
from textblob import TextBlob

# Import your trained sentiment model
try:
    from predict import predict_sentiment
except ImportError:
    print("⚠️  predict_sentiment not available, using fallback methods")

# ======================================
# 🔧 MODEL LOADING
# ======================================
def load_aspect_model():
    """Load the trained aspect detection model."""
    model_path = "aspect_detector_model.pkl"
    if not os.path.exists(model_path):
        print("❌ aspect_detector_model.pkl not found. Run 'train_asc.py' first.")
        return None
    try:
        model = joblib.load(model_path)
        print("✅ Aspect model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load aspect model: {e}")
        return None


# ======================================
# 🧠 ENHANCED ASPECT EXTRACTION
# ======================================
def extract_aspects(text, model):
    """Extract possible aspects from feedback text using model + expanded vocabulary."""
    if model is None:
        return []

    text_lower = text.lower()
    aspects = set()

    # Expanded aspect vocabulary with categories
    aspect_categories = {
        'power supply': ['power', 'battery', 'charging', 'adapter', 'power supply', 'electricity', 'outage', 'blackout'],
        'processor': ['processor', 'cpu', 'speed', 'performance', 'chip', 'core', 'intel', 'amd', 'ryzen'],
        'display': ['screen', 'display', 'monitor', 'resolution', 'brightness', 'colors', 'pixel', 'hd', '4k'],
        'keyboard': ['keyboard', 'keys', 'typing', 'backlight', 'keypad'],
        'storage': ['storage', 'hard disk', 'hdd', 'ssd', 'memory', 'capacity', 'gb', 'tb'],
        'ram': ['ram', 'memory', 'ddr', '8gb', '16gb', '32gb'],
        'software': ['software', 'os', 'windows', 'linux', 'application', 'app', 'program'],
        'network': ['wifi', 'internet', 'connection', 'bluetooth', 'ethernet', 'wireless'],
        'camera': ['camera', 'webcam', 'photo', 'video', 'recording', 'megapixel'],
        'audio': ['sound', 'speaker', 'audio', 'volume', 'microphone', 'headphone', 'jack'],
        'heating': ['heating', 'temperature', 'cooling', 'fan', 'overheat', 'thermal'],
        'build': ['build', 'design', 'material', 'plastic', 'metal', 'body', 'durability'],
        'service': ['service', 'support', 'customer care', 'help', 'technician'],
        'delivery': ['delivery', 'shipping', 'packaging', 'courier', 'dispatch'],
        'price': ['price', 'cost', 'expensive', 'affordable', 'budget', 'money', 'value']
    }

    # Check for aspect categories in text
    for category, keywords in aspect_categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspects.add(category)
                break

    # Use ML model for additional aspect detection
    words = re.findall(r'\b\w+\b', text_lower)
    
    for i, word in enumerate(words):
        if len(word) < 3:
            continue

        # Single word check with model
        try:
            if model.predict_proba([word])[0][1] > 0.6:  # Lowered threshold
                aspects.add(word)
        except Exception:
            pass

        # Two-word phrase check
        if i < len(words) - 1:
            phrase = f"{word} {words[i + 1]}"
            try:
                if model.predict_proba([phrase])[0][1] > 0.6:
                    aspects.add(phrase)
            except Exception:
                pass

    return list(aspects)


# ======================================
# 💬 ENHANCED SENTIMENT ANALYSIS
# ======================================
def enhanced_sentiment_analysis(text):
    """Use the model first, then rule-based + TextBlob fallback with improved logic."""
    
    # Try the trained model first
    try:
        if 'predict_sentiment' in globals():
            prediction = predict_sentiment([text])[0]
            if prediction and prediction != "neutral":
                return prediction
    except Exception as e:
        print(f"⚠️  Sentiment model failed: {e}")

    # Enhanced rule-based sentiment analysis
    text_lower = text.lower()
    
    # Expanded positive words
    positive_words = {
        "good", "great", "excellent", "love", "amazing", "perfect", "best", "awesome",
        "fast", "comfortable", "smooth", "reliable", "easy", "happy", "satisfied",
        "working", "nice", "fine", "okay", "decent", "acceptable", "proper", "correct",
        "fixed", "resolved", "helpful", "quick", "efficient", "worth", "valuable",
        "improved", "better", "fantastic", "wonderful", "outstanding", "superb",
        "pleased", "content", "impressed", "recommend", "liked", "enjoyed"
    }
    
    # Expanded negative words with stronger weight
    negative_words = {
        "bad", "poor", "terrible", "hate", "slow", "noisy", "broken", "worst", "problem",
        "issue", "expensive", "faulty", "disappointing", "damaged", "not working", 
        "failed", "failure", "defective", "useless", "waste", "rubbish", "trash",
        "horrible", "awful", "pathetic", "unacceptable", "disgusting", "annoying",
        "frustrating", "angry", "upset", "displeased", "unhappy", "regret",
        "complaint", "sucks", "pathetic", "junk", "scam", "fraud", "cheat"
    }
    
    # Negation words that flip sentiment
    negation_words = {"not", "no", "never", "nothing", "none", "without", "cannot"}
    
    # Count sentiment words with negation handling
    pos_count = 0
    neg_count = 0
    
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in negation_words:
            # Skip next word for negation context
            continue
            
        if word in positive_words:
            # Check if previous word was a negation
            if i > 0 and words[i-1] in negation_words:
                neg_count += 1
            else:
                pos_count += 1
                
        elif word in negative_words:
            # Check if previous word was a negation
            if i > 0 and words[i-1] in negation_words:
                pos_count += 1
            else:
                neg_count += 1
    
    # Strong negative indicators
    strong_negative_phrases = {
        "not working", "not good", "no good", "never again", "worst experience",
        "bad experience", "poor quality", "waste of money", "don't buy", "do not buy"
    }
    
    for phrase in strong_negative_phrases:
        if phrase in text_lower:
            neg_count += 2
    
    # Decision logic
    if neg_count > pos_count:
        return "negative"
    elif pos_count > neg_count:
        return "positive"
    
    # Fallback to TextBlob with adjusted thresholds
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.05:  # Lowered threshold
            return "positive"
        elif polarity < -0.05:  # Lowered threshold
            return "negative"
    except Exception as e:
        print(f"⚠️  TextBlob failed: {e}")
    
    return "neutral"


def get_aspect_sentiment(text, aspect):
    """Determine sentiment for a specific aspect using context words."""
    text_lower = text.lower()
    aspect_lower = aspect.lower()
    
    if aspect_lower not in text_lower:
        return "neutral"

    words = text_lower.split()
    aspect_words = aspect_lower.split()

    # Find aspect position in text
    for i in range(len(words) - len(aspect_words) + 1):
        if words[i:i + len(aspect_words)] == aspect_words:
            # Get context around the aspect
            start = max(0, i - 5)
            end = min(len(words), i + len(aspect_words) + 5)
            context = words[start:end]
            context_text = " ".join(context)

            # Enhanced positive words for aspect-specific sentiment
            pos_words = {
                "good", "great", "excellent", "amazing", "awesome", "wonderful", 
                "fantastic", "perfect", "best", "beautiful", "nice", "love", 
                "happy", "satisfied", "pleased", "comfortable", "fast", "smooth", 
                "reliable", "easy", "bright", "efficient", "improved", "enjoyable",
                "safe", "correct", "success", "working", "stable", "clean", "light", 
                "strong", "fun", "quick", "friendly", "responsive", "helpful", 
                "effective", "innovative", "intuitive", "clear", "professional", 
                "flexible", "powerful", "rich", "durable", "soft", "modern", "healthy",
                "logical", "smart", "optimal", "convenient", "affordable", "trustworthy",
                "secure", "lightweight", "efficient", "positive", "beneficial", "advanced"
            }
            
            # Enhanced negative words for aspect-specific sentiment
            neg_words = {
                "bad", "poor", "terrible", "awful", "horrible", "worst", "hate", 
                "disappointing", "uncomfortable", "slow", "noisy", "broken", "damaged", 
                "faulty", "problem", "issue", "error", "crash", "freeze", "lag", 
                "unreliable", "expensive", "overpriced", "cheap", "dim", "heavy", 
                "hard", "weak", "stiff", "confusing", "annoying", "unhelpful", 
                "incorrect", "inefficient", "frustrating", "messy", "ugly", "unpleasant", 
                "inaccurate", "dangerous", "insecure", "slowly", "worse", "unhappy", 
                "difficult", "laggy", "useless", "inconsistent", "boring", "painful", 
                "negative", "disadvantage", "fail", "failure", "subpar", "glitch", 
                "buggy", "unfit", "unsatisfied", "problematic", "unresponsive", 
                "mediocre", "unreliable", "defective", "junk", "garbage"
            }
            
            # Negation handling for aspect context
            negation_words = {"not", "no", "never", "nothing", "none", "without"}
            
            pos_count = 0
            neg_count = 0
            context_words = context_text.split()
            
            for j, word in enumerate(context_words):
                if word in negation_words:
                    continue
                    
                if word in pos_words:
                    if j > 0 and context_words[j-1] in negation_words:
                        neg_count += 1
                    else:
                        pos_count += 1
                elif word in neg_words:
                    if j > 0 and context_words[j-1] in negation_words:
                        pos_count += 1
                    else:
                        neg_count += 1

            # Decision with confidence
            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            
            return "neutral"

    return "neutral"


def calculate_overall_sentiment(aspect_sentiments, text_sentiment):
    """Calculate overall sentiment based on aspect sentiments with intelligent logic."""
    if not aspect_sentiments:
        return text_sentiment
    
    # Count aspect sentiments
    pos_count = sum(1 for aspect in aspect_sentiments if aspect['sentiment'] == 'positive')
    neg_count = sum(1 for aspect in aspect_sentiments if aspect['sentiment'] == 'negative')
    neu_count = sum(1 for aspect in aspect_sentiments if aspect['sentiment'] == 'neutral')
    
    total_aspects = len(aspect_sentiments)
    
    # Decision logic based on your requirements
    if total_aspects == 1:
        # Single aspect - use its sentiment directly
        return aspect_sentiments[0]['sentiment']
    
    elif total_aspects == 2:
        if neg_count == 2:
            return "negative"
        elif pos_count == 2:
            return "positive"
        else:
            return "neutral"  # One positive + one negative/neutral
    
    elif total_aspects >= 3:
        if neg_count >= 2 and neg_count > pos_count:
            return "negative"
        elif pos_count >= 2 and pos_count > neg_count:
            return "positive"
        elif neg_count == pos_count:
            return "neutral"
        else:
            # If we have mixed sentiments but no clear majority
            if neg_count > 0 and pos_count > 0:
                return "neutral"
            elif neg_count > 0:
                return "negative"
            elif pos_count > 0:
                return "positive"
    
    # Fallback to original text sentiment
    return text_sentiment


# ======================================
# 📥 FILE HANDLING
# ======================================
def read_input_file(file_path):
    """Read feedbacks from the raw_texts.txt file."""
    if not os.path.exists(file_path):
        print(f"❌ Input file not found: {file_path}")
        return []

    feedbacks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|", 3)
            if len(parts) != 4:
                print(f"⚠️  Skipping malformed line {line_num}: {line}")
                continue

            feedbacks.append({
                "id": parts[0].strip(),
                "organization": parts[1].strip(),
                "urgency": parts[2].strip(),
                "text": parts[3].strip(),
            })

    print(f"✅ Loaded {len(feedbacks)} valid feedbacks.")
    return feedbacks


def write_output_file(results, output_path):
    """Write final classified output to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                if r["aspects"]:
                    aspect_str = ",".join(f"{a['aspect']}:{a['sentiment']}" for a in r["aspects"])
                else:
                    aspect_str = "General:neutral"
                f.write(f"{r['id']}|{r['overall_sentiment']}|{aspect_str}\n")

        print(f"✅ Results saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing output: {e}")
        return False


# ======================================
# ⚙️ COMPLETE PIPELINE
# ======================================
def analyze_pipeline(feedbacks):
    """Run the complete classification pipeline."""
    aspect_model = load_aspect_model()
    
    results = []
    for idx, fb in enumerate(feedbacks, 1):
        text = fb["text"]
        print(f"\n[{idx}/{len(feedbacks)}] Processing ID {fb['id']}...")
        print(f"   → Text: {text[:90]}{'...' if len(text) > 90 else ''}")

        aspects = extract_aspects(text, aspect_model)
        print(f"   🧩 Aspects found: {aspects or 'None'}")

        # Get text-based sentiment first
        text_sentiment = enhanced_sentiment_analysis(text)
        print(f"   💬 Text sentiment: {text_sentiment}")

        # Get aspect sentiments
        aspect_sentiments = [
            {"aspect": a, "sentiment": get_aspect_sentiment(text, a)} for a in aspects
        ]

        # Calculate final overall sentiment based on aspect logic
        overall_sentiment = calculate_overall_sentiment(aspect_sentiments, text_sentiment)
        print(f"   🎯 Overall sentiment: {overall_sentiment}")

        for a in aspect_sentiments:
            print(f"      - {a['aspect']}: {a['sentiment']}")

        results.append({
            **fb,
            "overall_sentiment": overall_sentiment,
            "aspects": aspect_sentiments,
            "text_sentiment": text_sentiment  # Keep for reference
        })

    return results


# ======================================
# 🚀 MAIN ENTRY
# ======================================
def main():
    print("=" * 65)
    print("🤖 AUTOMATED CITIZEN FEEDBACK CLASSIFICATION PIPELINE")
    print("=" * 65)

    downloads = os.path.expanduser("~/Downloads")
    input_path = os.path.join(downloads, "raw_texts.txt")
    output_path = os.path.join(downloads, "classified_output.txt")

    feedbacks = read_input_file(input_path)
    if not feedbacks:
        print("🚫 No valid feedbacks to process.")
        return

    results = analyze_pipeline(feedbacks)
    if not results:
        print("🚫 No results generated.")
        return

    write_output_file(results, output_path)

    # 📊 Enhanced Summary
    sentiments = {}
    total_aspects = sum(len(r["aspects"]) for r in results)
    aspect_types = {}

    for r in results:
        sentiments[r["overall_sentiment"]] = sentiments.get(r["overall_sentiment"], 0) + 1
        
        # Count aspect types
        for aspect_data in r["aspects"]:
            aspect = aspect_data["aspect"]
            aspect_types[aspect] = aspect_types.get(aspect, 0) + 1

    print("\n📈 ENHANCED SUMMARY REPORT")
    print("-" * 40)
    print(f"Total Feedbacks: {len(results)}")
    print(f"Total Aspects Detected: {total_aspects}")
    print("Sentiment Distribution:")
    for s, c in sentiments.items():
        print(f"  • {s.capitalize()}: {c} ({(c / len(results)) * 100:.1f}%)")
    
    if aspect_types:
        print("\nTop Aspects Detected:")
        for aspect, count in sorted(aspect_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {aspect}: {count} times")

    print("\n✅ Processing complete!")
    print(f"📁 Output saved at: {output_path}")


if __name__ == "__main__":
    main()