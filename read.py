# read.py - Automated classified output reader for web app integration
import os
import json
from datetime import datetime

def read_raw_feedbacks():
    """
    Read the raw feedbacks to get the original text
    """
    raw_file_path = r"C:\Users\91959\Downloads\raw_texts.txt"
    feedbacks_dict = {}
    
    if os.path.exists(raw_file_path):
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            feedback_id = parts[0].strip()
                            organization = parts[1].strip()
                            urgency = parts[2].strip()
                            feedback_text = '|'.join(parts[3:]).strip()
                            feedbacks_dict[feedback_id] = {
                                'organization': organization,
                                'urgency': urgency,
                                'text': feedback_text
                            }
            print(f"✅ Read {len(feedbacks_dict)} raw feedbacks")
        except Exception as e:
            print(f"❌ Error reading raw feedbacks: {e}")
    
    return feedbacks_dict

def read_classified_output():
    """
    Read classified output and combine with raw feedbacks
    Returns JSON data for web app
    """
    classified_path = r"C:\Users\91959\Downloads\classified_output.txt"
    raw_feedbacks = read_raw_feedbacks()
    
    combined_data = []
    
    if not os.path.exists(classified_path):
        print("❌ Classified output file not found")
        return {"error": "Classified output file not found", "data": []}
    
    try:
        with open(classified_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        valid_count = 0
        
        for line_number, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            
            if len(parts) < 3:
                continue
            
            # Extract classification components
            feedback_id = parts[0].strip()
            sentiment = parts[1].strip()
            aspects_text = parts[2].strip()
            aspects = [aspect.strip() for aspect in aspects_text.split(',') if aspect.strip()]
            
            # Create combined entry
            entry = {
                'id': feedback_id,
                'sentiment': sentiment,
                'aspects': aspects,
                'timestamp': datetime.now().isoformat(),
                'userId': 'system_import',
                'imported': True
            }
            
            # Add original feedback data if available
            if feedback_id in raw_feedbacks:
                original_data = raw_feedbacks[feedback_id]
                entry['text'] = original_data['text']
                entry['targetOrganization'] = original_data['organization']
                entry['urgency'] = original_data['urgency']
                entry['hasOriginalText'] = True
            else:
                entry['text'] = "Classification data imported - original text not found"
                entry['targetOrganization'] = "Unknown Organization"
                entry['urgency'] = "Medium"
                entry['hasOriginalText'] = False
            
            combined_data.append(entry)
            valid_count += 1
        
        print(f"✅ Processed {valid_count} classified entries")
        return {"success": True, "count": valid_count, "data": combined_data}
        
    except Exception as e:
        print(f"❌ Error processing classified output: {e}")
        return {"error": str(e), "data": []}

def export_for_web_app():
    """
    Main function to export data in web app format
    """
    print("🚀 Starting automated import process...")
    print("📁 Reading files from C:\\Users\\91959\\Downloads\\")
    
    result = read_classified_output()
    
    # Save to a temporary JSON file for web app to read
    output_file = "imported_data.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Data exported to {output_file}")
        print(f"📊 Total entries: {result.get('count', 0)}")
        
        # Print sample of what was exported
        if result.get('data'):
            print(f"📝 Sample entry: {result['data'][0]['id']} - {result['data'][0]['text'][:50]}...")
            
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
    
    return result

if __name__ == "__main__":
    # Run the import process and export data
    result = export_for_web_app()
    
    # Make sure we return proper exit code
    if result.get('success'):
        exit(0)
    else:
        exit(1)