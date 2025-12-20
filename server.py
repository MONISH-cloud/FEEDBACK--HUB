from flask import Flask, jsonify
import subprocess
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/run-import', methods=['POST'])
def run_import():
    try:
        result = subprocess.run(['python', 'read.py'], capture_output=True, text=True)
        if result.returncode == 0:
            if os.path.exists('imported_data.json'):
                with open('imported_data.json', 'r') as f:
                    return jsonify(json.load(f))
        return jsonify({"error": "Import failed", "details": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000)