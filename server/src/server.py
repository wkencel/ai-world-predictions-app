# app.py
import os
from flask import Flask, jsonify, request
from services.openai import generate_response
from dotenv import load_dotenv

load_dotenv()
PORT = os.getenv("PORT")

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Welcome to My Flask Server!</h1>"

@app.route('/hello/<name>')
def hello(name):
    return f"<h2>Hello, {name}!</h2>"

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        posted_data = request.get_json()
        return jsonify({"you_sent": posted_data}), 201
    else:
        return jsonify({"message": "Send me some JSON data!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Please provide a prompt in the request body"}), 400

        prompt = data['prompt']
        mode = data.get('mode', 'fast')
        timeframe = data.get('timeframe', 'short')

        response = generate_response(prompt, mode=mode, timeframe=timeframe)

        return jsonify({
            "success": True,
            "prediction_result": response
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
