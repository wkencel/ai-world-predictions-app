# app.py
from flask import Flask, jsonify, request

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
