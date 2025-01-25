# app.py
from flask import Flask, jsonify, request
from services.kalshi import get_events, get_event, get_markets, get_market, get_trades
from services.openai import generate_response
from framework.prediction_framework import PredictionFramework
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Initialize the framework
framework = PredictionFramework()

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
async def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Please provide data in the request body"}), 400

        query = data.get('query')
        if not query:
            return jsonify({"error": "Please provide a query in the request body"}), 400

        # Generate prediction using the framework
        result = await framework.generate_prediction(query)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/update-performance', methods=['POST'])
async def update_performance():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Please provide data in the request body"}), 400

        prediction_id = data.get('prediction_id')
        actual_outcome = data.get('actual_outcome')

        if not prediction_id or actual_outcome is None:
            return jsonify({"error": "Missing required fields: prediction_id or actual_outcome"}), 400

        # Update the model performance
        framework.update_model_performance(prediction_id, actual_outcome)

        return jsonify({"success": True})

    except Exception as e:
        print(f"Error in update-performance endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/kalshi/events', methods=['GET'])
def kalshi_events():
    try:
        limit = int(request.args.get('limit', 100))
        cursor = request.args.get('cursor', None)
        status = request.args.get('status', None)
        series_ticker = request.args.get('series_ticker', None)
        with_nested_markets = request.args.get('with_nested_markets', 'false').lower() == 'true'
        events = get_events(limit=limit, cursor=cursor, status=status, series_ticker=series_ticker, with_nested_markets=with_nested_markets)
        return jsonify(events)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/kalshi/event/<event_ticker>', methods=['GET'])
def kalshi_event(event_ticker):
    try:
        with_nested_markets = request.args.get('with_nested_markets', 'false').lower() == 'true'
        event = get_event(event_ticker, with_nested_markets=with_nested_markets)
        return jsonify(event)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/kalshi/markets', methods=['GET'])
def kalshi_markets():
    try:
        # Use the mock data from get_events
        events_data = get_events(limit=10)

        # Transform the events data into a format suitable for the frontend
        markets = []
        for event in events_data.get('events', []):
            for market in event.get('markets', []):
                markets.append({
                    'ticker': market['ticker'],
                    'title': event['title'],
                    'status': event['status'],
                    'odds': market.get('odds'),
                    'volume': market.get('volume'),
                    'description': f"Event: {event['title']} - Market: {market['ticker']}"
                })

        return jsonify(markets)
    except Exception as e:
        print(f"Error in /kalshi/markets: {str(e)}")
        print(traceback.format_exc())  # Print the full stack trace
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/kalshi/market/<ticker>', methods=['GET'])
def kalshi_market(ticker):
    try:
        market = get_market(ticker)
        return jsonify(market)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/kalshi/trades', methods=['GET'])
def kalshi_trades():
    try:
        cursor = request.args.get('cursor', None)
        limit = int(request.args.get('limit', 100))
        ticker = request.args.get('ticker', None)
        min_ts = request.args.get('min_ts', None)
        max_ts = request.args.get('max_ts', None)

        trades = get_trades(cursor=cursor, limit=limit, ticker=ticker, min_ts=min_ts, max_ts=max_ts)
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
