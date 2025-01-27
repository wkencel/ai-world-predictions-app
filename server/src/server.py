# app.py
import requests
from flask import Flask, jsonify, request

from services.polymarket import get_polymarket_market, get_polymarket_markets
from services.kalshi import get_events, get_event, get_markets, get_market, get_trades
from services.openai import generate_response
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

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

        # Handle different response formats based on mode
        if mode == 'council':
            # Return the full council response structure
            return jsonify({
                "success": True,
                "consensus": {
                    "final_prediction": response.get("consensus", {}).get("final_prediction", "Consensus building..."),
                    "confidence_level": response.get("consensus", {}).get("confidence_level", 0)
                },
                "discussion": response.get("discussion", []),
                "mode": "council"
            })
        else:
            # Return simple prediction for fast/deep modes
            return jsonify({
                "success": True,
                "prediction_result": response
            })

    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
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

@app.route('/polymarket/markets', methods=['GET'])
def polymarket_markets():
    try:
        markets = get_polymarket_markets()
        return jsonify(markets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/polymarket/market/<condition_id>', methods=['GET'])
def polymarket_market(condition_id):
    try:
        market = get_polymarket_market(condition_id)
        return jsonify(market)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to Polymarket API.", "details": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
