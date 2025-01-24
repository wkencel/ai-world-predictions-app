# app.py
from flask import Flask, jsonify, request
from services.kalshi import get_events, get_event, get_markets, get_market, get_trades
from services.openai import generate_response

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
        limit = int(request.args.get('limit', 100))
        cursor = request.args.get('cursor', None)
        event_ticker = request.args.get('event_ticker', None)
        series_ticker = request.args.get('series_ticker', None)
        status = request.args.get('status', None)
        tickers = request.args.get('tickers', None)
        markets = get_markets(limit=limit, cursor=cursor, event_ticker=event_ticker, series_ticker=series_ticker, status=status, tickers=tickers)
        return jsonify(markets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
