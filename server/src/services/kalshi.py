import os
import time
import base64
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import logging

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../.env'))

load_dotenv(dotenv_path=dotenv_path)

# KALSHI_API_URL = os.getenv("KALSHI_API_URL")
# KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
# KALSHI_API_PRIVATE_KEY = os.getenv("KALSHI_API_PRIVATE_KEY")

# For testing, use mock data if API keys aren't available
KALSHI_API_URL = os.getenv("KALSHI_API_URL", "https://trading-api.kalshi.com")
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "mock_key_for_testing")
KALSHI_API_PRIVATE_KEY = os.getenv("KALSHI_API_PRIVATE_KEY", "mock_private_key_for_testing")

if not KALSHI_API_URL or not KALSHI_API_KEY_ID or not KALSHI_API_PRIVATE_KEY:
    raise ValueError("Missing Kalshi API configuration in environment variables")

# Comment out private key loading
# private_key = serialization.load_pem_private_key(
#     KALSHI_API_PRIVATE_KEY.encode(),
#     password=None,
#     backend=default_backend()
# )

last_api_call = datetime.now()

def rate_limit():
    """
    Enforces a minimum delay between successive API requests to avoid exceeding rate limits.
    Sleeps for at least 100 milliseconds if the previous request was made too recently.
    """
    global last_api_call
    threshold_in_milliseconds = 100
    now = datetime.now()
    threshold_in_microseconds = 1000 * threshold_in_milliseconds
    threshold_in_seconds = threshold_in_milliseconds / 1000
    if now - last_api_call < timedelta(microseconds=threshold_in_microseconds):
        print(f"Rate limiting applied: Sleeping for {threshold_in_seconds} seconds")
        time.sleep(threshold_in_seconds)
    last_api_call = datetime.now()


def generate_signature(timestamp: str, method: str, path: str) -> str:
    """
    Mock signature for testing. Returns a dummy signature.
    """
    # Comment out real signature generation
    # message = f"{timestamp}{method}{path}".encode("utf-8")
    # signature = private_key.sign(
    #     message,
    #     padding.PSS(
    #         mgf=padding.MGF1(hashes.SHA256()),
    #         salt_length=padding.PSS.MAX_LENGTH
    #     ),
    #     hashes.SHA256()
    # )
    # return base64.b64encode(signature).decode("utf-8")
    return "mock_signature_for_testing"


def get_headers(method: str, path: str) -> dict:
    """
    Generates authentication headers for the API request.

    Args:
        method (str): The HTTP method (e.g., GET, POST).
        path (str): The API endpoint path.

    Returns:
        dict: A dictionary containing the required authentication headers.
    """
    timestamp = str(int(time.time() * 1000))  # Current timestamp in milliseconds
    signature = generate_signature(timestamp, method, path)
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


def make_request(method: str, path: str, params=None):
    """
    Makes a signed request to the Kalshi API.

    Args:
        method (str): The HTTP method (e.g., GET, POST).
        path (str): The API endpoint path.
        params (dict, optional): Query parameters or request body data.

    Returns:
        dict: JSON response from the API.
    """
    rate_limit()
    url = f"{KALSHI_API_URL}{path}"
    headers = get_headers(method, path)
    response = requests.get(url, headers=headers, params=params) if method == "GET" else requests.post(url,
                                                                                                       headers=headers,
                                                                                                       json=params)
    response.raise_for_status()
    return response.json()


def get_events(limit=100, cursor=None, status=None, series_ticker=None, with_nested_markets=False):
    """Mock implementation for testing"""
    return {
        "events": [
            {
                "ticker": "NBA_GSW_LAL_20240320",
                "title": "Warriors vs Lakers",
                "status": "active",
                "markets": [
                    {
                        "ticker": "GSW_WIN",
                        "odds": 1.85,
                        "volume": 150000
                    }
                ]
            }
        ]
    }

# def get_events(limit=100, cursor=None, status=None, series_ticker=None, with_nested_markets=False):
#     """
#     Fetches a list of events from the Kalshi API.

#     Args:
#         limit (int): Number of results per page (default: 100).
#         cursor (str, optional): Pagination cursor.
#         status (str, optional): Filter by event status.
#         series_ticker (str, optional): Filter by series ticker.
#         with_nested_markets (bool, optional): Include nested markets in the response.

#     Returns:
#         dict: JSON response containing event data.
#     """
#     path = "/trade-api/v2/events"
#     params = {k: v for k, v in {
#         "limit": limit,
#         "cursor": cursor,
#         "status": status,
#         "series_ticker": series_ticker,
#         "with_nested_markets": str(with_nested_markets).lower(),
#     }.items() if v is not None}
#     return make_request("GET", path, params=params)



def get_event(event_ticker, with_nested_markets=False):
    """
    Fetches details for a specific event by its ticker.

    Args:
        event_ticker (str): The event ticker.
        with_nested_markets (bool, optional): Include nested markets in the response.

    Returns:
        dict: JSON response containing event details.
    """
    path = f"/trade-api/v2/events/{event_ticker}"
    params = {"with_nested_markets": str(with_nested_markets).lower()}
    return make_request("GET", path, params=params)


def get_markets(limit=100, cursor=None, event_ticker=None, series_ticker=None, status=None, tickers=None):
    """
    Fetches a list of markets from the Kalshi API.

    Args:
        limit (int): Number of results per page (default: 100).
        cursor (str, optional): Pagination cursor.
        event_ticker (str, optional): Filter by event ticker.
        series_ticker (str, optional): Filter by series ticker.
        status (str, optional): Filter by market status.
        tickers (str, optional): Filter by specific market tickers.

    Returns:
        dict: JSON response containing market data.
    """
    path = "/trade-api/v2/markets"
    params = {k: v for k, v in {
        "limit": limit,
        "cursor": cursor,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "status": status,
        "tickers": tickers,
    }.items() if v is not None}
    return make_request("GET", path, params=params)


def get_market(ticker: str):
    """
    Fetches details for a specific market by its ticker.

    Args:
        ticker (str): The market ticker.

    Returns:
        dict: JSON response containing market details.
    """
    path = f"/trade-api/v2/markets/{ticker}"
    return make_request("GET", path)


def get_trades(cursor=None, limit=100, ticker=None, min_ts=None, max_ts=None):
    """
    Fetches trades for all markets or a specific market.

    Args:
        cursor (str, optional): Pagination cursor.
        limit (int, optional): Number of results per page (default: 100).
        ticker (str, optional): Filter by market ticker.
        min_ts (int, optional): Minimum timestamp for trades.
        max_ts (int, optional): Maximum timestamp for trades.

    Returns:
        dict: JSON response containing trade data.
    """
    path = "/trade-api/v2/markets/trades"
    params = {k: v for k, v in {
        "cursor": cursor,
        "limit": limit,
        "ticker": ticker,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }.items() if v is not None}
    return make_request("GET", path, params=params)
