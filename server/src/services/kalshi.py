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
KALSHI_API_URL = os.getenv("KALSHI_API_URL", "https://api.elections.kalshi.com")
logging.info(f"Kalshi API URL being used: {KALSHI_API_URL}")
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


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Add this debug decorator to your request method
def log_request(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"Making request to Kalshi API: {KALSHI_API_URL}")
        return func(*args, **kwargs)
    return wrapper


# Add the decorator to your request method
@log_request
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
    print(f"DEBUG - KALSHI_API_URL: {KALSHI_API_URL}")  # Debug line
    print(f"DEBUG - Making request to: {url}")  # Debug line

    # Add required headers
    headers = {
        'accept': 'application/json'
    }

    try:
        response = requests.request(
            method,
            url,
            params=params,
            headers=headers  # Add headers to the request
        )
        print(f"DEBUG - Response status code: {response.status_code}")  # Debug line
        print(f"DEBUG - Response text: {response.text[:200]}...")  # Debug line
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"DEBUG - Request failed: {str(e)}")  # Debug line
        raise RuntimeError(f"Request to Kalshi API failed: {str(e)}")


# def get_events(limit=100, cursor=None, status=None, series_ticker=None, with_nested_markets=False):
#     """Mock implementation for testing"""
#     return {
#         "events": [
#             {
#                 "ticker": "NBA_GSW_LAL_20240320",
#                 "title": "Warriors vs Lakers",
#                 "status": "active",
#                 "markets": [
#                     {
#                         "ticker": "GSW_WIN",
#                         "odds": 1.85,
#                         "volume": 150000,
#                         "description": "Golden State Warriors to win"
#                     }
#                 ]
#             },
#             {
#                 "ticker": "NBA_BOS_MIA_20240320",
#                 "title": "Celtics vs Heat",
#                 "status": "active",
#                 "markets": [
#                     {
#                         "ticker": "BOS_WIN",
#                         "odds": 1.65,
#                         "volume": 180000,
#                         "description": "Boston Celtics to win"
#                     }
#                 ]
#             },
#             {
#                 "ticker": "NBA_DEN_PHX_20240320",
#                 "title": "Nuggets vs Suns",
#                 "status": "upcoming",
#                 "markets": [
#                     {
#                         "ticker": "DEN_WIN",
#                         "odds": 1.95,
#                         "volume": 120000,
#                         "description": "Denver Nuggets to win"
#                     }
#                 ]
#             }
#         ]
#     }

def get_events(limit=100, cursor=None, with_nested_markets=False, status=None, series_ticker=None):
    """
    Fetches events from Kalshi Elections API.

    Args:
        limit (int, optional): Number of results per page (default: 100).
        cursor (str, optional): Pagination cursor.
        with_nested_markets (bool, optional): Include nested markets data.
        status (str, optional): Filter events by status.
        series_ticker (str, optional): Filter events by series ticker.

    Returns:
        dict: JSON response containing event data.
    """
    path = "/trade-api/v2/events"
    params = {
        "limit": limit,
        "with_nested_markets": str(with_nested_markets).lower()
    }
    if cursor:
        params["cursor"] = cursor
    if status:
        params["status"] = status
    if series_ticker:
        params["series_ticker"] = series_ticker

    print(f"DEBUG - Full URL with params: {KALSHI_API_URL}{path}")  # Debug line
    return make_request("GET", path, params=params)


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


def get_markets(limit=100, cursor=None, event_ticker=None, series_ticker=None, status=None, tickers=None, min_volume=None, category=None):
    """
    Fetches a list of markets from the Kalshi API with enhanced filtering.

    Args:
        limit (int): Number of results per page (default: 100)
        cursor (str, optional): Pagination cursor
        event_ticker (str, optional): Filter by event ticker
        series_ticker (str, optional): Filter by series ticker
        status (str, optional): Filter by market status ('active', 'settled', 'closed')
        tickers (str, optional): Filter by specific market tickers (comma-separated)
        min_volume (int, optional): Filter markets with minimum trading volume
        category (str, optional): Filter by market category

    Returns:
        dict: JSON response containing filtered market data
    """
    path = "/trade-api/v2/markets"

    # Build base params with specific focus on Super Bowl related markets
    params = {
        "limit": limit,
        "status": status or "active",  # Default to active markets
    }

    # Add optional filters
    if event_ticker:
        params["event_ticker"] = event_ticker
    if series_ticker:
        params["series_ticker"] = "NFL-SB"  # Focus on Super Bowl markets
    if tickers:
        params["tickers"] = tickers

    try:
        # Make the API request
        response = make_request("GET", path, params=params)

        if response and 'markets' in response:
            markets = response['markets']

            # Filter by minimum volume if specified
            if min_volume:
                markets = [m for m in markets if m.get('volume', 0) >= min_volume]

            # Filter by category if specified
            if category:
                markets = [m for m in markets if m.get('category', '').lower() == category.lower()]

            # Add calculated fields for each market
            for market in markets:
                yes_price = float(market.get('yes_price', 0)) / 100
                no_price = float(market.get('no_price', 0)) / 100

                # Calculate ROI for both YES and NO positions
                market['yes_roi'] = ((1 - yes_price) / yes_price) * 100 if yes_price > 0 else 0
                market['no_roi'] = ((1 - no_price) / no_price) * 100 if no_price > 0 else 0

                # Calculate implied probabilities
                market['yes_implied_prob'] = yes_price
                market['no_implied_prob'] = no_price

            response['markets'] = markets
            response['filtered_count'] = len(markets)

        return response

    except Exception as e:
        logger.error(f"Error fetching markets: {str(e)}")
        return {"markets": [], "error": str(e)}


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

# Debug print of environment variables
kalshi_env_vars = {k:v for k,v in os.environ.items() if 'KALSHI' in k}
logger.debug(f"Kalshi environment variables: {kalshi_env_vars}")

# At the top of the file, after loading environment variables:
print(f"DEBUG - Initial KALSHI_API_URL value: {KALSHI_API_URL}")  # Debug line

# Verify the URL is correct
if not KALSHI_API_URL.startswith("https://api.elections.kalshi.com"):
    print("WARNING: Incorrect Kalshi API URL detected")
    KALSHI_API_URL = "https://api.elections.kalshi.com"
