import os
import requests
from dotenv import load_dotenv

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../.env'))
load_dotenv(dotenv_path=dotenv_path)

POLYMARKET_API_URL = os.getenv("POLYMARKET_API_URL")

if not POLYMARKET_API_URL:
    raise ValueError("Missing Polymarket API URL in environment variables.")


def make_request(method: str, path: str, params=None):
    """
    Makes a request to the Polymarket API.

    Args:
        method (str): The HTTP method (GET, POST, etc.).
        path (str): The endpoint path.
        params (dict, optional): Query parameters.

    Returns:
        dict: JSON response from the API.
    """
    url = f"{POLYMARKET_API_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request to Polymarket API failed: {str(e)}")

def get_polymarket_markets():
    """
    Fetches all available markets from Polymarket.

    Returns:
        dict: JSON response containing market data.
    """
    path = "/markets"
    return make_request("GET", path)

def get_polymarket_market(condition_id: str):
    """
    Fetches details for a specific market by its ID.

    Args:
        condition_id (str): The ID of the market.

    Returns:
        dict: JSON response containing market details.
    """
    path = f"/markets/{condition_id}"
    return make_request("GET", path)
