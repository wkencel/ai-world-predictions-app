import os
import tweepy
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

from server.src.utils.logger import color_logger

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(dotenv_path=dotenv_path)

# Twitter API credentials
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Add debug logging for credentials and their values
color_logger.info("Twitter API Credentials Check:")
color_logger.info(f"API Key exists: {bool(TWITTER_API_KEY)}")
color_logger.info(f"API Key value: {TWITTER_API_KEY}")
color_logger.info(f"API Secret exists: {bool(TWITTER_API_SECRET_KEY)}")
color_logger.info(f"Access Token exists: {bool(TWITTER_ACCESS_TOKEN)}")
color_logger.info(f"Access Token Secret exists: {bool(TWITTER_ACCESS_TOKEN_SECRET)}")
color_logger.info(f"Bearer Token exists: {bool(TWITTER_BEARER_TOKEN)}")
color_logger.info(f"Bearer Token prefix: {TWITTER_BEARER_TOKEN[:10]}..." if TWITTER_BEARER_TOKEN else "No Bearer Token")
color_logger.info(f"Env file path: {os.path.abspath(dotenv_path)}")
# Check if we have the minimum required credentials
TWITTER_ENABLED = all([
    TWITTER_API_KEY,
    TWITTER_API_SECRET_KEY,
    TWITTER_BEARER_TOKEN
])




if not TWITTER_ENABLED:
    color_logger.warning("Twitter API credentials missing - Twitter features will be disabled")
    client = None
else:
    try:
        # Initialize API v2 client with authentication
        client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET_KEY,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
        color_logger.info("Twitter API client initialized successfully")
    except Exception as e:
        color_logger.error(f"Failed to initialize Twitter client: {str(e)}")
        client = None

# Rate limiting
last_api_call = datetime.now()

def rate_limit():
    """Enforce rate limiting between API calls"""
    global last_api_call
    min_time_between_calls = 1.1  # seconds

    time_since_last_call = (datetime.now() - last_api_call).total_seconds()
    if time_since_last_call < min_time_between_calls:
        time.sleep(min_time_between_calls - time_since_last_call)

    last_api_call = datetime.now()

def search_tweets(query: str, max_results: int = 100) -> Dict:
    """
    Search for recent tweets matching a query.

    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return (default: 100)

    Returns:
        dict: Dictionary containing tweets and metadata
    """
    try:
        rate_limit()
        color_logger.info(f"Searching tweets for query: {query}")
        color_logger.info(f"Using client with bearer token: {bool(client.bearer_token)}")

        # Search tweets
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),  # Twitter API limit
            tweet_fields=['created_at', 'public_metrics', 'context_annotations']
        )

        if not tweets.data:
            color_logger.info("No tweets found for query")
            return {
                "tweets": [],
                "meta": {"result_count": 0}
            }

        # Process tweets into a more usable format
        processed_tweets = []
        for tweet in tweets.data:
            processed_tweet = {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat(),
                "metrics": tweet.public_metrics,
                "context": tweet.context_annotations if hasattr(tweet, 'context_annotations') else None
            }
            processed_tweets.append(processed_tweet)

        color_logger.info(f"Successfully found {len(processed_tweets)} tweets")
        return {
            "tweets": processed_tweets,
            "meta": tweets.meta
        }

    except Exception as e:
        color_logger.error(f"Error searching tweets: {str(e)}")
        color_logger.error(f"Error type: {type(e).__name__}")
        color_logger.error(f"Full error details: {repr(e)}")
        return {
            "error": str(e),
            "tweets": [],
            "meta": {"result_count": 0}
        }

def get_user_tweets(username: str, max_results: int = 100) -> Dict:
    """
    Get recent tweets from a specific user.

    Args:
        username (str): Twitter username (without @)
        max_results (int): Maximum number of tweets to return

    Returns:
        dict: Dictionary containing tweets and metadata
    """
    try:
        rate_limit()
        color_logger.info(f"Fetching tweets for user: {username}")

        # First get the user ID
        user = client.get_user(username=username)
        if not user.data:
            raise ValueError(f"User {username} not found")

        user_id = user.data.id

        # Get user's tweets (ensure min 5 results)
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=max(min(max_results, 100), 5),  # Between 5 and 100
            tweet_fields=['created_at', 'public_metrics']
        )

        if not tweets.data:
            return {
                "tweets": [],
                "meta": {"result_count": 0}
            }

        processed_tweets = []
        for tweet in tweets.data:
            processed_tweet = {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat(),
                "metrics": tweet.public_metrics
            }
            processed_tweets.append(processed_tweet)

        return {
            "tweets": processed_tweets,
            "meta": tweets.meta
        }

    except Exception as e:
        color_logger.error(f"Error getting user tweets: {str(e)}")
        return {
            "error": str(e),
            "tweets": [],
            "meta": {"result_count": 0}
        }

def analyze_sentiment(tweets: List[Dict]) -> Dict:
    """
    Analyze sentiment of a collection of tweets.

    Args:
        tweets (List[Dict]): List of processed tweets

    Returns:
        dict: Sentiment analysis results
    """
    if not tweets:
        return {
            "overall_sentiment": "neutral",
            "sentiment_scores": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "engagement_metrics": {
                "total_likes": 0,
                "total_retweets": 0,
                "total_replies": 0
            }
        }

    # Calculate engagement metrics
    total_likes = sum(tweet.get('metrics', {}).get('like_count', 0) for tweet in tweets)
    total_retweets = sum(tweet.get('metrics', {}).get('retweet_count', 0) for tweet in tweets)
    total_replies = sum(tweet.get('metrics', {}).get('reply_count', 0) for tweet in tweets)

    # For now, return mock sentiment analysis
    # In a real implementation, you would use a sentiment analysis model here
    return {
        "overall_sentiment": "positive",
        "sentiment_scores": {
            "positive": 0.6,
            "neutral": 0.3,
            "negative": 0.1
        },
        "engagement_metrics": {
            "total_likes": total_likes,
            "total_retweets": total_retweets,
            "total_replies": total_replies
        }
    }

def get_topic_sentiment(topic: str) -> Dict:
    """
    Get sentiment analysis for a specific topic.

    Args:
        topic (str): The topic to analyze

    Returns:
        dict: Topic sentiment analysis results
    """
    try:
        # Search for tweets about the topic
        search_results = search_tweets(topic, max_results=100)

        # Analyze sentiment of the tweets
        sentiment_results = analyze_sentiment(search_results.get('tweets', []))

        return {
            "topic": topic,
            "tweet_count": len(search_results.get('tweets', [])),
            "sentiment_analysis": sentiment_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        color_logger.error(f"Error analyzing topic sentiment: {str(e)}")
        return {
            "error": str(e),
            "topic": topic,
            "tweet_count": 0,
            "sentiment_analysis": None,
            "timestamp": datetime.now().isoformat()
        }

def search_relevant_signals(topic: str, max_results: int = 2) -> Dict:
    """
    Search for high-quality signals about market-moving events.
    Returns empty results if Twitter is not configured.
    """
    if not TWITTER_ENABLED or client is None:
        color_logger.warning("Twitter API not configured - returning empty results")
        return {
            "signals": [],
            "meta": {
                "result_count": 0,
                "status": "Twitter API not configured"
            }
        }

    try:
        rate_limit()

        # Define high-signal keywords and patterns
        injury_keywords = "injury injured hurt out questionable doubtful GTD"
        lineup_keywords = "starting lineup benched suspended"

        # Construct focused query
        query = f"({topic}) ({injury_keywords} OR {lineup_keywords})"
        query += " -RT lang:en is:verified"

        color_logger.info(f"Searching for high-signal tweets with query: {query}")

        # Search tweets with minimal fields
        tweets = client.search_recent_tweets(
            query=query,
            max_results=10,  # Set minimum required by Twitter API
            tweet_fields=['created_at', 'public_metrics', 'author_id'],
            user_fields=['verified', 'username'],
            expansions=['author_id']
        )

        if not tweets.data:
            color_logger.info("No high-signal tweets found")
            return {
                "signals": [],
                "meta": {"result_count": 0}
            }

        # Process and filter tweets
        high_signal_tweets = []
        users = {user.id: user for user in tweets.includes['users']} if 'users' in tweets.includes else {}

        trusted_sources = ['wojespn', 'ShamsCharania', 'AdamSchefter']

        for tweet in tweets.data:
            author = users.get(tweet.author_id)
            if not author or not author.verified:
                continue

            # Prioritize trusted sources
            if author.username in trusted_sources:
                signal_type = "injury" if any(kw in tweet.text.lower() for kw in injury_keywords.split()) else "lineup"
                high_signal_tweets.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "author": {"username": author.username, "verified": True},
                    "metrics": tweet.public_metrics,
                    "signal_type": signal_type
                })

                if len(high_signal_tweets) >= 2:
                    break

        color_logger.info(f"Found {len(high_signal_tweets)} high-signal tweets")
        return {
            "signals": high_signal_tweets[:2],
            "meta": {"result_count": len(high_signal_tweets[:2])}
        }

    except Exception as e:
        color_logger.error(f"Error searching high-signal tweets: {str(e)}")
        return {
            "error": str(e),
            "signals": [],
            "meta": {
                "result_count": 0,
                "status": "error"
            }
        }
