import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

from server.src.services.twitter import search_tweets, get_user_tweets, analyze_sentiment, search_relevant_signals
from server.src.utils.logger import color_logger

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(dotenv_path=dotenv_path)

def test_market_sentiment():
    """Test getting real-time market sentiment for potential arbitrage opportunities"""
    try:
        # Example: Testing NBA game sentiment
        game = "Warriors Lakers"
        print(f"\nAnalyzing sentiment for: {game}")

        # Search recent tweets about the game
        search_results = search_tweets(f"{game} game", max_results=2)

        if "error" in search_results:
            print(f"Error in search_tweets: {search_results['error']}")
            return

        # Analyze sentiment
        sentiment = analyze_sentiment(search_results.get('tweets', []))

        print("\nSentiment Analysis Results:")
        print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
        print(f"Sentiment Scores: {sentiment['sentiment_scores']}")
        print(f"Engagement: {sentiment['engagement_metrics']}")

        # Print sample tweets
        print("\nSample Tweets:")
        for tweet in search_results.get('tweets', [])[:3]:
            print(f"- {tweet['text'][:100]}...")
            print(f"  Likes: {tweet['metrics'].get('like_count', 0)}, "
                  f"Retweets: {tweet['metrics'].get('retweet_count', 0)}")

        # Get key player tweets
        players = ["StephenCurry30", "KingJames"]
        print("\nKey Player Recent Activity:")
        for player in players:
            player_tweets = get_user_tweets(player, max_results=3)
            if "error" not in player_tweets:
                print(f"\n{player}'s recent tweets:")
                for tweet in player_tweets.get('tweets', []):
                    print(f"- {tweet['text'][:100]}...")
                    print(f"  Engagement: {tweet['metrics']}")

    except Exception as e:
        print(f"Test failed: {str(e)}")

def test_market_signals():
    """Test getting high-quality market signals"""
    try:
        # Example: Testing NBA game signals
        game = "Warriors Lakers"
        print(f"\nSearching for market-moving signals about: {game}")

        # Search for high-signal tweets
        signals = search_relevant_signals(game, max_results=4)  # Reduced from 10

        if "error" in signals:
            print(f"Error in search: {signals['error']}")
            return

        print("\nHigh Signal Tweets Found:")
        for signal in signals['signals']:
            print(f"\nAuthor: @{signal['author']['username']} ({'✓' if signal['author']['verified'] else 'x'})")
            print(f"Time: {signal['created_at']}")
            print(f"Signal Type: {signal['signal_type'].upper()}")
            print(f"Tweet: {signal['text']}")
            print(f"Engagement: {signal['metrics']['like_count']} likes, {signal['metrics']['retweet_count']} retweets")
            print("-" * 80)

    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    # Only run the market signals test
    test_market_signals()  # Removed test_market_sentiment()
