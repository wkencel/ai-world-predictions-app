import json
import logging
from datetime import datetime
from typing import Dict
from db.pinecone.setup_pinecone import query_pinecone
from services.kalshi import get_events

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_test_data() -> Dict:
    """
    Simulates data that would come from our various data sources
    Returns a dictionary containing all test data needed for the prediction
    """
    # Get real Kalshi data if available
    try:
        kalshi_data = get_events(limit=1)
    except Exception as e:
        logger.warning(f"Failed to get Kalshi data: {str(e)}")
        kalshi_data = None

    # Get RAG context if available
    try:
        rag_context = query_pinecone("Warriors Lakers betting history")
    except Exception as e:
        logger.warning(f"Failed to get RAG context: {str(e)}")
        rag_context = None

    # Combine real and simulated data
    return {
        # Injury Data (simulating web scraper)
        "injury_data": [
            {
                "name": "Stephen Curry",
                "team": "Golden State Warriors",
                "injuryStatus": "Healthy",
                "position": "Guard",
                "stats": ["30.5 PPG", "5.8 APG", "1.2 SPG"],
                "reference": "https://www.sportingnews.com/us"
            },
            {
                "name": "Novak Djokovic",
                "injuryStatus": "Rumored injury, but known to have won majors despite such rumors",
                "reference": "https://www.sportingnews.com/us/tennis/news/djokovic-vs-zverev-score-result-australian-open-semi-finals/d4f22b13e77f8388f9d221c7"
            }
        ],

        # Market Data (simulating Kalshi/Polymarket API)
        "market_data": kalshi_data if kalshi_data else {
            "Warriors vs Lakers": {
                "market_id": "GSW_LAL_20240320",
                "odds": {
                    "warriors_win": 1.85,
                    "lakers_win": 2.10
                },
                "volume": 150000,
                "last_updated": datetime.now().isoformat()
            }
        },

        # News Sentiment (simulating news API)
        "news_sentiment": {
            "Golden State Warriors": {
                "recent_articles": [
                    {
                        "title": "Curry's Return Boosts Warriors' Playoff Hopes",
                        "sentiment": 0.85,
                        "source": "ESPN",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "overall_sentiment": 0.75
            }
        },

        # Historical Performance (simulating database)
        "historical_data": {
            "warriors_last_5": [
                {"opponent": "Clippers", "result": "W", "score": "134-120"},
                {"opponent": "Lakers", "result": "W", "score": "128-110"},
                {"opponent": "Suns", "result": "L", "score": "112-118"},
                {"opponent": "Kings", "result": "W", "score": "122-114"},
                {"opponent": "Spurs", "result": "W", "score": "130-115"}
            ]
        },

        # Add RAG context if available
        "historical_context": rag_context if rag_context else None
    }

def format_prompt(data: Dict) -> str:
    """
    Formats the test data into a prompt for the LLM
    """
    # Get relevant player
    player = data['injury_data'][0]  # Curry's data

    # Get market odds
    market = data['market_data']['Warriors vs Lakers']

    # Format the prompt
    prompt = f"""Analyze this NBA betting opportunity:

PLAYER ANALYSIS:
- Player: {player['name']}
- Team: {player['team']}
- Status: {player['injuryStatus']}
- Recent Stats: {', '.join(player['stats'])}

MARKET ODDS:
- Game: Warriors vs Lakers
- Warriors Win Odds: {market['odds']['warriors_win']}
- Market Volume: ${market['volume']:,}

RECENT TEAM PERFORMANCE:
{format_historical_data(data['historical_data']['warriors_last_5'])}

NEWS SENTIMENT:
- Overall Team Sentiment: {data['news_sentiment']['Golden State Warriors']['overall_sentiment']}
- Latest News: {data['news_sentiment']['Golden State Warriors']['recent_articles'][0]['title']}

Based on this data, provide:
1. Win probability assessment
2. Recommended bet size (1-10 units)
3. Key factors supporting the prediction
4. Potential risks to consider
"""
    return prompt

def format_historical_data(games):
    """Helper function to format historical game data"""
    return "\n".join([
        f"- vs {game['opponent']}: {game['result']} ({game['score']})"
        for game in games
    ])

def simulate_game_result() -> Dict:
    """
    Simulates the actual game result for testing the prediction accuracy
    """
    return {
        "final_score": "Warriors 128, Lakers 115",
        "winner": "Warriors",
        "actual_odds_payout": 1.85,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    from openai import generate_response, update_prediction_outcome

    try:
        # Get test data
        test_data = get_test_data()

        # Format prompt
        prompt = format_prompt(test_data)

        # Get prediction using council mode
        prediction = generate_response(
            prompt=prompt,
            mode='council',
            max_tokens=500,
            timeframe='short',
            current_price=test_data['market_data']['Warriors vs Lakers']['odds']['warriors_win']
        )

        # Print prediction results
        logger.info("\n=== BETTING ANALYSIS RESULTS ===")
        logger.info(json.dumps(prediction, indent=2))

        # Simulate game result and update tracker
        result = simulate_game_result()
        for discussion in prediction['discussion']:
            agent_id = f"{discussion['expert']}_{discussion['analysis'].get('confidence', 'NA')}"
            update_prediction_outcome(agent_id, result['actual_odds_payout'])

        logger.info("\n=== GAME RESULT ===")
        logger.info(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
