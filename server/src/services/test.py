import json
from datetime import datetime
from typing import Dict
from db.pinecone.setup_pinecone import query_pinecone
from services.kalshi import get_events
from utils.logger import color_logger  # Update import

@color_logger.log_service_call('test')
def get_test_data() -> Dict:
    """
    Enhanced test data with better logging and more realistic data
    """
    test_data = {}

    # Get real Kalshi data
    try:
        kalshi_data = get_events(limit=1)
        color_logger.logger.info(json.dumps({
            'component': 'test_data',
            'source': 'kalshi',
            'status': 'success'
        }))
        test_data['market_data'] = kalshi_data
    except Exception as e:
        color_logger.logger.warning(json.dumps({
            'component': 'test_data',
            'source': 'kalshi',
            'status': 'failed',
            'error': str(e)
        }))
        test_data['market_data'] = {
            "Warriors vs Lakers": {
                "market_id": "GSW_LAL_20240320",
                "odds": {"warriors_win": 1.85, "lakers_win": 2.10},
                "volume": 150000,
                "last_updated": datetime.now().isoformat(),
                "source": "mock_data"
            }
        }

    # Get RAG context
    try:
        rag_context = query_pinecone("Warriors Lakers betting history")
        color_logger.logger.info(json.dumps({
            'component': 'test_data',
            'source': 'pinecone',
            'status': 'success',
            'matches_found': len(rag_context.get('matches', []))
        }))
        test_data['historical_context'] = rag_context
    except Exception as e:
        color_logger.logger.warning(json.dumps({
            'component': 'test_data',
            'source': 'pinecone',
            'status': 'failed',
            'error': str(e)
        }))

    # Enhanced test data
    test_data.update({
        "injury_data": [
            {
                "name": "Stephen Curry",
                "team": "Golden State Warriors",
                "injuryStatus": "Healthy",
                "position": "Guard",
                "stats": ["30.5 PPG", "5.8 APG", "1.2 SPG"],
                "last_5_games": [
                    {"points": 32, "assists": 6, "date": "2024-03-15"},
                    {"points": 28, "assists": 5, "date": "2024-03-13"},
                    {"points": 35, "assists": 7, "date": "2024-03-11"},
                    {"points": 29, "assists": 4, "date": "2024-03-09"},
                    {"points": 28, "assists": 7, "date": "2024-03-07"}
                ],
                "reference": "https://www.sportingnews.com/us"
            }
        ],
        "news_sentiment": {
            "Golden State Warriors": {
                "recent_articles": [
                    {
                        "title": "Curry's Return Boosts Warriors' Playoff Hopes",
                        "sentiment": 0.85,
                        "source": "ESPN",
                        "timestamp": datetime.now().isoformat(),
                        "summary": "Stephen Curry's recent performances have significantly improved Warriors' playoff chances"
                    }
                ],
                "overall_sentiment": 0.75,
                "sentiment_trend": "positive",
                "analysis_timestamp": datetime.now().isoformat()
            }
        },
        "historical_data": {
            "warriors_last_5": [
                {"opponent": "Clippers", "result": "W", "score": "134-120", "date": "2024-03-15"},
                {"opponent": "Lakers", "result": "W", "score": "128-110", "date": "2024-03-13"},
                {"opponent": "Suns", "result": "L", "score": "112-118", "date": "2024-03-11"},
                {"opponent": "Kings", "result": "W", "score": "122-114", "date": "2024-03-09"},
                {"opponent": "Spurs", "result": "W", "score": "130-115", "date": "2024-03-07"}
            ],
            "head_to_head_last_5": [
                {"date": "2024-03-13", "result": "W", "score": "128-110", "location": "home"},
                {"date": "2024-02-15", "result": "W", "score": "125-118", "location": "away"},
                {"date": "2024-01-27", "result": "L", "score": "115-120", "location": "away"},
                {"date": "2023-12-23", "result": "W", "score": "130-125", "location": "home"},
                {"date": "2023-12-12", "result": "W", "score": "122-115", "location": "home"}
            ]
        }
    })

    color_logger.logger.info(json.dumps({
        'component': 'test_data',
        'status': 'complete',
        'data_sources': list(test_data.keys())
    }))

    return test_data

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
        color_logger.logger.info("\n=== BETTING ANALYSIS RESULTS ===")
        color_logger.logger.info(json.dumps(prediction, indent=2))

        # Simulate game result and update tracker
        result = simulate_game_result()
        for discussion in prediction['discussion']:
            agent_id = f"{discussion['expert']}_{discussion['analysis'].get('confidence', 'NA')}"
            update_prediction_outcome(agent_id, result['actual_odds_payout'])

        color_logger.logger.info("\n=== GAME RESULT ===")
        color_logger.logger.info(json.dumps(result, indent=2))

    except Exception as e:
        color_logger.logger.error(f"Test failed: {str(e)}")
