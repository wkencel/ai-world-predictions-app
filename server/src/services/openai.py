# services/openai.py
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from db.pinecone.setup_pinecone import query_pinecone
from services.kalshi import get_events
from utils.logger import color_logger
import openai
from collections import defaultdict

# Determine the path to the root directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=dotenv_path)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

client = AsyncOpenAI(
    api_key=api_key,
    timeout=60.0,  # Increase timeout for longer operations
    max_retries=3  # Add retries for reliability
)

# Model definitions
DEEP_MODEL = os.getenv('DEEP_MODEL')  # Deep thinking model
FAST_MODEL = os.getenv('FAST_MODEL')  # Quick response model
COUNCIL_MODEL = "gpt-4"  # Model for expert council

# todo: pull in some RAG store data
# todo: pull in some api data
# todo: pull in some news data
# todo: pull in web scraper data
# todo: pull in some other data

# Architecture: user input -> prompt -> model  -> response
# we also need some way of getting our prediction outcome into the model

# Move AgentPerformanceTracker class to the top of the file, after imports
class AgentPerformanceTracker:
    """Tracks and manages expert agent performance"""

    def __init__(self):
        self.performance_data = defaultdict(lambda: {
            'total_trades': 0,
            'total_pnl': 0,
            'wins': 0,
            'losses': 0
        })
        self._load_initial_data()
        self._log_leaderboard()  # Now safe to call after initialization
        color_logger.info("Performance tracker initialized with historical data")

    def _load_initial_data(self):
        """Load initial performance data"""
        try:
            # Initialize with empty data for each expert role
            default_roles = [
                "Technical Analyst",
                "Sentiment Analyst",
                "Macro Economist",
                "Risk Manager"
            ]

            for role in default_roles:
                if role not in self.performance_data:
                    self.performance_data[role] = {
                        'total_trades': 0,
                        'total_pnl': 0,
                        'wins': 0,
                        'losses': 0
                    }

            color_logger.info("Initialized performance tracking data")
        except Exception as e:
            color_logger.error(f"Error loading initial performance data: {str(e)}")

    def _log_leaderboard(self):
        """Log current performance leaderboard"""
        try:
            leaderboard = []
            for agent, data in self.performance_data.items():
                total_trades = data["total_trades"]
                win_rate = (data["wins"] / total_trades) if total_trades > 0 else 0
                avg_pnl = (data["total_pnl"] / total_trades) if total_trades > 0 else 0

                leaderboard.append({
                    "agent": agent,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl
                })

            # Sort by win rate
            leaderboard.sort(key=lambda x: x["win_rate"], reverse=True)

            color_logger.info("Performance Leaderboard:")
            for entry in leaderboard:
                color_logger.info(
                    f"{entry['agent']}: "
                    f"Win Rate: {entry['win_rate']:.2%}, "
                    f"Avg PNL: {entry['avg_pnl']:.2f}, "
                    f"Total Trades: {entry['total_trades']}"
                )
        except Exception as e:
            color_logger.error(f"Error generating leaderboard: {str(e)}")

    def update_performance(self, agent: str, prediction_id: str, actual_outcome: float, was_correct: bool):
        """Update performance metrics for an agent"""
        try:
            if agent in self.performance_data:
                self.performance_data[agent]['total_trades'] += 1
                self.performance_data[agent]['total_pnl'] += actual_outcome
                if was_correct:
                    self.performance_data[agent]['wins'] += 1
                else:
                    self.performance_data[agent]['losses'] += 1

                color_logger.info(f"Updated performance for {agent}: {self.performance_data[agent]}")
                self._log_leaderboard()  # Update leaderboard after performance change
        except Exception as e:
            color_logger.error(f"Error updating performance: {str(e)}")

    def log_prediction(self, prediction_data: Dict):
        """Log a new prediction"""
        try:
            # Ensure we're handling the prediction data correctly
            if isinstance(prediction_data, dict):
                color_logger.info(f"New prediction logged: {json.dumps(prediction_data, indent=2)}")
                # Add the prediction to performance tracking if needed
                agent = prediction_data.get('agent')
                if agent and agent in self.performance_data:
                    self.performance_data[agent]['total_trades'] += 1
            else:
                color_logger.warning(f"Invalid prediction data format: {type(prediction_data)}")
        except Exception as e:
            color_logger.error(f"Error logging prediction: {str(e)}")

class ExpertCouncil:
    """Manages a council of expert agents for prediction generation"""

    def __init__(self):
        self.experts = [
            {"role": "Technical Analyst", "bias": "chart-focused", "style": "conservative"},
            {"role": "Sentiment Analyst", "bias": "social-media-driven", "style": "aggressive"},
            {"role": "Macro Economist", "bias": "fundamentals-based", "style": "moderate"},
            {"role": "Risk Manager", "bias": "risk-focused", "style": "cautious"}
        ]
        self.performance_tracker = tracker  # Use the global tracker instance

    async def get_predictions(self, query: str = None, **kwargs) -> Dict:
        """Alias for get_consensus to maintain compatibility"""
        # Use query if provided, otherwise look for prompt in kwargs
        prompt = query if query is not None else kwargs.get('prompt', '')
        return await self.get_consensus(prompt)

    async def get_consensus(self, prompt: str) -> Dict:
        """Get consensus from all experts"""
        expert_predictions = {}

        for expert in self.experts:
            try:
                expert_prompt = generate_expert_prompt(expert, prompt)
                response = await client.chat.completions.create(
                    model=COUNCIL_MODEL,
                    messages=[{"role": "user", "content": expert_prompt}],
                    temperature=0.7,
                )
                prediction = parse_expert_response(response.choices[0].message.content)
                expert_predictions[expert['role']] = prediction

            except Exception as e:
                color_logger.error(f"Error getting prediction from {expert['role']}: {str(e)}")

        consensus = await self.generate_consensus(expert_predictions)
        return {
            "consensus": consensus,
            "expert_predictions": expert_predictions
        }

    async def generate_consensus(self, expert_predictions: Dict) -> Dict:
        """Generate consensus from expert predictions"""
        try:
            if not expert_predictions:
                return {"error": "No expert predictions available"}

            # Aggregate predictions
            consensus_prompt = self._create_consensus_prompt(expert_predictions)

            response = await client.chat.completions.create(
                model=COUNCIL_MODEL,
                messages=[{"role": "user", "content": consensus_prompt}],
                temperature=0.5,
            )

            # Extract the prediction from response
            consensus_text = response.choices[0].message.content

            return {
                "prediction": consensus_text,
                "confidence": self._calculate_confidence(expert_predictions),
                "expert_predictions": expert_predictions,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            color_logger.error(f"Error generating consensus: {str(e)}")
            return {
                "prediction": "",
                "confidence": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_consensus_prompt(self, expert_predictions: Dict) -> str:
        """Create a prompt for generating consensus"""
        prompt = "Based on the following expert predictions, provide a consensus summary:\n\n"
        for role, prediction in expert_predictions.items():
            prompt += f"{role}: {prediction}\n"
        prompt += "\nConsensus summary:"
        return prompt

    def _calculate_confidence(self, expert_predictions: Dict) -> float:
        """Calculate confidence score based on expert agreement"""
        try:
            # Simple implementation - can be made more sophisticated
            return len(expert_predictions) / len(self.experts)
        except Exception:
            return 0.0

    def update_performance(self, prediction_id: str, actual_outcome: float):
        """Update expert performance metrics"""
        try:
            # Update performance for each expert
            for expert in self.experts:
                self.performance_tracker.update_performance(
                    expert['role'],
                    prediction_id,
                    actual_outcome,
                    actual_outcome > 0
                )
        except Exception as e:
            color_logger.error(f"Error updating performance: {str(e)}")

# Initialize a single instance of the tracker
tracker = AgentPerformanceTracker()

# Initialize historical data
historical_outcomes = [
    {"agent": "Technical Analyst", "prediction": "win", "actual": "win", "pnl": 100},
    {"agent": "Sentiment Analyst", "prediction": "loss", "actual": "loss", "pnl": 50},
    {"agent": "Macro Economist", "prediction": "win", "actual": "win", "pnl": 75},
    {"agent": "Risk Manager", "prediction": "win", "actual": "loss", "pnl": -25}
]

for outcome in historical_outcomes:
    prediction_data = {
        "agent": outcome["agent"],
        "prediction": outcome["prediction"],
        "confidence": 80,
        "pnl": outcome["pnl"]
    }
    tracker.log_prediction(prediction_data)

@color_logger.log_service_call('openai')
async def generate_response(prompt, mode='fast', max_tokens=150, timeframe='short', current_price=None):
    """Generate a response using OpenAI with different modes"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        color_logger.info("="*50)
        color_logger.info(f"🤖 Starting new prediction request - Mode: {mode}")

        # Enrich prompt with context
        enriched_prompt = enrich_prompt_with_context(prompt, current_price)
        color_logger.info(f"📝 Enriched Prompt: {enriched_prompt[:200]}...")

        if mode == 'fast':
            color_logger.info("🚀 Initiating FAST mode")
            messages = [{"role": "user", "content": enriched_prompt}]
            response = await client.chat.completions.create(
                model=FAST_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip()

            # Log the prediction
            prediction_data = {
                'agent': 'FastAnalyst',
                'prediction': result,
                'timeframe': timeframe,
                'current_price': current_price
            }
            tracker.log_prediction(prediction_data)

            return result

        elif mode == 'council':
            color_logger.info("👥 Initiating COUNCIL mode")
            expert_council = ExpertCouncil()
            consensus = await expert_council.get_consensus(enriched_prompt)

            # Log each expert's prediction
            for expert, prediction in consensus['expert_predictions'].items():
                prediction_data = {
                    'agent': expert,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'timeframe': timeframe,
                    'current_price': current_price
                }
                tracker.log_prediction(prediction_data)

            return consensus

    except Exception as e:
        color_logger.error(f"Error in generate_response: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

# New function to update prediction outcomes
def update_prediction_outcome(agent_id: str, actual_price: float):
    """Update the outcome of a prediction"""
    tracker.update_outcome(agent_id, actual_price, datetime.now())

# Add this function after the AgentPerformanceTracker class
def enrich_prompt_with_context(prompt: str, market_data: Dict) -> str:
    """Enrich the prompt with RAG context and market data"""

    # Get relevant context from Pinecone
    try:
        rag_results = query_pinecone(prompt)
        context_data = "\n".join([match['metadata']['text'] for match in rag_results['matches']])
    except Exception as e:
        color_logger.warning(f"Failed to get RAG context: {str(e)}")
        context_data = "No historical context available"

    # Get latest market data from Kalshi
    try:
        kalshi_events = get_events(limit=5)
        market_context = json.dumps(kalshi_events, indent=2)
    except Exception as e:
        color_logger.warning(f"Failed to get Kalshi data: {str(e)}")
        market_context = "No market data available"

    # Enhance prompt with additional context
    enhanced_prompt = f"""
{prompt}

HISTORICAL CONTEXT:
{context_data}

LIVE MARKET DATA:
{market_context}
"""
    return enhanced_prompt

# Update the response parsing
def parse_expert_response(response_text: str) -> Dict:
    """Parse expert response with better error handling"""
    try:
        # Clean the response text
        clean_text = (response_text.replace('```json\n', '')
                     .replace('```', '')
                     .replace('\n', ' ')
                     .strip())

        # Find the first '{' and last '}'
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in response")

        # Extract just the JSON object
        json_str = clean_text[start_idx:end_idx + 1]

        # Parse JSON
        parsed = json.loads(json_str)

        # Validate and clean the response
        cleaned_response = {
            "prediction": str(parsed.get('prediction', '')).strip(),
            "confidence": int(float(parsed.get('confidence', 0))),
            "factors": [str(f).strip() for f in parsed.get('factors', [])[:3]],
            "risks": [str(r).strip() for r in parsed.get('risks', [])[:3]]
        }

        # Validate required fields
        if not cleaned_response["prediction"] or not cleaned_response["factors"]:
            raise ValueError("Missing required fields in response")

        color_logger.info(json.dumps({
            'component': 'expert_response',
            'status': 'success',
            'prediction': cleaned_response["prediction"],
            'confidence': cleaned_response["confidence"]
        }))

        return cleaned_response

    except Exception as e:
        color_logger.error(f"Response parsing error: {str(e)}\nResponse text: {response_text[:200]}...")
        return {
            "prediction": "Error parsing response",
            "confidence": 0,
            "factors": ["Error parsing expert response"],
            "risks": ["Unable to analyze risks due to parsing error"]
        }

def generate_expert_prompt(expert: Dict, enriched_prompt: str) -> str:
    """Generate a more structured expert prompt"""
    expertise_focus = {
        "Technical Analyst": "ONLY analyze price patterns, technical indicators, and market volume data",
        "Sentiment Analyst": "ONLY analyze social media trends, news sentiment, and public opinion",
        "Macro Economist": "ONLY analyze market fundamentals, economic indicators, and industry trends",
        "Risk Manager": "ONLY analyze risk metrics, position sizing, and risk/reward ratios"
    }

    return f"""You are a {expert['role']} with a {expert['bias']} approach and {expert['style']} trading style.
{expertise_focus[expert['role']]}

Return ONLY a valid JSON object with NO additional text or formatting:
{{
    "prediction": "clear win/loss prediction",
    "confidence": number between 0-100,
    "factors": [
        "factor1 specific to {expert['role']}",
        "factor2 specific to {expert['role']}",
        "factor3 specific to {expert['role']}"
    ],
    "risks": [
        "risk1 specific to {expert['role']}",
        "risk2 specific to {expert['role']}",
        "risk3 specific to {expert['role']}"
    ]
}}

Scenario: {enriched_prompt}"""

# Update DataFrame handling
def log_prediction_safe(tracker, agent_role, prediction, timeframe, entry_price):
    prediction_data = {
        'agent': agent_role,
        'prediction': prediction.get('prediction', ''),
        'confidence': float(prediction.get('confidence', 0)),
        'timeframe': timeframe,
        'entry_price': float(entry_price)
    }
    tracker.log_prediction(prediction_data)

# Export the classes and functions
__all__ = ['AgentPerformanceTracker', 'ExpertCouncil', 'generate_response', 'tracker']
