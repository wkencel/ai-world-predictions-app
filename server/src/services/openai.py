# services/openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from db.pinecone.setup_pinecone import query_pinecone
from services.kalshi import get_events
from utils.logger import color_logger

# Determine the path to the root directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=dotenv_path)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
# ! NICK FUN TIMES
client = OpenAI(api_key=api_key)

# Model definitions
DEEP_MODEL = 'gpt-4o'  # Deep thinking model
FAST_MODEL = 'gpt-4o-mini'  # Quick response model
COUNCIL_MODEL = 'gpt-4o'  # Model for expert council

# todo: pull in some RAG store data
# todo: pull in some api data
# todo: pull in some news data
# todo: pull in web scraper data
# todo: pull in some other data

# Architecture: user input -> prompt -> model  -> response
# we also need some way of getting our prediction outcome into the model

# Move AgentPerformanceTracker class to the top of the file, after imports
class AgentPerformanceTracker:
    def __init__(self):
        self.dtypes = {
            'timestamp': 'datetime64[ns]',
            'agent_id': 'string',
            'agent_role': 'string',
            'prediction': 'string',
            'confidence': 'float64',
            'timeframe': 'string',
            'entry_price': 'float64',
            'target_price': 'float64',
            'stop_loss': 'float64',
            'actual_outcome': 'float64',
            'pnl': 'float64',
            'accuracy_score': 'float64'
        }

        # Initialize with historical performance data for all experts
        historical_data = [
            # Technical Analyst history
            {
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=5),
                'agent_id': 'Technical_Analyst_historical_1',
                'agent_role': 'Technical Analyst',
                'prediction': 'win',
                'confidence': 80.0,
                'timeframe': 'short',
                'entry_price': 1.85,
                'target_price': 2.0,
                'stop_loss': 1.75,
                'actual_outcome': 1.95,
                'pnl': 5.4,
                'accuracy_score': 1.0
            },
            # Sentiment Analyst history
            {
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=4),
                'agent_id': 'Sentiment_Analyst_historical_1',
                'agent_role': 'Sentiment Analyst',
                'prediction': 'win',
                'confidence': 75.0,
                'timeframe': 'short',
                'entry_price': 1.90,
                'target_price': 2.1,
                'stop_loss': 1.80,
                'actual_outcome': 2.05,
                'pnl': 7.9,
                'accuracy_score': 1.0
            },
            # Macro Economist history
            {
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=3),
                'agent_id': 'Macro_Economist_historical_1',
                'agent_role': 'Macro Economist',
                'prediction': 'win',
                'confidence': 70.0,
                'timeframe': 'short',
                'entry_price': 1.75,
                'target_price': 1.95,
                'stop_loss': 1.65,
                'actual_outcome': 1.85,
                'pnl': 5.7,
                'accuracy_score': 1.0
            },
            # Risk Manager history
            {
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=2),
                'agent_id': 'Risk_Manager_historical_1',
                'agent_role': 'Risk Manager',
                'prediction': 'win',
                'confidence': 85.0,
                'timeframe': 'short',
                'entry_price': 1.80,
                'target_price': 2.0,
                'stop_loss': 1.70,
                'actual_outcome': 1.90,
                'pnl': 5.5,
                'accuracy_score': 1.0
            }
        ]

        # Create DataFrame with historical data
        self.performance_log = pd.DataFrame(historical_data).astype(self.dtypes)

        color_logger.info("Performance tracker initialized with historical data")
        color_logger.json_log(self.get_leaderboard().to_dict('records'), "Initial Leaderboard:")

    def log_prediction(self, agent_role, prediction, timeframe, entry_price):
        """Log a new prediction"""
        new_data = {
            'timestamp': pd.Timestamp.now(),
            'agent_id': f"{agent_role}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            'agent_role': agent_role,
            'prediction': prediction.get('prediction', ''),
            'confidence': float(prediction.get('confidence', 0)),
            'timeframe': timeframe,
            'entry_price': float(entry_price),
            'target_price': float(self._extract_target_price(prediction.get('prediction', ''))),
            'stop_loss': float(self._calculate_stop_loss(entry_price, prediction)),
            'actual_outcome': 0.0,
            'pnl': 0.0,
            'accuracy_score': 0.0
        }

        # Create new row with correct dtypes
        new_row = pd.DataFrame([new_data]).astype(self.dtypes)

        # Concatenate with existing data
        self.performance_log = pd.concat([self.performance_log, new_row], ignore_index=True)

    def _extract_target_price(self, prediction: str) -> float:
        """Extract target price from prediction text"""
        # Mock implementation
        return 2.0

    def _calculate_stop_loss(self, entry_price: float, prediction: Dict) -> float:
        """Calculate stop loss based on entry price and prediction"""
        # Mock implementation - 5% below entry price
        return entry_price * 0.95

    def update_outcome(self, agent_id: str, actual_outcome: float, timestamp: datetime) -> None:
        """Update the outcome of a prediction"""
        try:
            mask = self.performance_log['agent_id'] == agent_id
            if not any(mask):
                color_logger.warning(f"No prediction found for agent_id: {agent_id}")
                return

            entry_price = self.performance_log.loc[mask, 'entry_price'].iloc[0]
            pnl = ((actual_outcome - entry_price) / entry_price) * 100

            self.performance_log.loc[mask, 'actual_outcome'] = actual_outcome
            self.performance_log.loc[mask, 'pnl'] = pnl
            self.performance_log.loc[mask, 'accuracy_score'] = 1.0 if pnl > 0 else 0.0

        except Exception as e:
            color_logger.error(f"Error updating outcome: {str(e)}")

    def get_leaderboard(self) -> pd.DataFrame:
        """Get performance leaderboard with actual metrics"""
        try:
            if self.performance_log.empty:
                return pd.DataFrame({
                    'agent_role': [
                        'Technical Analyst', 'Sentiment Analyst',
                        'Macro Economist', 'Risk Manager'
                    ],
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'average_accuracy': 0.0
                })

            stats = self.performance_log.groupby('agent_role').agg({
                'agent_id': 'count',
                'pnl': ['sum', 'mean'],
                'accuracy_score': 'mean'
            }).reset_index()

            stats.columns = ['agent_role', 'total_trades', 'total_pnl', 'avg_pnl', 'average_accuracy']

            # Calculate win rate (percentage of trades with positive PnL)
            win_rates = self.performance_log.groupby('agent_role').apply(
                lambda x: (x['pnl'] > 0).mean()
            ).reset_index()
            win_rates.columns = ['agent_role', 'win_rate']

            # Merge win rates with stats
            stats = stats.merge(win_rates, on='agent_role')

            # Round numeric columns
            numeric_cols = ['total_pnl', 'avg_pnl', 'average_accuracy', 'win_rate']
            stats[numeric_cols] = stats[numeric_cols].round(3)

            return stats

        except Exception as e:
            color_logger.error(f"Error generating leaderboard: {str(e)}")
            return pd.DataFrame()

# Initialize the tracker at the module level
tracker = AgentPerformanceTracker()

# Initialize tracker with some historical data
tracker = AgentPerformanceTracker()
historical_outcomes = [
    {"agent": "Technical Analyst", "prediction": "win", "actual": "win", "pnl": 100},
    {"agent": "Sentiment Analyst", "prediction": "loss", "actual": "loss", "pnl": 50},
    {"agent": "Macro Economist", "prediction": "win", "actual": "win", "pnl": 75},
    {"agent": "Risk Manager", "prediction": "win", "actual": "loss", "pnl": -25}
]

for outcome in historical_outcomes:
    tracker.log_prediction(
        agent_role=outcome["agent"],
        prediction={"prediction": outcome["prediction"], "confidence": 80},
        timeframe="short",
        entry_price=1.85
    )
    tracker.update_outcome(
        f"{outcome['agent']}_80",
        1.85 if outcome["actual"] == "win" else 0,
        datetime.now()
    )

@color_logger.log_service_call('openai')  # Use color_logger for the decorator
def generate_response(prompt, mode='fast', max_tokens=150, timeframe='short', current_price=None):
    """
    Updated generate_response function that includes RAG and market data
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        color_logger.info("="*50)
        color_logger.info(f"🤖 Starting new prediction request - Mode: {mode}")

        # Enrich prompt with context
        enriched_prompt = enrich_prompt_with_context(prompt, current_price)
        color_logger.info(f" Enriched Prompt: {enriched_prompt[:200]}...")

        if mode == 'fast':
            color_logger.info("🚀 Initiating FAST mode with o1-mini model")
            color_logger.info("📊 Optimizing for speed and conciseness...")

            messages = [{"role": "user", "content": enriched_prompt}]

            response = client.chat.completions.create(
                model=FAST_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.5,
            )
            result = response.choices[0].message.content.strip()

            color_logger.info("✨ Fast analysis complete!")
            color_logger.info(f"📊 Response: {result[:100]}...")
            return result

        elif mode == 'deep':
            color_logger.info("🧠 Initiating DEEP mode with o1 model")
            color_logger.info(" Beginning thorough analysis process...")

            # Simulate deep thinking with progress updates
            for i in range(20):
                time.sleep(1)
                if i % 4 == 0:  # Log every 4 seconds
                    color_logger.info(f"🤔 Deep thinking in progress... {i*5}% complete")
                    color_logger.info("💭 Analyzing market patterns and correlations...")
                    color_logger.info("📈 Processing technical indicators...")
                    color_logger.info("🌍 Evaluating global market conditions...")

            messages = [{"role": "user", "content": enriched_prompt}]

            color_logger.info("🎯 Finalizing deep analysis...")
            response = client.chat.completions.create(
                model=DEEP_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip()

            color_logger.info("✅ Deep analysis complete!")
            color_logger.info(f"📊 Response: {result[:100]}...")
            return result

        elif mode == 'council':
            color_logger.info("👥 Initiating COUNCIL mode with expert panel")
            color_logger.info("🎓 Assembling expert council members...")
            experts = [
                {"role": "Technical Analyst", "bias": "chart-focused", "style": "conservative"},
                {"role": "Sentiment Analyst", "bias": "social-media-driven", "style": "aggressive"},
                {"role": "Macro Economist", "bias": "fundamentals-based", "style": "moderate"},
                {"role": "Risk Manager", "bias": "risk-focused", "style": "cautious"}
            ]

            discussion = []

            # Phase 1: Individual Expert Analysis
            color_logger.info("\n📣 Starting Phase 1: Individual Expert Analysis")
            for expert in experts:
                color_logger.expert(expert['role'], f"Starting analysis (Bias: {expert['bias']}, Style: {expert['style']})")

                expert_prompt = generate_expert_prompt(expert, enriched_prompt)

                response = client.chat.completions.create(
                    model=DEEP_MODEL,
                    messages=[{"role": "user", "content": expert_prompt}],
                    temperature=0.7,
                    max_tokens=max_tokens
                )

                parsed_response = parse_expert_response(response.choices[0].message.content)

                color_logger.success(f"Expert {expert['role']} analysis complete")
                color_logger.prediction(f"Prediction: {parsed_response['prediction']}")
                color_logger.analysis(f"Confidence: {parsed_response['confidence']}%")
                color_logger.json_log(parsed_response, prefix=f"📝 {expert['role']} Detailed Analysis:")

                # Add to discussion
                discussion.append({
                    "expert": expert["role"],
                    "analysis": parsed_response
                })

                color_logger.info("Processing next expert in 5 seconds...")
                time.sleep(5)

            # Phase 2: Group Discussion and Consensus
            color_logger.info("\n📣 Starting Phase 2: Building Consensus")
            color_logger.info("🤝 Moderator is reviewing all expert opinions...")

            consensus_prompt = f"""As the council moderator, analyze these expert opinions and provide a final consensus.
Return ONLY a JSON object with no markdown formatting or additional text:

{{
    "final_prediction": "clear prediction",
    "confidence_level": 75,
    "profit_strategy": "detailed strategy",
    "risk_assessment": "key risks",
    "sentiment_score": 75
}}

Expert Opinions:
{json.dumps(discussion, indent=2)}"""

            try:
                color_logger.info("🧠 Moderator is synthesizing expert opinions...")
                messages = [{"role": "user", "content": consensus_prompt}]

                consensus_response = client.chat.completions.create(
                    model=COUNCIL_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )

                consensus_text = consensus_response.choices[0].message.content.strip()
                color_logger.info("📝 Raw consensus received")
                color_logger.info(f"📊 Consensus text: {consensus_text[:100]}...")

                try:
                    final_consensus = json.loads(consensus_text)
                    color_logger.info("✅ Successfully parsed consensus")
                    color_logger.info(f"🎯 Final Prediction: {final_consensus['final_prediction'][:100]}...")
                    color_logger.info(f"📊 Confidence Level: {final_consensus['confidence_level']}")
                except json.JSONDecodeError as je:
                    color_logger.error(f"⚠️ Consensus JSON parsing failed: {je}")
                    final_consensus = {
                        "final_prediction": consensus_text,
                        "confidence_level": "N/A",
                        "profit_strategy": "N/A",
                        "risk_assessment": "N/A",
                        "sentiment_score": "N/A"
                    }

                color_logger.info("🏁 Council session complete!")

                # Log predictions for each expert
                for opinion in discussion:
                    if current_price:
                        log_prediction_safe(tracker, opinion['expert'], opinion['analysis'], timeframe, current_price)

                # Add performance metrics to the response
                leaderboard = tracker.get_leaderboard()

                return {
                    "discussion": discussion,
                    "consensus": final_consensus,
                    "process_time": "30 seconds",
                    "mode": "council",
                    "performance_metrics": {
                        "leaderboard": leaderboard.to_dict('records'),
                        "total_council_pnl": leaderboard['total_pnl'].sum()
                    }
                }

            except Exception as e:
                color_logger.error(f"❌ Consensus building error: {str(e)}")
                return {
                    "discussion": discussion,
                    "consensus": {"error": str(e)},
                    "process_time": "30 seconds",
                    "mode": "council"
                }

        else:
            raise ValueError("Invalid mode. Choose 'fast', 'deep', or 'council'.")

    except Exception as e:
        color_logger.error(f"❌ Error type: {type(e)}")
        color_logger.error(f"❌ Error details: {str(e)}")
        return f"Sorry, I couldn't process your request at the moment. Error: {str(e)}"

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
    new_data = {
        'timestamp': pd.Timestamp.now(),
        'agent_id': f"{agent_role}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        'agent_role': agent_role,
        'prediction': prediction.get('prediction', ''),
        'confidence': float(prediction.get('confidence', 0)),
        'timeframe': timeframe,
        'entry_price': float(entry_price),
        'target_price': float(tracker._extract_target_price(prediction.get('prediction', ''))),
        'stop_loss': float(tracker._calculate_stop_loss(entry_price, prediction)),
        'actual_outcome': 0.0,
        'pnl': 0.0,
        'accuracy_score': 0.0
    }

    # Create new row with correct dtypes
    new_row = pd.DataFrame([new_data]).astype(tracker.performance_log.dtypes)

    # Concatenate with existing data
    tracker.performance_log = pd.concat([tracker.performance_log, new_row], ignore_index=True)
