# services/openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from db.pinecone.setup_pinecone import query_pinecone
from services.kalshi import get_events

# Determine the path to the root directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=dotenv_path)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
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

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# New class to track agent performance
class AgentPerformanceTracker:
    def __init__(self):
        self.performance_log = pd.DataFrame(columns=[
            'timestamp',
            'agent_id',
            'agent_role',
            'prediction',
            'confidence',
            'timeframe',
            'entry_price',
            'target_price',
            'stop_loss',
            'actual_outcome',
            'pnl',
            'accuracy_score'
        ])

        # Agent performance metrics
        self.agent_metrics = {
            'Technical Analyst': {'wins': 0, 'losses': 0, 'pnl': 0.0},
            'Sentiment Analyst': {'wins': 0, 'losses': 0, 'pnl': 0.0},
            'Macro Economist': {'wins': 0, 'losses': 0, 'pnl': 0.0},
            'Risk Manager': {'wins': 0, 'losses': 0, 'pnl': 0.0}
        }

    def log_prediction(self,
                      agent_role: str,
                      prediction: Dict,
                      timeframe: str,
                      entry_price: float) -> None:
        """Log a new prediction from an agent"""
        self.performance_log = self.performance_log.append({
            'timestamp': datetime.now(),
            'agent_id': f"{agent_role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'agent_role': agent_role,
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'timeframe': timeframe,
            'entry_price': entry_price,
            'target_price': self._extract_target_price(prediction['prediction']),
            'stop_loss': self._calculate_stop_loss(entry_price, prediction),
            'actual_outcome': None,
            'pnl': 0.0,
            'accuracy_score': 0.0
        }, ignore_index=True)

    def update_outcome(self,
                      agent_id: str,
                      actual_price: float,
                      timestamp: datetime) -> None:
        """Update the actual outcome and calculate P&L"""
        idx = self.performance_log[self.performance_log['agent_id'] == agent_id].index
        if len(idx) > 0:
            prediction = self.performance_log.loc[idx[0]]

            # Calculate P&L
            pnl = self._calculate_pnl(
                prediction['entry_price'],
                actual_price,
                prediction['target_price'],
                prediction['stop_loss']
            )

            # Update performance log
            self.performance_log.loc[idx[0], 'actual_outcome'] = actual_price
            self.performance_log.loc[idx[0], 'pnl'] = pnl
            self.performance_log.loc[idx[0], 'accuracy_score'] = self._calculate_accuracy(
                prediction['target_price'],
                actual_price
            )

            # Update agent metrics
            agent_role = prediction['agent_role']
            self.agent_metrics[agent_role]['pnl'] += pnl
            if pnl > 0:
                self.agent_metrics[agent_role]['wins'] += 1
            else:
                self.agent_metrics[agent_role]['losses'] += 1

    def get_agent_performance(self, agent_role: str) -> Dict:
        """Get performance metrics for a specific agent"""
        metrics = self.agent_metrics[agent_role]
        total_trades = metrics['wins'] + metrics['losses']

        return {
            'win_rate': metrics['wins'] / total_trades if total_trades > 0 else 0,
            'total_pnl': metrics['pnl'],
            'total_trades': total_trades,
            'average_accuracy': self.performance_log[
                self.performance_log['agent_role'] == agent_role
            ]['accuracy_score'].mean()
        }

    def get_leaderboard(self) -> pd.DataFrame:
        """Generate a leaderboard of agent performance"""
        return pd.DataFrame([
            {
                'agent_role': role,
                **self.get_agent_performance(role)
            }
            for role in self.agent_metrics.keys()
        ]).sort_values('total_pnl', ascending=False)

    def _extract_target_price(self, prediction: str) -> Optional[float]:
        """Extract target price from prediction text"""
        # TODO: Implement price extraction logic
        # This would use regex or NLP to extract price targets from prediction text
        pass

    def _calculate_stop_loss(self, entry_price: float, prediction: Dict) -> float:
        """Calculate stop loss based on prediction confidence"""
        confidence = float(prediction['confidence'])
        # More confident predictions get wider stops
        stop_percentage = (100 - confidence) / 100 * 0.05  # 5% max stop
        return entry_price * (1 - stop_percentage)

    def _calculate_pnl(self,
                      entry: float,
                      actual: float,
                      target: float,
                      stop: float) -> float:
        """Calculate P&L for a trade"""
        # Implement P&L calculation logic based on your trading rules
        pass

    def _calculate_accuracy(self,
                          predicted: float,
                          actual: float) -> float:
        """Calculate prediction accuracy score"""
        # Implement accuracy calculation logic
        pass

# Update the generate_response function to use the tracker
tracker = AgentPerformanceTracker()

def generate_response(prompt, mode='fast', max_tokens=150, timeframe='short', current_price=None):
    """
    Updated generate_response function that includes RAG and market data
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        logger.info("="*50)
        logger.info(f"🤖 Starting new prediction request - Mode: {mode}")

        # Enrich prompt with context
        enriched_prompt = enrich_prompt_with_context(prompt, current_price)
        logger.info(f"📝 Enriched Prompt: {enriched_prompt[:200]}...")

        if mode == 'fast':
            logger.info("🚀 Initiating FAST mode with o1-mini model")
            logger.info("📊 Optimizing for speed and conciseness...")

            messages = [{"role": "user", "content": enriched_prompt}]

            response = client.chat.completions.create(
                model=FAST_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.5,
            )
            result = response.choices[0].message.content.strip()

            logger.info("✨ Fast analysis complete!")
            logger.info(f"📊 Response: {result[:100]}...")
            return result

        elif mode == 'deep':
            logger.info("🧠 Initiating DEEP mode with o1 model")
            logger.info("📚 Beginning thorough analysis process...")

            # Simulate deep thinking with progress updates
            for i in range(20):
                time.sleep(1)
                if i % 4 == 0:  # Log every 4 seconds
                    logger.info(f"🤔 Deep thinking in progress... {i*5}% complete")
                    logger.info("💭 Analyzing market patterns and correlations...")
                    logger.info("📈 Processing technical indicators...")
                    logger.info("🌍 Evaluating global market conditions...")

            messages = [{"role": "user", "content": enriched_prompt}]

            logger.info("🎯 Finalizing deep analysis...")
            response = client.chat.completions.create(
                model=DEEP_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip()

            logger.info("✅ Deep analysis complete!")
            logger.info(f"📊 Response: {result[:100]}...")
            return result

        elif mode == 'council':
            logger.info("👥 Initiating COUNCIL mode with expert panel")
            logger.info("🎓 Assembling expert council members...")
            experts = [
                {"role": "Technical Analyst", "bias": "chart-focused", "style": "conservative"},
                {"role": "Sentiment Analyst", "bias": "social-media-driven", "style": "aggressive"},
                {"role": "Macro Economist", "bias": "fundamentals-based", "style": "moderate"},
                {"role": "Risk Manager", "bias": "risk-focused", "style": "cautious"}
            ]

            discussion = []

            # Phase 1: Individual Expert Analysis
            logger.info("\n📣 Starting Phase 1: Individual Expert Analysis")
            for expert in experts:
                logger.info(f"\n👤 Consulting {expert['role']}...")
                logger.info(f"💭 Expert Bias: {expert['bias']}")
                logger.info(f"🎯 Trading Style: {expert['style']}")

                expert_prompt = f"""You are a {expert['role']} with a {expert['bias']} approach
                and {expert['style']} trading style. Analyze this scenario and provide your response in this EXACT format:
                {{
                    "prediction": "your specific prediction here",
                    "confidence": "confidence level 0-100",
                    "factors": ["factor1", "factor2", "factor3"],
                    "risks": ["risk1", "risk2", "risk3"]
                }}

                Scenario: {enriched_prompt}"""

                try:
                    logger.info("🤔 Expert is analyzing the scenario...")
                    messages = [{"role": "user", "content": expert_prompt}]

                    response = client.chat.completions.create(
                        model=COUNCIL_MODEL,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )

                    expert_response = response.choices[0].message.content.strip()
                    logger.info(f"📝 Raw expert response received: {expert_response[:100]}...")

                    try:
                        expert_opinion = json.loads(expert_response)
                        logger.info("✅ Successfully parsed expert opinion")
                        logger.info(f"🎯 Prediction: {expert_opinion['prediction'][:100]}...")
                        logger.info(f"📊 Confidence: {expert_opinion['confidence']}")
                    except json.JSONDecodeError as je:
                        logger.error(f"⚠️ JSON parsing failed: {je}")
                        expert_opinion = {
                            "prediction": expert_response,
                            "confidence": "N/A",
                            "factors": [],
                            "risks": []
                        }

                    discussion.append({
                        "expert": expert['role'],
                        "analysis": expert_opinion
                    })

                except Exception as e:
                    logger.error(f"❌ Expert consultation error: {str(e)}")
                    discussion.append({
                        "expert": expert['role'],
                        "analysis": {"error": str(e)}
                    })

                logger.info("⏳ Processing next expert in 5 seconds...")
                time.sleep(5)

            # Phase 2: Group Discussion and Consensus
            logger.info("\n📣 Starting Phase 2: Building Consensus")
            logger.info("🤝 Moderator is reviewing all expert opinions...")

            consensus_prompt = f"""As the council moderator, analyze these expert opinions and provide a final consensus in this EXACT format:
            {{
                "final_prediction": "specific prediction here",
                "confidence_level": "0-100",
                "profit_strategy": "detailed strategy here",
                "risk_assessment": "risk assessment here",
                "sentiment_score": "0-100"
            }}

            Expert Opinions:
            {json.dumps(discussion, indent=2)}"""

            try:
                logger.info("🧠 Moderator is synthesizing expert opinions...")
                messages = [{"role": "user", "content": consensus_prompt}]

                consensus_response = client.chat.completions.create(
                    model=COUNCIL_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )

                consensus_text = consensus_response.choices[0].message.content.strip()
                logger.info("📝 Raw consensus received")
                logger.info(f"📊 Consensus text: {consensus_text[:100]}...")

                try:
                    final_consensus = json.loads(consensus_text)
                    logger.info("✅ Successfully parsed consensus")
                    logger.info(f"🎯 Final Prediction: {final_consensus['final_prediction'][:100]}...")
                    logger.info(f"📊 Confidence Level: {final_consensus['confidence_level']}")
                except json.JSONDecodeError as je:
                    logger.error(f"⚠️ Consensus JSON parsing failed: {je}")
                    final_consensus = {
                        "final_prediction": consensus_text,
                        "confidence_level": "N/A",
                        "profit_strategy": "N/A",
                        "risk_assessment": "N/A",
                        "sentiment_score": "N/A"
                    }

                logger.info("🏁 Council session complete!")

                # Log predictions for each expert
                for opinion in discussion:
                    if current_price:
                        tracker.log_prediction(
                            agent_role=opinion['expert'],
                            prediction=opinion['analysis'],
                            timeframe=timeframe,
                            entry_price=current_price
                        )

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
                logger.error(f"❌ Consensus building error: {str(e)}")
                return {
                    "discussion": discussion,
                    "consensus": {"error": str(e)},
                    "process_time": "30 seconds",
                    "mode": "council"
                }

        else:
            raise ValueError("Invalid mode. Choose 'fast', 'deep', or 'council'.")

    except Exception as e:
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {str(e)}")
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
        logger.warning(f"Failed to get RAG context: {str(e)}")
        context_data = "No historical context available"

    # Get latest market data from Kalshi
    try:
        kalshi_events = get_events(limit=5)
        market_context = json.dumps(kalshi_events, indent=2)
    except Exception as e:
        logger.warning(f"Failed to get Kalshi data: {str(e)}")
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
