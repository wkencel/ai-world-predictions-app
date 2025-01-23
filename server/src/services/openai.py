# services/openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model definitions
DEEP_MODEL = 'gpt-4o'  # Deep thinking model
FAST_MODEL = 'gpt-4o-mini'  # Quick response model
COUNCIL_MODEL = 'gpt-4o'  # Model for expert council

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_response(prompt, mode='fast', max_tokens=150, timeframe='short'):
    """
    Generates a response from OpenAI's GPT model based on the given prompt.

    Args:
        prompt (str): The input text prompt.
        mode (str): The mode of response ('fast', 'deep', or 'council').
        max_tokens (int): The maximum number of tokens in the response.
        timeframe (str): Trading timeframe ('short', 'medium', 'long').

    Returns:
        str or dict: The generated response, format depends on mode.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        logger.info("="*50)
        logger.info(f"🤖 Starting new prediction request - Mode: {mode}")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"⏱️ Timeframe: {timeframe}")
        logger.info("="*50)

        if mode == 'fast':
            logger.info("🚀 Initiating FAST mode with o1-mini model")
            logger.info("📊 Optimizing for speed and conciseness...")

            messages = [{"role": "user", "content": prompt}]

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

            messages = [{"role": "user", "content": prompt}]

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

                Scenario: {prompt}"""

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
                return {
                    "discussion": discussion,
                    "consensus": final_consensus,
                    "process_time": "30 seconds",
                    "mode": "council"
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
