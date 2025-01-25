import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.webScraping.scraper import WebScraper
from db.pinecone.setup_pinecone import PineconeManager
from services.sentiment import SentimentAnalyzer
from utils.logger import color_logger

async def test_full_system():
    """Test the entire system flow"""
    try:
        # 1. Test WebScraper
        color_logger.info("🧪 Testing WebScraper...")
        scraper = WebScraper()
        scraped_data = await scraper.fetch_data()
        if not scraped_data:
            color_logger.warning("⚠️ WebScraper returned empty data")
        else:
            color_logger.info(f"✅ WebScraper found {len(scraped_data.get('news', []))} news items")

        # 2. Test Pinecone Connection
        color_logger.info("🧪 Testing Pinecone...")
        pinecone = PineconeManager()
        test_data = {
            'vector': [0.1] * 1536,
            'metadata': {'text': 'Test data', 'timestamp': datetime.now().isoformat()},
            'source': 'system_test'
        }
        store_result = await pinecone.store_data(test_data)
        if store_result:
            color_logger.info("✅ Pinecone storage test passed")

        # 3. Test Sentiment Analysis
        color_logger.info("🧪 Testing SentimentAnalyzer...")
        sentiment = SentimentAnalyzer()
        # Use the correct method from SentimentAnalyzer
        sentiment_result = sentiment._analyze_texts(["This is a positive test message"])
        if sentiment_result:
            color_logger.info(f"✅ Sentiment score: {sentiment_result.get('score', 0)}")

        color_logger.info("✅ All system tests completed!")
        return True

    except Exception as e:
        color_logger.error(f"❌ System test failed: {str(e)}")
        return False

def main():
    """Main entry point for system tests"""
    try:
        # Load environment variables
        load_dotenv()

        # Run the test
        import asyncio
        asyncio.run(test_full_system())

    except KeyboardInterrupt:
        color_logger.info("Test interrupted by user")
    except Exception as e:
        color_logger.error(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    main()
