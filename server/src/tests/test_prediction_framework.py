import asyncio
import sys
import os
from datetime import datetime
from typing import Dict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.prediction_framework import PredictionFramework
from utils.logger import color_logger

class PredictionFrameworkTester:
    def __init__(self):
        self.framework = PredictionFramework()

    async def test_full_prediction_cycle(self):
        """Test the entire prediction framework cycle"""
        color_logger.info("🏁 Starting full prediction cycle test")

        try:
            # 1. Test data gathering
            color_logger.info("📥 Testing data gathering from all sources...")
            raw_data = await self.framework._gather_data()
            self._log_data_gathering_results(raw_data)

            # 2. Test data processing
            color_logger.info("🔄 Testing data processing...")
            processed_data = self.framework._process_data(raw_data)
            self._log_data_processing_results(processed_data)

            # 3. Test prediction generation
            color_logger.info("🤖 Testing prediction generation...")
            test_query = "Will BTC-USD close above $50,000 by end of week?"
            prediction_result = await self.framework.generate_prediction(test_query)
            self._log_prediction_results(prediction_result)

            # 4. Test performance tracking
            color_logger.info("📊 Testing performance tracking...")
            if prediction_result.get('prediction_id'):
                self.framework.update_model_performance(
                    prediction_result['prediction_id'],
                    actual_outcome=45000.0  # Example outcome
                )

            color_logger.success("✅ Full prediction cycle test completed successfully!")
            return True

        except Exception as e:
            color_logger.error(f"❌ Error during prediction cycle test: {str(e)}")
            return False

    def _log_data_gathering_results(self, raw_data: Dict):
        """Log results from data gathering phase"""
        for source, data in raw_data.items():
            if data:
                color_logger.success(f"✅ Successfully gathered data from {source}")
                color_logger.info(f"📊 {source} data sample: {str(data)[:200]}...")
            else:
                color_logger.error(f"❌ Failed to gather data from {source}")

    def _log_data_processing_results(self, processed_data: Dict):
        """Log results from data processing phase"""
        for source, data in processed_data.items():
            if data:
                color_logger.success(f"✅ Successfully processed {source} data")
                color_logger.info(f"📊 Processed {source} metrics: {str(data)[:200]}...")
            else:
                color_logger.error(f"❌ Failed to process {source} data")

    def _log_prediction_results(self, prediction_result: Dict):
        """Log results from prediction generation phase"""
        if prediction_result.get('success'):
            color_logger.success("✅ Successfully generated prediction")
            color_logger.prediction(f"🎯 Prediction: {prediction_result.get('prediction', {}).get('consensus', {})}")
            color_logger.info(f"📊 Confidence: {prediction_result.get('prediction', {}).get('confidence', 0)}")
            color_logger.info(f"💡 Supporting Data: {str(prediction_result.get('supporting_data', ''))[:200]}...")
        else:
            color_logger.error(f"❌ Failed to generate prediction: {prediction_result.get('error')}")

def run_tests():
    """Run all prediction framework tests"""
    color_logger.info("🚀 Starting Prediction Framework Tests")

    # Create test instance
    tester = PredictionFrameworkTester()

    # Run tests
    asyncio.run(tester.test_full_prediction_cycle())

if __name__ == "__main__":
    run_tests()
