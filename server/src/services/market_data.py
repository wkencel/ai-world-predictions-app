from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from utils.logger import color_logger
from services.kalshi import get_markets, get_market, get_trades

class MarketDataCollector:
    """Collects and processes market data from various sources"""

    def __init__(self):
        self.markets_cache = {}
        self.cache_expiry = {}
        self.CACHE_DURATION = timedelta(minutes=5)

    async def fetch_data(self):
        return {}

    def process_data(self, data):
        return {}

    def _calculate_market_metrics(self, markets_data: Dict, trades_data: Dict) -> Dict:
        """Calculate various market metrics"""
        try:
            markets = markets_data.get('markets', [])
            trades = trades_data.get('trades', [])

            return {
                'total_markets': len(markets),
                'active_markets': len([m for m in markets if m.get('status') == 'active']),
                'total_volume_24h': sum(float(t.get('volume', 0)) for t in trades),
                'average_price': self._calculate_average_price(trades)
            }
        except Exception as e:
            color_logger.error(f"Error calculating market metrics: {str(e)}")
            return {}

    def _calculate_average_odds(self, markets: List[Dict]) -> float:
        """Calculate average odds across markets"""
        try:
            odds = [float(m.get('odds', 0)) for m in markets if m.get('odds')]
            return sum(odds) / len(odds) if odds else 0
        except Exception:
            return 0

    def _calculate_momentum(self, trades: List[Dict]) -> float:
        """Calculate price momentum from recent trades"""
        try:
            if not trades:
                return 0

            df = pd.DataFrame(trades)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate price changes
            price_changes = df['price'].diff()
            return float(price_changes.mean())

        except Exception:
            return 0

    def _analyze_market_sentiment(self, markets: List[Dict], trades: List[Dict]) -> float:
        """Analyze market sentiment based on trading activity"""
        try:
            # Simple sentiment calculation based on buy vs sell pressure
            buy_volume = sum(float(t.get('volume', 0)) for t in trades if t.get('side') == 'buy')
            sell_volume = sum(float(t.get('volume', 0)) for t in trades if t.get('side') == 'sell')

            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return 0

            # Return sentiment score between -1 and 1
            return (buy_volume - sell_volume) / total_volume

        except Exception:
            return 0

    def _calculate_average_price(self, trades: List[Dict]) -> float:
        """Calculate volume-weighted average price"""
        try:
            total_volume = 0
            volume_price = 0

            for trade in trades:
                volume = float(trade.get('volume', 0))
                price = float(trade.get('price', 0))
                volume_price += volume * price
                total_volume += volume

            return volume_price / total_volume if total_volume > 0 else 0

        except Exception:
            return 0
