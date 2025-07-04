"""
Advanced Trading System - Market Data Aggregator
Unified data collection and distribution system for multiple data sources.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import aiohttp
import websockets
import json
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    mark_price: float = 0.0

@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: float
    size: float

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float = 0.0
    spread: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0

@dataclass
class MarketSummary:
    """Market summary statistics."""
    symbol: str
    last_price: float
    volume_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    high_24h: float
    low_24h: float
    volatility_estimate: float = 0.0
    liquidity_score: float = 0.0

class MarketDataAggregator:
    """
    Unified market data aggregation system.
    
    Collects data from multiple sources (exchanges, APIs) and provides
    a unified interface for strategies to access market information.
    """
    
    def __init__(self):
        """Initialize the market data aggregator."""
        self.is_active = False
        self.data_feeds: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Data storage
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.market_summaries: Dict[str, MarketSummary] = {}
        self.funding_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Data processing
        self.data_processors: List[asyncio.Task] = []
        self.last_update_times: Dict[str, datetime] = {}
        
        # Configuration
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.exchanges = ["binance", "bybit", "okex"]
        self.update_frequency = 1.0  # seconds
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("MarketDataAggregator initialized")
    
    async def start(self) -> bool:
        """Start data aggregation."""
        try:
            self.logger.info("Starting market data aggregation...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=100)
            )
            
            # Start data feeds
            await self._start_data_feeds()
            
            # Start data processing tasks
            self.data_processors = [
                asyncio.create_task(self._price_data_processor()),
                asyncio.create_task(self._order_book_processor()),
                asyncio.create_task(self._funding_rate_processor()),
                asyncio.create_task(self._market_summary_processor())
            ]
            
            self.is_active = True
            self.logger.info("Market data aggregation started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting market data aggregation: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop data aggregation."""
        try:
            self.logger.info("Stopping market data aggregation...")
            
            self.is_active = False
            
            # Cancel processing tasks
            for task in self.data_processors:
                task.cancel()
            
            # Close websocket connections
            for connection in self.websocket_connections.values():
                if hasattr(connection, 'close'):
                    await connection.close()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            self.logger.info("Market data aggregation stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping market data aggregation: {e}")
            return False
    
    async def _start_data_feeds(self):
        """Initialize data feeds from various sources."""
        try:
            # Start Binance data feeds
            await self._start_binance_feeds()
            
            # Start additional exchange feeds (would implement for production)
            # await self._start_bybit_feeds()
            # await self._start_okex_feeds()
            
        except Exception as e:
            self.logger.error(f"Error starting data feeds: {e}")
    
    async def _start_binance_feeds(self):
        """Start Binance data feeds."""
        try:
            # Binance WebSocket streams
            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                streams.extend([
                    f"{symbol_lower}@ticker",
                    f"{symbol_lower}@depth20@100ms",
                    f"{symbol_lower}@trade"
                ])
            
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            # Start WebSocket connection task
            asyncio.create_task(self._binance_websocket_handler(stream_url))
            
        except Exception as e:
            self.logger.error(f"Error starting Binance feeds: {e}")
    
    async def _binance_websocket_handler(self, url: str):
        """Handle Binance WebSocket connection."""
        while self.is_active:
            try:
                async with websockets.connect(url) as websocket:
                    self.websocket_connections["binance"] = websocket
                    self.logger.info("Connected to Binance WebSocket")
                    
                    async for message in websocket:
                        if not self.is_active:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_binance_message(data)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing Binance message: {e}")
                            
            except Exception as e:
                self.logger.error(f"Binance WebSocket error: {e}")
                if self.is_active:
                    await asyncio.sleep(5)  # Reconnect delay
    
    async def _process_binance_message(self, data: Dict[str, Any]):
        """Process incoming Binance WebSocket message."""
        try:
            if isinstance(data, dict) and 'stream' in data:
                stream = data['stream']
                payload = data['data']
                
                if '@ticker' in stream:
                    await self._process_ticker_data(payload, "binance")
                elif '@depth' in stream:
                    await self._process_orderbook_data(payload, "binance")
                elif '@trade' in stream:
                    await self._process_trade_data(payload, "binance")
                    
        except Exception as e:
            self.logger.error(f"Error processing Binance message: {e}")
    
    async def _process_ticker_data(self, data: Dict[str, Any], exchange: str):
        """Process ticker data."""
        try:
            symbol = data.get('s', '')
            if symbol not in self.symbols:
                return
            
            market_data = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(data.get('c', 0)),  # Last price
                volume=float(data.get('v', 0)),  # 24h volume
                bid=float(data.get('b', 0)),  # Best bid
                ask=float(data.get('a', 0))   # Best ask
            )
            
            # Store in data queue
            self.market_data[f"{exchange}_{symbol}"].append(market_data)
            
            # Update market summary
            await self._update_market_summary(symbol, data, exchange)
            
        except Exception as e:
            self.logger.error(f"Error processing ticker data: {e}")
    
    async def _process_orderbook_data(self, data: Dict[str, Any], exchange: str):
        """Process order book data."""
        try:
            symbol = data.get('s', '')
            if symbol not in self.symbols:
                return
            
            # Parse bids and asks
            bids = [OrderBookLevel(float(level[0]), float(level[1])) 
                   for level in data.get('bids', [])]
            asks = [OrderBookLevel(float(level[0]), float(level[1])) 
                   for level in data.get('asks', [])]
            
            if not bids or not asks:
                return
            
            # Calculate mid price and spread
            best_bid = bids[0].price
            best_ask = asks[0].price
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Calculate total volumes
            total_bid_volume = sum(level.size * level.price for level in bids)
            total_ask_volume = sum(level.size * level.price for level in asks)
            
            order_book = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread=spread,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume
            )
            
            self.order_books[f"{exchange}_{symbol}"] = order_book
            
        except Exception as e:
            self.logger.error(f"Error processing order book data: {e}")
    
    async def _process_trade_data(self, data: Dict[str, Any], exchange: str):
        """Process trade data."""
        try:
            symbol = data.get('s', '')
            if symbol not in self.symbols:
                return
            
            # Create market data point from trade
            market_data = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000),
                price=float(data.get('p', 0)),
                volume=float(data.get('q', 0))
            )
            
            # Store trade data
            trade_key = f"{exchange}_{symbol}_trades"
            self.market_data[trade_key].append(market_data)
            
        except Exception as e:
            self.logger.error(f"Error processing trade data: {e}")
    
    async def _update_market_summary(self, symbol: str, ticker_data: Dict[str, Any], exchange: str):
        """Update market summary statistics."""
        try:
            last_price = float(ticker_data.get('c', 0))
            volume_24h = float(ticker_data.get('v', 0))
            price_change_24h = float(ticker_data.get('P', 0))
            high_24h = float(ticker_data.get('h', 0))
            low_24h = float(ticker_data.get('l', 0))
            
            # Calculate volatility estimate
            volatility = self._calculate_volatility_estimate(symbol, exchange)
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(symbol, exchange)
            
            summary = MarketSummary(
                symbol=symbol,
                last_price=last_price,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                price_change_pct_24h=price_change_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                volatility_estimate=volatility,
                liquidity_score=liquidity_score
            )
            
            self.market_summaries[f"{exchange}_{symbol}"] = summary
            
        except Exception as e:
            self.logger.error(f"Error updating market summary: {e}")
    
    def _calculate_volatility_estimate(self, symbol: str, exchange: str) -> float:
        """Calculate volatility estimate from recent price data."""
        try:
            data_key = f"{exchange}_{symbol}"
            if data_key not in self.market_data or len(self.market_data[data_key]) < 2:
                return 0.0
            
            # Get recent prices
            recent_data = list(self.market_data[data_key])[-100:]  # Last 100 points
            prices = [point.price for point in recent_data]
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            if not returns:
                return 0.0
            
            # Calculate volatility (annualized)
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(365 * 24)  # Assuming hourly data
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility estimate: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, symbol: str, exchange: str) -> float:
        """Calculate liquidity score from order book data."""
        try:
            orderbook_key = f"{exchange}_{symbol}"
            if orderbook_key not in self.order_books:
                return 0.5  # Default medium liquidity
            
            order_book = self.order_books[orderbook_key]
            
            # Simple liquidity score based on spread and volume
            if order_book.mid_price <= 0 or order_book.spread <= 0:
                return 0.5
            
            # Spread component (lower spread = higher liquidity)
            spread_pct = order_book.spread / order_book.mid_price
            spread_score = max(0, 1 - spread_pct * 1000)  # Normalize spread
            
            # Volume component (higher volume = higher liquidity)
            total_volume = order_book.total_bid_volume + order_book.total_ask_volume
            volume_score = min(1, total_volume / 1000000)  # Normalize to $1M
            
            # Combined score
            liquidity_score = (spread_score * 0.6 + volume_score * 0.4)
            return np.clip(liquidity_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    async def _price_data_processor(self):
        """Process and clean price data."""
        while self.is_active:
            try:
                # Clean old data
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for data_key, data_queue in self.market_data.items():
                    # Remove old data points
                    while data_queue and data_queue[0].timestamp < cutoff_time:
                        data_queue.popleft()
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in price data processor: {e}")
                await asyncio.sleep(60)
    
    async def _order_book_processor(self):
        """Process and analyze order book data."""
        while self.is_active:
            try:
                # Update order book analytics
                for book_key, order_book in self.order_books.items():
                    # Check if order book is stale
                    if datetime.now() - order_book.timestamp > timedelta(seconds=30):
                        self.logger.warning(f"Stale order book data: {book_key}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order book processor: {e}")
                await asyncio.sleep(30)
    
    async def _funding_rate_processor(self):
        """Fetch and process funding rate data."""
        while self.is_active:
            try:
                # Fetch funding rates from exchanges
                await self._fetch_binance_funding_rates()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in funding rate processor: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_binance_funding_rates(self):
        """Fetch funding rates from Binance."""
        try:
            if not self.session:
                return
            
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data:
                        symbol = item.get('symbol', '')
                        if symbol in [s.replace('USDT', 'USDT') for s in self.symbols]:
                            funding_rate = float(item.get('lastFundingRate', 0))
                            self.funding_rates['binance'][symbol] = funding_rate
                            
        except Exception as e:
            self.logger.error(f"Error fetching Binance funding rates: {e}")
    
    async def _market_summary_processor(self):
        """Process and update market summaries."""
        while self.is_active:
            try:
                # Update market summaries with additional calculations
                for summary_key, summary in self.market_summaries.items():
                    # Add any additional processing here
                    pass
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in market summary processor: {e}")
                await asyncio.sleep(60)
    
    async def get_current_data(self) -> Dict[str, Any]:
        """Get current market data snapshot."""
        try:
            current_data = {
                "timestamp": datetime.now().isoformat(),
                "prices": {},
                "order_books": {},
                "summaries": {},
                "funding_rates": dict(self.funding_rates)
            }
            
            # Get latest prices
            for data_key, data_queue in self.market_data.items():
                if data_queue and '_trades' not in data_key:
                    latest_point = data_queue[-1]
                    current_data["prices"][data_key] = {
                        "symbol": latest_point.symbol,
                        "price": latest_point.price,
                        "volume": latest_point.volume,
                        "timestamp": latest_point.timestamp.isoformat()
                    }
            
            # Get order books
            for book_key, order_book in self.order_books.items():
                current_data["order_books"][book_key] = {
                    "symbol": order_book.symbol,
                    "mid_price": order_book.mid_price,
                    "spread": order_book.spread,
                    "best_bid": order_book.bids[0].price if order_book.bids else 0,
                    "best_ask": order_book.asks[0].price if order_book.asks else 0,
                    "total_bid_volume": order_book.total_bid_volume,
                    "total_ask_volume": order_book.total_ask_volume,
                    "timestamp": order_book.timestamp.isoformat()
                }
            
            # Get market summaries
            for summary_key, summary in self.market_summaries.items():
                current_data["summaries"][summary_key] = {
                    "symbol": summary.symbol,
                    "last_price": summary.last_price,
                    "volume_24h": summary.volume_24h,
                    "price_change_pct_24h": summary.price_change_pct_24h,
                    "high_24h": summary.high_24h,
                    "low_24h": summary.low_24h,
                    "volatility_estimate": summary.volatility_estimate,
                    "liquidity_score": summary.liquidity_score
                }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error getting current data: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, exchange: str = "binance", 
                                periods: int = 100) -> List[MarketDataPoint]:
        """Get historical market data."""
        try:
            data_key = f"{exchange}_{symbol}"
            if data_key not in self.market_data:
                return []
            
            data_queue = self.market_data[data_key]
            return list(data_queue)[-periods:] if data_queue else []
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return []
    
    async def get_order_book(self, symbol: str, exchange: str = "binance") -> Optional[OrderBookSnapshot]:
        """Get current order book."""
        try:
            book_key = f"{exchange}_{symbol}"
            return self.order_books.get(book_key)
            
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            return None
    
    async def get_funding_rate(self, symbol: str, exchange: str = "binance") -> float:
        """Get current funding rate."""
        try:
            return self.funding_rates.get(exchange, {}).get(symbol, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting funding rate: {e}")
            return 0.0
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get data aggregator status."""
        return {
            "is_active": self.is_active,
            "total_data_points": sum(len(queue) for queue in self.market_data.values()),
            "order_books_count": len(self.order_books),
            "summaries_count": len(self.market_summaries),
            "websocket_connections": len(self.websocket_connections),
            "last_update_times": {
                key: time.isoformat() for key, time in self.last_update_times.items()
            },
            "symbols": self.symbols,
            "exchanges": self.exchanges
        }