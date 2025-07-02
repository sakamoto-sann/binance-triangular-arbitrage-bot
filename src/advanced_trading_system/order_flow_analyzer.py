"""
Order Flow Analysis and Smart Execution System
Analyzes market microstructure for optimal order placement and execution
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics
from scipy import stats
import json

@dataclass
class OrderBookLevel:
    """Data class for order book level"""
    price: float
    quantity: float
    orders: int

@dataclass
class OrderBookSnapshot:
    """Data class for order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    mid_price: float

@dataclass
class TradeData:
    """Data class for individual trade"""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    is_buyer_maker: bool

@dataclass
class OrderFlowMetrics:
    """Data class for order flow analysis metrics"""
    bid_ask_imbalance: float
    volume_imbalance: float
    trade_flow_imbalance: float
    order_book_pressure: float
    liquidity_score: float
    market_impact_estimate: float
    execution_difficulty: float
    directional_bias: str  # 'bullish', 'bearish', 'neutral'
    confidence: float

@dataclass
class ExecutionStrategy:
    """Data class for execution strategy recommendation"""
    strategy_type: str  # 'market', 'limit', 'iceberg', 'twap', 'vwap'
    recommended_price: float
    recommended_quantity: float
    time_horizon: int  # seconds
    expected_slippage: float
    market_impact: float
    execution_cost: float
    confidence: float
    reasoning: str

class OrderFlowAnalyzer:
    """
    Professional order flow analyzer that examines market microstructure
    for optimal order placement and execution strategies.
    """
    
    def __init__(self, binance_client=None, config: Dict = None):
        """Initialize the order flow analyzer"""
        self.binance_client = binance_client
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Analysis configuration
        self.order_book_depth = self.config.get('order_book_depth', 20)  # levels
        self.trade_flow_window = self.config.get('trade_flow_window', 300)  # seconds
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.7)  # 70% imbalance
        
        # Data storage
        self.order_book_history = {}  # symbol -> deque of OrderBookSnapshot
        self.trade_history = {}       # symbol -> deque of TradeData
        self.flow_metrics_history = {}  # symbol -> deque of OrderFlowMetrics
        
        # History limits
        self.max_order_book_history = 100
        self.max_trade_history = 1000
        self.max_metrics_history = 50
        
        # Market impact models
        self.impact_coefficients = {
            'BTCUSDT': 0.0001,   # Lower impact for highly liquid assets
            'ETHUSDT': 0.0002,
            'BNBUSDT': 0.0005,
            'ADAUSDT': 0.001,
            'SOLUSDT': 0.0008
        }
        
        # Execution strategy parameters
        self.strategy_parameters = {
            'market': {'max_quantity': 10000, 'urgency': 'high'},
            'limit': {'max_quantity': 50000, 'urgency': 'low'},
            'iceberg': {'max_quantity': 100000, 'urgency': 'medium'},
            'twap': {'max_quantity': 200000, 'urgency': 'low'},
            'vwap': {'max_quantity': 500000, 'urgency': 'medium'}
        }

    async def analyze_order_book_imbalance(self, symbol: str) -> OrderFlowMetrics:
        """
        Analyze bid/ask imbalances for directional bias.
        Returns comprehensive order flow metrics.
        """
        try:
            # Get current order book
            order_book = await self._get_order_book_snapshot(symbol)
            if not order_book:
                return self._get_default_metrics()
            
            # Store order book history
            self._store_order_book_snapshot(symbol, order_book)
            
            # Calculate bid-ask imbalance
            bid_ask_imbalance = self._calculate_bid_ask_imbalance(order_book)
            
            # Calculate volume imbalance
            volume_imbalance = self._calculate_volume_imbalance(order_book)
            
            # Get recent trades for trade flow analysis
            recent_trades = await self._get_recent_trades(symbol)
            trade_flow_imbalance = self._calculate_trade_flow_imbalance(recent_trades)
            
            # Calculate order book pressure
            order_book_pressure = self._calculate_order_book_pressure(order_book)
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(order_book)
            
            # Estimate market impact
            market_impact_estimate = self._estimate_market_impact(symbol, order_book)
            
            # Calculate execution difficulty
            execution_difficulty = self._calculate_execution_difficulty(
                order_book, trade_flow_imbalance, liquidity_score
            )
            
            # Determine directional bias
            directional_bias = self._determine_directional_bias(
                bid_ask_imbalance, volume_imbalance, trade_flow_imbalance
            )
            
            # Calculate confidence score
            confidence = self._calculate_metrics_confidence(
                order_book, recent_trades, bid_ask_imbalance
            )
            
            metrics = OrderFlowMetrics(
                bid_ask_imbalance=bid_ask_imbalance,
                volume_imbalance=volume_imbalance,
                trade_flow_imbalance=trade_flow_imbalance,
                order_book_pressure=order_book_pressure,
                liquidity_score=liquidity_score,
                market_impact_estimate=market_impact_estimate,
                execution_difficulty=execution_difficulty,
                directional_bias=directional_bias,
                confidence=confidence
            )
            
            # Store metrics history
            self._store_flow_metrics(symbol, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Order book imbalance analysis failed: {e}")
            return self._get_default_metrics()

    async def _get_order_book_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot"""
        try:
            if not self.binance_client:
                return None
            
            # Get order book from Binance
            order_book_data = self.binance_client.get_order_book(
                symbol=symbol, 
                limit=self.order_book_depth
            )
            
            # Parse bids and asks
            bids = []
            asks = []
            
            for bid in order_book_data['bids']:
                bids.append(OrderBookLevel(
                    price=float(bid[0]),
                    quantity=float(bid[1]),
                    orders=1  # Binance doesn't provide order count
                ))
            
            for ask in order_book_data['asks']:
                asks.append(OrderBookLevel(
                    price=float(ask[0]),
                    quantity=float(ask[1]),
                    orders=1
                ))
            
            if not bids or not asks:
                return None
            
            # Calculate derived metrics
            best_bid = bids[0].price
            best_ask = asks[0].price
            spread = best_ask - best_bid
            spread_pct = spread / best_bid if best_bid > 0 else 0
            mid_price = (best_bid + best_ask) / 2
            
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                spread_pct=spread_pct,
                mid_price=mid_price
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get order book snapshot: {e}")
            return None

    def _calculate_bid_ask_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate bid-ask imbalance at different depths"""
        try:
            # Calculate imbalance at multiple levels
            imbalances = []
            
            # Level 1 (best bid/ask)
            if len(order_book.bids) > 0 and len(order_book.asks) > 0:
                bid_qty = order_book.bids[0].quantity
                ask_qty = order_book.asks[0].quantity
                level1_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0
                imbalances.append(level1_imbalance)
            
            # Top 5 levels
            if len(order_book.bids) >= 5 and len(order_book.asks) >= 5:
                bid_qty_5 = sum(level.quantity for level in order_book.bids[:5])
                ask_qty_5 = sum(level.quantity for level in order_book.asks[:5])
                level5_imbalance = (bid_qty_5 - ask_qty_5) / (bid_qty_5 + ask_qty_5) if (bid_qty_5 + ask_qty_5) > 0 else 0
                imbalances.append(level5_imbalance)
            
            # Top 10 levels
            if len(order_book.bids) >= 10 and len(order_book.asks) >= 10:
                bid_qty_10 = sum(level.quantity for level in order_book.bids[:10])
                ask_qty_10 = sum(level.quantity for level in order_book.asks[:10])
                level10_imbalance = (bid_qty_10 - ask_qty_10) / (bid_qty_10 + ask_qty_10) if (bid_qty_10 + ask_qty_10) > 0 else 0
                imbalances.append(level10_imbalance)
            
            # Return weighted average imbalance
            if imbalances:
                weights = [0.5, 0.3, 0.2]  # Weight recent levels more heavily
                weighted_imbalance = sum(imp * weight for imp, weight in zip(imbalances, weights[:len(imbalances)]))
                return weighted_imbalance / sum(weights[:len(imbalances)])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Bid-ask imbalance calculation failed: {e}")
            return 0.0

    def _calculate_volume_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate volume imbalance based on order book depth"""
        try:
            # Calculate volume at different price levels
            total_bid_volume = sum(level.quantity for level in order_book.bids)
            total_ask_volume = sum(level.quantity for level in order_book.asks)
            
            if total_bid_volume + total_ask_volume == 0:
                return 0.0
            
            # Volume imbalance ratio
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Apply sigmoid transformation to normalize extreme values
            normalized_imbalance = np.tanh(volume_imbalance * 2)  # Scale and apply tanh
            
            return normalized_imbalance
            
        except Exception as e:
            self.logger.error(f"Volume imbalance calculation failed: {e}")
            return 0.0

    async def _get_recent_trades(self, symbol: str) -> List[TradeData]:
        """Get recent trades for flow analysis"""
        try:
            if not self.binance_client:
                return []
            
            # Get recent trades
            trades_data = self.binance_client.get_recent_trades(symbol=symbol, limit=100)
            
            trades = []
            cutoff_time = datetime.now() - timedelta(seconds=self.trade_flow_window)
            
            for trade in trades_data:
                trade_time = datetime.fromtimestamp(int(trade['time']) / 1000)
                
                if trade_time >= cutoff_time:
                    trades.append(TradeData(
                        symbol=symbol,
                        timestamp=trade_time,
                        price=float(trade['price']),
                        quantity=float(trade['qty']),
                        side='buy' if trade['isBuyerMaker'] else 'sell',
                        is_buyer_maker=trade['isBuyerMaker']
                    ))
            
            # Store trade history
            self._store_trade_data(symbol, trades)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to get recent trades: {e}")
            return []

    def _calculate_trade_flow_imbalance(self, trades: List[TradeData]) -> float:
        """Calculate trade flow imbalance (buy vs sell pressure)"""
        try:
            if not trades:
                return 0.0
            
            # Separate buy and sell trades
            buy_volume = sum(trade.quantity for trade in trades if not trade.is_buyer_maker)  # Taker buys
            sell_volume = sum(trade.quantity for trade in trades if trade.is_buyer_maker)     # Taker sells
            
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
            
            # Flow imbalance: positive means more buying pressure
            flow_imbalance = (buy_volume - sell_volume) / total_volume
            
            # Apply exponential weighting for recent trades
            if len(trades) > 10:
                recent_trades = trades[-10:]  # Last 10 trades
                recent_buy_volume = sum(trade.quantity for trade in recent_trades if not trade.is_buyer_maker)
                recent_sell_volume = sum(trade.quantity for trade in recent_trades if trade.is_buyer_maker)
                recent_total = recent_buy_volume + recent_sell_volume
                
                if recent_total > 0:
                    recent_flow_imbalance = (recent_buy_volume - recent_sell_volume) / recent_total
                    # Weight recent flow more heavily
                    flow_imbalance = 0.7 * recent_flow_imbalance + 0.3 * flow_imbalance
            
            return flow_imbalance
            
        except Exception as e:
            self.logger.error(f"Trade flow imbalance calculation failed: {e}")
            return 0.0

    def _calculate_order_book_pressure(self, order_book: OrderBookSnapshot) -> float:
        """Calculate order book pressure based on large orders and walls"""
        try:
            pressure_factors = []
            
            # Detect large orders (walls)
            bid_quantities = [level.quantity for level in order_book.bids]
            ask_quantities = [level.quantity for level in order_book.asks]
            
            if bid_quantities and ask_quantities:
                # Calculate order size statistics
                all_quantities = bid_quantities + ask_quantities
                mean_quantity = np.mean(all_quantities)
                std_quantity = np.std(all_quantities)
                
                # Identify large orders (>2 standard deviations from mean)
                large_order_threshold = mean_quantity + 2 * std_quantity
                
                # Count large bid orders
                large_bid_orders = sum(1 for qty in bid_quantities if qty > large_order_threshold)
                large_ask_orders = sum(1 for qty in ask_quantities if qty > large_order_threshold)
                
                # Pressure from large orders
                if large_bid_orders + large_ask_orders > 0:
                    large_order_pressure = (large_bid_orders - large_ask_orders) / (large_bid_orders + large_ask_orders)
                    pressure_factors.append(large_order_pressure)
            
            # Price level concentration
            if len(order_book.bids) >= 5 and len(order_book.asks) >= 5:
                # Top 3 levels vs next 7 levels concentration
                top3_bid_volume = sum(level.quantity for level in order_book.bids[:3])
                next7_bid_volume = sum(level.quantity for level in order_book.bids[3:10])
                
                top3_ask_volume = sum(level.quantity for level in order_book.asks[:3])
                next7_ask_volume = sum(level.quantity for level in order_book.asks[3:10])
                
                if next7_bid_volume > 0 and next7_ask_volume > 0:
                    bid_concentration = top3_bid_volume / next7_bid_volume
                    ask_concentration = top3_ask_volume / next7_ask_volume
                    
                    concentration_pressure = (bid_concentration - ask_concentration) / (bid_concentration + ask_concentration)
                    pressure_factors.append(concentration_pressure * 0.5)  # Lower weight
            
            # Return average pressure
            return np.mean(pressure_factors) if pressure_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Order book pressure calculation failed: {e}")
            return 0.0

    def _calculate_liquidity_score(self, order_book: OrderBookSnapshot) -> float:
        """Calculate overall liquidity score"""
        try:
            liquidity_factors = []
            
            # Spread-based liquidity
            spread_liquidity = max(0.0, 1.0 - (order_book.spread_pct * 1000))  # Lower spread = higher liquidity
            liquidity_factors.append(spread_liquidity)
            
            # Depth-based liquidity
            if order_book.bids and order_book.asks:
                total_volume = sum(level.quantity for level in order_book.bids + order_book.asks)
                # Normalize by a reference volume (would be symbol-specific in production)
                depth_liquidity = min(1.0, total_volume / 10000)  # Simplified normalization
                liquidity_factors.append(depth_liquidity)
            
            # Level count liquidity
            level_count = len(order_book.bids) + len(order_book.asks)
            level_liquidity = min(1.0, level_count / 40)  # Max 40 levels
            liquidity_factors.append(level_liquidity)
            
            # Price distribution liquidity
            if len(order_book.bids) >= 10 and len(order_book.asks) >= 10:
                bid_price_range = order_book.bids[0].price - order_book.bids[9].price
                ask_price_range = order_book.asks[9].price - order_book.asks[0].price
                
                # Wider price range with reasonable volume = better liquidity
                avg_price_range = (bid_price_range + ask_price_range) / 2
                range_liquidity = min(1.0, (avg_price_range / order_book.mid_price) * 100)
                liquidity_factors.append(range_liquidity)
            
            return np.mean(liquidity_factors) if liquidity_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Liquidity score calculation failed: {e}")
            return 0.5

    def _estimate_market_impact(self, symbol: str, order_book: OrderBookSnapshot) -> float:
        """Estimate market impact for typical order sizes"""
        try:
            # Get impact coefficient for the symbol
            impact_coeff = self.impact_coefficients.get(symbol, 0.001)
            
            # Calculate impact for different order sizes
            reference_order_size = 10000  # $10k reference order
            
            # Simple linear impact model: Impact = coefficient * sqrt(order_size / average_volume)
            if order_book.bids and order_book.asks:
                # Use top 5 levels average volume as reference
                avg_volume = np.mean([
                    level.quantity for level in (order_book.bids[:5] + order_book.asks[:5])
                ])
                
                if avg_volume > 0:
                    # Convert order size to quantity
                    order_quantity = reference_order_size / order_book.mid_price
                    
                    # Market impact estimate (as percentage)
                    impact = impact_coeff * np.sqrt(order_quantity / avg_volume)
                    
                    return min(0.05, impact)  # Cap at 5%
            
            return impact_coeff  # Default impact
            
        except Exception as e:
            self.logger.error(f"Market impact estimation failed: {e}")
            return 0.001

    def _calculate_execution_difficulty(self, order_book: OrderBookSnapshot, 
                                      trade_flow_imbalance: float, liquidity_score: float) -> float:
        """Calculate overall execution difficulty score"""
        try:
            difficulty_factors = []
            
            # Spread difficulty
            spread_difficulty = min(1.0, order_book.spread_pct * 2000)  # Higher spread = more difficult
            difficulty_factors.append(spread_difficulty)
            
            # Liquidity difficulty
            liquidity_difficulty = 1.0 - liquidity_score
            difficulty_factors.append(liquidity_difficulty)
            
            # Flow imbalance difficulty
            flow_difficulty = abs(trade_flow_imbalance) * 0.5  # Strong imbalance makes execution harder
            difficulty_factors.append(flow_difficulty)
            
            # Volatility difficulty (approximated from recent price range)
            if len(order_book.bids) >= 10 and len(order_book.asks) >= 10:
                price_range = order_book.asks[9].price - order_book.bids[9].price
                volatility_difficulty = min(1.0, (price_range / order_book.mid_price) * 50)
                difficulty_factors.append(volatility_difficulty)
            
            return np.mean(difficulty_factors) if difficulty_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Execution difficulty calculation failed: {e}")
            return 0.5

    def _determine_directional_bias(self, bid_ask_imbalance: float, 
                                  volume_imbalance: float, trade_flow_imbalance: float) -> str:
        """Determine overall directional bias"""
        try:
            # Combine different imbalance measures
            imbalances = [bid_ask_imbalance, volume_imbalance, trade_flow_imbalance]
            avg_imbalance = np.mean(imbalances)
            
            # Thresholds for bias determination
            strong_threshold = 0.3
            weak_threshold = 0.1
            
            if avg_imbalance > strong_threshold:
                return 'strongly_bullish'
            elif avg_imbalance > weak_threshold:
                return 'bullish'
            elif avg_imbalance < -strong_threshold:
                return 'strongly_bearish'
            elif avg_imbalance < -weak_threshold:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'

    def _calculate_metrics_confidence(self, order_book: OrderBookSnapshot, 
                                    trades: List[TradeData], bid_ask_imbalance: float) -> float:
        """Calculate confidence in the metrics"""
        try:
            confidence_factors = []
            
            # Data quality confidence
            if len(order_book.bids) >= 10 and len(order_book.asks) >= 10:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.6)
            
            # Trade data confidence
            if len(trades) >= 20:
                confidence_factors.append(1.0)
            elif len(trades) >= 10:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Spread confidence (tighter spreads = more confident)
            spread_confidence = max(0.3, 1.0 - (order_book.spread_pct * 1000))
            confidence_factors.append(spread_confidence)
            
            # Imbalance magnitude confidence
            imbalance_confidence = min(1.0, abs(bid_ask_imbalance) * 3)  # Stronger signals = more confident
            confidence_factors.append(imbalance_confidence)
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5

    def optimize_order_placement(self, symbol: str, side: str, quantity: float, 
                               market_conditions: Dict[str, Any]) -> ExecutionStrategy:
        """
        Optimize order placement based on market microstructure analysis.
        Returns the best execution strategy for the given order.
        """
        try:
            # Get current order flow metrics
            flow_metrics = market_conditions.get('flow_metrics')
            if not flow_metrics:
                # Get fresh metrics if not provided
                asyncio.create_task(self.analyze_order_book_imbalance(symbol))
                flow_metrics = self._get_default_metrics()
            
            # Get current order book
            order_book = market_conditions.get('order_book')
            if not order_book:
                order_book = asyncio.create_task(self._get_order_book_snapshot(symbol))
            
            # Analyze order characteristics
            order_size_usd = quantity * (order_book.mid_price if order_book else 50000)  # Estimate
            urgency = market_conditions.get('urgency', 'medium')
            
            # Determine optimal strategy
            strategy_type = self._select_execution_strategy(
                order_size_usd, urgency, flow_metrics, order_book, side
            )
            
            # Calculate optimal parameters for the selected strategy
            strategy_params = self._calculate_strategy_parameters(
                strategy_type, symbol, side, quantity, flow_metrics, order_book
            )
            
            return strategy_params
            
        except Exception as e:
            self.logger.error(f"Order placement optimization failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _select_execution_strategy(self, order_size_usd: float, urgency: str, 
                                 flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot, side: str = 'buy') -> str:
        """Select the optimal execution strategy"""
        try:
            # Strategy selection logic
            
            # Small orders with high urgency -> Market order
            if order_size_usd < 5000 and urgency == 'high':
                return 'market'
            
            # Large orders with low urgency -> TWAP or VWAP
            if order_size_usd > 100000 and urgency == 'low':
                if flow_metrics.liquidity_score > 0.7:
                    return 'vwap'
                else:
                    return 'twap'
            
            # High execution difficulty -> Iceberg or TWAP
            if flow_metrics.execution_difficulty > 0.7:
                if order_size_usd > 50000:
                    return 'iceberg'
                else:
                    return 'twap'
            
            # Strong directional bias against our order -> Careful execution
            if ((side == 'buy' and 'bearish' in flow_metrics.directional_bias) or
                (side == 'sell' and 'bullish' in flow_metrics.directional_bias)):
                return 'iceberg' if order_size_usd > 20000 else 'limit'
            
            # Default to limit orders for medium-sized orders
            if order_size_usd < 50000:
                return 'limit'
            else:
                return 'iceberg'
                
        except Exception:
            return 'limit'  # Safe default

    def _calculate_strategy_parameters(self, strategy_type: str, symbol: str, side: str, 
                                     quantity: float, flow_metrics: OrderFlowMetrics, 
                                     order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate specific parameters for the execution strategy"""
        try:
            if strategy_type == 'market':
                return self._calculate_market_strategy(symbol, side, quantity, flow_metrics, order_book)
            elif strategy_type == 'limit':
                return self._calculate_limit_strategy(symbol, side, quantity, flow_metrics, order_book)
            elif strategy_type == 'iceberg':
                return self._calculate_iceberg_strategy(symbol, side, quantity, flow_metrics, order_book)
            elif strategy_type == 'twap':
                return self._calculate_twap_strategy(symbol, side, quantity, flow_metrics, order_book)
            elif strategy_type == 'vwap':
                return self._calculate_vwap_strategy(symbol, side, quantity, flow_metrics, order_book)
            else:
                return self._get_default_execution_strategy(symbol, side, quantity)
                
        except Exception as e:
            self.logger.error(f"Strategy parameter calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _calculate_market_strategy(self, symbol: str, side: str, quantity: float,
                                 flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate market order strategy parameters"""
        try:
            # Market orders execute immediately at current market price
            if side == 'buy':
                recommended_price = order_book.best_ask if order_book else 0
            else:
                recommended_price = order_book.best_bid if order_book else 0
            
            # Estimate slippage based on order book depth
            expected_slippage = self._estimate_slippage(side, quantity, order_book, 'market')
            
            # Market impact is immediate
            market_impact = flow_metrics.market_impact_estimate * quantity
            
            # Execution cost includes spread and slippage
            execution_cost = (order_book.spread if order_book else 0) + expected_slippage
            
            return ExecutionStrategy(
                strategy_type='market',
                recommended_price=recommended_price,
                recommended_quantity=quantity,
                time_horizon=0,  # Immediate execution
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                confidence=0.9,  # High confidence for immediate execution
                reasoning="Fast execution needed, willing to pay spread"
            )
            
        except Exception as e:
            self.logger.error(f"Market strategy calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _calculate_limit_strategy(self, symbol: str, side: str, quantity: float,
                                flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate limit order strategy parameters"""
        try:
            # Limit orders try to get better pricing
            if side == 'buy':
                # Try to buy below mid-price
                aggressive_factor = min(0.5, flow_metrics.execution_difficulty)
                price_improvement = order_book.spread * (0.5 - aggressive_factor) if order_book else 0
                recommended_price = order_book.mid_price - price_improvement if order_book else 0
            else:
                # Try to sell above mid-price
                aggressive_factor = min(0.5, flow_metrics.execution_difficulty)
                price_improvement = order_book.spread * (0.5 - aggressive_factor) if order_book else 0
                recommended_price = order_book.mid_price + price_improvement if order_book else 0
            
            # Lower slippage but uncertain execution
            expected_slippage = 0.0  # Limit orders control slippage
            
            # Market impact depends on whether order gets filled
            market_impact = flow_metrics.market_impact_estimate * quantity * 0.5
            
            # Execution cost is mainly opportunity cost
            execution_cost = order_book.spread * 0.3 if order_book else 0  # Partial spread savings
            
            # Time horizon depends on market conditions
            time_horizon = int(300 * (1 + flow_metrics.execution_difficulty))  # 5-10 minutes
            
            # Confidence depends on how aggressive the pricing is
            confidence = 0.7 * (1 - flow_metrics.execution_difficulty)
            
            return ExecutionStrategy(
                strategy_type='limit',
                recommended_price=recommended_price,
                recommended_quantity=quantity,
                time_horizon=time_horizon,
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                confidence=confidence,
                reasoning="Better pricing with execution risk"
            )
            
        except Exception as e:
            self.logger.error(f"Limit strategy calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _calculate_iceberg_strategy(self, symbol: str, side: str, quantity: float,
                                  flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate iceberg order strategy parameters"""
        try:
            # Iceberg orders hide order size
            slice_size = min(quantity * 0.1, 1000)  # 10% slices, max 1000 units
            
            # Price similar to limit orders but slightly more aggressive
            if side == 'buy':
                aggressive_factor = min(0.7, flow_metrics.execution_difficulty * 1.2)
                price_improvement = order_book.spread * (0.5 - aggressive_factor * 0.5) if order_book else 0
                recommended_price = order_book.mid_price - price_improvement if order_book else 0
            else:
                aggressive_factor = min(0.7, flow_metrics.execution_difficulty * 1.2)
                price_improvement = order_book.spread * (0.5 - aggressive_factor * 0.5) if order_book else 0
                recommended_price = order_book.mid_price + price_improvement if order_book else 0
            
            # Reduced market impact due to hidden size
            market_impact = flow_metrics.market_impact_estimate * slice_size
            
            # Longer execution time
            time_horizon = int(600 * (quantity / slice_size))  # Time per slice
            
            # Moderate slippage
            expected_slippage = self._estimate_slippage(side, slice_size, order_book, 'iceberg')
            
            execution_cost = order_book.spread * 0.4 if order_book else 0
            
            confidence = 0.8 * flow_metrics.confidence
            
            return ExecutionStrategy(
                strategy_type='iceberg',
                recommended_price=recommended_price,
                recommended_quantity=slice_size,  # Show slice size
                time_horizon=time_horizon,
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                confidence=confidence,
                reasoning=f"Large order hidden in {int(quantity/slice_size)} slices"
            )
            
        except Exception as e:
            self.logger.error(f"Iceberg strategy calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _calculate_twap_strategy(self, symbol: str, side: str, quantity: float,
                               flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate TWAP (Time-Weighted Average Price) strategy parameters"""
        try:
            # TWAP spreads execution over time
            num_slices = min(20, max(5, int(quantity / 500)))  # 5-20 slices
            slice_size = quantity / num_slices
            
            # Price at mid-point for minimal market impact
            recommended_price = order_book.mid_price if order_book else 0
            
            # Minimal market impact per slice
            market_impact = flow_metrics.market_impact_estimate * slice_size
            
            # Long execution time
            slice_interval = 60  # 1 minute between slices
            time_horizon = num_slices * slice_interval
            
            # Very low slippage
            expected_slippage = self._estimate_slippage(side, slice_size, order_book, 'twap')
            
            execution_cost = order_book.spread * 0.5 if order_book else 0  # Average of bid-ask
            
            confidence = 0.85  # High confidence for time-distributed execution
            
            return ExecutionStrategy(
                strategy_type='twap',
                recommended_price=recommended_price,
                recommended_quantity=slice_size,
                time_horizon=time_horizon,
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                confidence=confidence,
                reasoning=f"Execute {num_slices} slices over {time_horizon//60} minutes"
            )
            
        except Exception as e:
            self.logger.error(f"TWAP strategy calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _calculate_vwap_strategy(self, symbol: str, side: str, quantity: float,
                               flow_metrics: OrderFlowMetrics, order_book: OrderBookSnapshot) -> ExecutionStrategy:
        """Calculate VWAP (Volume-Weighted Average Price) strategy parameters"""
        try:
            # VWAP follows market volume patterns
            # Simplified: assume higher volume during certain periods
            
            # Adaptive slice size based on expected volume
            base_slice_size = quantity * 0.05  # 5% base slice
            volume_adjustment = flow_metrics.liquidity_score  # More liquid = larger slices
            slice_size = base_slice_size * (1 + volume_adjustment)
            
            num_slices = int(quantity / slice_size)
            
            # Price slightly favors market (more aggressive than TWAP)
            if side == 'buy':
                recommended_price = order_book.mid_price + order_book.spread * 0.2 if order_book else 0
            else:
                recommended_price = order_book.mid_price - order_book.spread * 0.2 if order_book else 0
            
            # Market impact varies with volume
            market_impact = flow_metrics.market_impact_estimate * slice_size * 0.8
            
            # Time horizon based on volume patterns
            time_horizon = num_slices * 45  # 45 seconds average between slices
            
            expected_slippage = self._estimate_slippage(side, slice_size, order_book, 'vwap')
            
            execution_cost = order_book.spread * 0.6 if order_book else 0
            
            confidence = 0.8 * flow_metrics.liquidity_score
            
            return ExecutionStrategy(
                strategy_type='vwap',
                recommended_price=recommended_price,
                recommended_quantity=slice_size,
                time_horizon=time_horizon,
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                execution_cost=execution_cost,
                confidence=confidence,
                reasoning=f"Volume-weighted execution in {num_slices} adaptive slices"
            )
            
        except Exception as e:
            self.logger.error(f"VWAP strategy calculation failed: {e}")
            return self._get_default_execution_strategy(symbol, side, quantity)

    def _estimate_slippage(self, side: str, quantity: float, order_book: OrderBookSnapshot, 
                          strategy_type: str) -> float:
        """Estimate slippage for given order parameters"""
        try:
            if not order_book:
                return 0.001  # Default 0.1% slippage
            
            # Get relevant order book side
            levels = order_book.asks if side == 'buy' else order_book.bids
            
            if not levels:
                return 0.001
            
            # Calculate weighted average price for the quantity
            remaining_quantity = quantity
            total_cost = 0.0
            
            for level in levels:
                if remaining_quantity <= 0:
                    break
                
                level_quantity = min(remaining_quantity, level.quantity)
                total_cost += level_quantity * level.price
                remaining_quantity -= level_quantity
            
            if remaining_quantity > 0:
                # Not enough liquidity, high slippage
                return 0.01  # 1% slippage for insufficient liquidity
            
            # Calculate average execution price
            avg_execution_price = total_cost / quantity if quantity > 0 else levels[0].price
            
            # Calculate slippage vs best price
            best_price = levels[0].price
            slippage = abs(avg_execution_price - best_price) / best_price
            
            # Adjust for strategy type
            strategy_multipliers = {
                'market': 1.0,
                'limit': 0.0,     # No slippage for limit orders (but execution risk)
                'iceberg': 0.3,   # Reduced slippage due to smaller visible size
                'twap': 0.1,      # Minimal slippage due to small slices
                'vwap': 0.2       # Low slippage, volume-sensitive
            }
            
            multiplier = strategy_multipliers.get(strategy_type, 1.0)
            return slippage * multiplier
            
        except Exception as e:
            self.logger.error(f"Slippage estimation failed: {e}")
            return 0.001

    def _store_order_book_snapshot(self, symbol: str, snapshot: OrderBookSnapshot):
        """Store order book snapshot in history"""
        try:
            if symbol not in self.order_book_history:
                self.order_book_history[symbol] = deque(maxlen=self.max_order_book_history)
            
            self.order_book_history[symbol].append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Failed to store order book snapshot: {e}")

    def _store_trade_data(self, symbol: str, trades: List[TradeData]):
        """Store trade data in history"""
        try:
            if symbol not in self.trade_history:
                self.trade_history[symbol] = deque(maxlen=self.max_trade_history)
            
            for trade in trades:
                self.trade_history[symbol].append(trade)
            
        except Exception as e:
            self.logger.error(f"Failed to store trade data: {e}")

    def _store_flow_metrics(self, symbol: str, metrics: OrderFlowMetrics):
        """Store flow metrics in history"""
        try:
            if symbol not in self.flow_metrics_history:
                self.flow_metrics_history[symbol] = deque(maxlen=self.max_metrics_history)
            
            self.flow_metrics_history[symbol].append(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to store flow metrics: {e}")

    def _get_default_metrics(self) -> OrderFlowMetrics:
        """Get default flow metrics when calculation fails"""
        return OrderFlowMetrics(
            bid_ask_imbalance=0.0,
            volume_imbalance=0.0,
            trade_flow_imbalance=0.0,
            order_book_pressure=0.0,
            liquidity_score=0.5,
            market_impact_estimate=0.001,
            execution_difficulty=0.5,
            directional_bias='neutral',
            confidence=0.3
        )

    def _get_default_execution_strategy(self, symbol: str, side: str, quantity: float) -> ExecutionStrategy:
        """Get default execution strategy when optimization fails"""
        return ExecutionStrategy(
            strategy_type='limit',
            recommended_price=0.0,  # Would need current price
            recommended_quantity=quantity,
            time_horizon=300,  # 5 minutes
            expected_slippage=0.001,
            market_impact=0.001,
            execution_cost=0.001,
            confidence=0.5,
            reasoning="Default strategy due to analysis failure"
        )

    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive order flow analysis for a symbol"""
        try:
            # Get all analysis components
            flow_metrics = await self.analyze_order_book_imbalance(symbol)
            order_book = await self._get_order_book_snapshot(symbol)
            recent_trades = await self._get_recent_trades(symbol)
            
            # Get optimization recommendations for different order sizes
            small_order_strategy = self.optimize_order_placement(
                symbol, 'buy', 100, {'flow_metrics': flow_metrics, 'order_book': order_book, 'urgency': 'medium'}
            )
            large_order_strategy = self.optimize_order_placement(
                symbol, 'buy', 10000, {'flow_metrics': flow_metrics, 'order_book': order_book, 'urgency': 'low'}
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'order_flow_metrics': {
                    'bid_ask_imbalance': flow_metrics.bid_ask_imbalance,
                    'volume_imbalance': flow_metrics.volume_imbalance,
                    'trade_flow_imbalance': flow_metrics.trade_flow_imbalance,
                    'order_book_pressure': flow_metrics.order_book_pressure,
                    'liquidity_score': flow_metrics.liquidity_score,
                    'market_impact_estimate': flow_metrics.market_impact_estimate,
                    'execution_difficulty': flow_metrics.execution_difficulty,
                    'directional_bias': flow_metrics.directional_bias,
                    'confidence': flow_metrics.confidence
                },
                'order_book_summary': {
                    'best_bid': order_book.best_bid if order_book else 0,
                    'best_ask': order_book.best_ask if order_book else 0,
                    'spread': order_book.spread if order_book else 0,
                    'spread_pct': order_book.spread_pct if order_book else 0,
                    'mid_price': order_book.mid_price if order_book else 0,
                    'levels_count': len(order_book.bids) + len(order_book.asks) if order_book else 0
                },
                'trade_flow_summary': {
                    'recent_trades_count': len(recent_trades),
                    'trade_flow_window_minutes': self.trade_flow_window / 60,
                    'avg_trade_size': np.mean([t.quantity for t in recent_trades]) if recent_trades else 0
                },
                'execution_recommendations': {
                    'small_orders': {
                        'strategy': small_order_strategy.strategy_type,
                        'recommended_price': small_order_strategy.recommended_price,
                        'confidence': small_order_strategy.confidence,
                        'reasoning': small_order_strategy.reasoning
                    },
                    'large_orders': {
                        'strategy': large_order_strategy.strategy_type,
                        'recommended_price': large_order_strategy.recommended_price,
                        'time_horizon_minutes': large_order_strategy.time_horizon / 60,
                        'confidence': large_order_strategy.confidence,
                        'reasoning': large_order_strategy.reasoning
                    }
                },
                'market_conditions': {
                    'overall_assessment': self._assess_overall_market_conditions(flow_metrics),
                    'execution_environment': 'favorable' if flow_metrics.execution_difficulty < 0.4 else 'challenging' if flow_metrics.execution_difficulty > 0.7 else 'normal'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _assess_overall_market_conditions(self, flow_metrics: OrderFlowMetrics) -> str:
        """Assess overall market conditions for execution"""
        try:
            if flow_metrics.liquidity_score > 0.8 and flow_metrics.execution_difficulty < 0.3:
                return 'excellent'
            elif flow_metrics.liquidity_score > 0.6 and flow_metrics.execution_difficulty < 0.5:
                return 'good'
            elif flow_metrics.liquidity_score > 0.4 and flow_metrics.execution_difficulty < 0.7:
                return 'fair'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'