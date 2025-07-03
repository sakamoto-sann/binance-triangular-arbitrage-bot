"""
Advanced Trading System - Liquidity Flow Analyzer
Order book depth analysis and market impact estimation for optimal execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol."""
    symbol: str
    exchange: str
    
    # Order book metrics
    bid_ask_spread: float = 0.0
    spread_bps: float = 0.0
    effective_spread: float = 0.0
    
    # Depth metrics
    depth_1_pct: float = 0.0  # Liquidity within 1% of mid
    depth_5_pct: float = 0.0  # Liquidity within 5% of mid
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    total_depth: float = 0.0
    
    # Impact metrics
    market_impact_10k: float = 0.0   # Impact of $10k trade
    market_impact_100k: float = 0.0  # Impact of $100k trade
    market_impact_1m: float = 0.0    # Impact of $1M trade
    
    # Flow metrics
    price_impact_coeff: float = 0.0  # Kyle's lambda
    order_flow_imbalance: float = 0.0
    volume_imbalance: float = 0.0
    
    # Resilience metrics
    recovery_time_estimate: float = 0.0  # Minutes to recover from impact
    liquidity_score: float = 0.0  # Overall liquidity score (0-1)
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OrderFlowSignal:
    """Order flow and liquidity signal."""
    symbol: str
    signal_type: str  # "buy_pressure", "sell_pressure", "balanced", "stressed"
    strength: float  # Signal strength (0-1)
    confidence: float  # Signal confidence (0-1)
    
    # Supporting metrics
    order_flow_delta: float = 0.0
    volume_weighted_pressure: float = 0.0
    microstructure_noise: float = 0.0
    
    # Recommendations
    optimal_trade_size: float = 0.0
    recommended_execution_style: str = "twap"  # "aggressive", "passive", "twap", "vwap"
    estimated_impact: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketImpactModel:
    """Market impact model parameters."""
    symbol: str
    
    # Linear impact model: Impact = alpha * (Size / ADV)^beta
    alpha: float = 0.0  # Base impact coefficient
    beta: float = 0.5   # Size scaling exponent
    
    # Temporary vs permanent impact
    temporary_impact_ratio: float = 0.7  # 70% temporary, 30% permanent
    decay_half_life: float = 5.0  # Minutes for temporary impact to decay
    
    # Volume-based scaling
    adv_20d: float = 0.0  # 20-day average daily volume
    current_volume_ratio: float = 1.0  # Current vs normal volume
    
    # Volatility adjustment
    volatility_multiplier: float = 1.0
    
    last_calibrated: datetime = field(default_factory=datetime.now)

class LiquidityFlowAnalyzer:
    """
    Advanced liquidity flow analysis system.
    
    Analyzes order book depth, market impact, and flow patterns
    to optimize trade execution and detect market stress.
    """
    
    def __init__(self, market_data_aggregator):
        """
        Initialize liquidity flow analyzer.
        
        Args:
            market_data_aggregator: Market data aggregation system
        """
        self.market_data = market_data_aggregator
        
        # Data storage
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        self.order_flow_signals: Dict[str, OrderFlowSignal] = {}
        self.impact_models: Dict[str, MarketImpactModel] = {}
        self.flow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Order book tracking
        self.order_book_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.trade_flow: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Analysis parameters
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.exchanges = ["binance", "bybit", "okex"]
        self.depth_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1% to 5%
        
        # Processing tasks
        self.analysis_tasks: List[asyncio.Task] = []
        self.is_active = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("LiquidityFlowAnalyzer initialized")
    
    async def start(self) -> bool:
        """Start liquidity flow analysis."""
        try:
            self.logger.info("Starting liquidity flow analysis...")
            
            # Initialize impact models
            await self._initialize_impact_models()
            
            # Start analysis tasks
            self.analysis_tasks = [
                asyncio.create_task(self._liquidity_metrics_calculator()),
                asyncio.create_task(self._order_flow_analyzer()),
                asyncio.create_task(self._market_impact_calibrator()),
                asyncio.create_task(self._flow_signal_generator())
            ]
            
            self.is_active = True
            self.logger.info("Liquidity flow analysis started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting liquidity flow analysis: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop liquidity flow analysis."""
        try:
            self.logger.info("Stopping liquidity flow analysis...")
            
            self.is_active = False
            
            # Cancel analysis tasks
            for task in self.analysis_tasks:
                task.cancel()
            
            self.logger.info("Liquidity flow analysis stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping liquidity flow analysis: {e}")
            return False
    
    async def _initialize_impact_models(self):
        """Initialize market impact models for all symbols."""
        try:
            for symbol in self.symbols:
                # Initialize with reasonable defaults
                model = MarketImpactModel(
                    symbol=symbol,
                    alpha=0.001,  # 0.1% base impact
                    beta=0.5,     # Square root scaling
                    adv_20d=1000000.0,  # $1M default ADV
                    temporary_impact_ratio=0.7,
                    decay_half_life=5.0
                )
                
                self.impact_models[symbol] = model
                
        except Exception as e:
            self.logger.error(f"Error initializing impact models: {e}")
    
    async def _liquidity_metrics_calculator(self):
        """Calculate comprehensive liquidity metrics."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        # Get current order book
                        order_book = await self.market_data.get_order_book(symbol, exchange)
                        
                        if order_book and order_book.bids and order_book.asks:
                            metrics = await self._calculate_symbol_liquidity(symbol, exchange, order_book)
                            if metrics:
                                key = f"{exchange}_{symbol}"
                                self.liquidity_metrics[key] = metrics
                                self.flow_history[key].append(metrics)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in liquidity metrics calculator: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_symbol_liquidity(self, symbol: str, exchange: str, 
                                        order_book) -> Optional[LiquidityMetrics]:
        """Calculate liquidity metrics for a specific symbol."""
        try:
            if not order_book.bids or not order_book.asks:
                return None
            
            metrics = LiquidityMetrics(symbol=symbol, exchange=exchange)
            
            # Basic spread metrics
            best_bid = order_book.bids[0].price
            best_ask = order_book.asks[0].price
            mid_price = (best_bid + best_ask) / 2
            
            metrics.bid_ask_spread = best_ask - best_bid
            metrics.spread_bps = (metrics.bid_ask_spread / mid_price) * 10000
            
            # Effective spread (would need trade data for proper calculation)
            metrics.effective_spread = metrics.bid_ask_spread * 0.8  # Approximation
            
            # Depth analysis
            await self._calculate_depth_metrics(metrics, order_book, mid_price)
            
            # Market impact estimation
            await self._estimate_market_impact(metrics, symbol, order_book, mid_price)
            
            # Order flow metrics
            await self._calculate_flow_metrics(metrics, symbol, exchange)
            
            # Overall liquidity score
            metrics.liquidity_score = self._calculate_liquidity_score(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity for {symbol}/{exchange}: {e}")
            return None
    
    async def _calculate_depth_metrics(self, metrics: LiquidityMetrics, order_book, mid_price: float):
        """Calculate order book depth metrics."""
        try:
            # Calculate depth at different levels
            for depth_pct in self.depth_levels:
                bid_depth = 0.0
                ask_depth = 0.0
                
                # Calculate bid side depth
                price_threshold = mid_price * (1 - depth_pct)
                for level in order_book.bids:
                    if level.price >= price_threshold:
                        bid_depth += level.size * level.price
                    else:
                        break
                
                # Calculate ask side depth
                price_threshold = mid_price * (1 + depth_pct)
                for level in order_book.asks:
                    if level.price <= price_threshold:
                        ask_depth += level.size * level.price
                    else:
                        break
                
                # Store depth metrics
                if depth_pct == 0.01:  # 1% depth
                    metrics.depth_1_pct = bid_depth + ask_depth
                elif depth_pct == 0.05:  # 5% depth
                    metrics.depth_5_pct = bid_depth + ask_depth
            
            # Total visible depth
            metrics.bid_depth = sum(level.size * level.price for level in order_book.bids[:20])
            metrics.ask_depth = sum(level.size * level.price for level in order_book.asks[:20])
            metrics.total_depth = metrics.bid_depth + metrics.ask_depth
            
        except Exception as e:
            self.logger.error(f"Error calculating depth metrics: {e}")
    
    async def _estimate_market_impact(self, metrics: LiquidityMetrics, symbol: str, 
                                    order_book, mid_price: float):
        """Estimate market impact for different trade sizes."""
        try:
            if symbol not in self.impact_models:
                return
            
            model = self.impact_models[symbol]
            trade_sizes = [10000, 100000, 1000000]  # $10k, $100k, $1M
            
            for i, trade_size in enumerate(trade_sizes):
                # Simple order book walk for impact estimation
                impact = await self._calculate_order_book_impact(
                    order_book, trade_size, mid_price, "buy"
                )
                
                # Store in appropriate metric
                if i == 0:
                    metrics.market_impact_10k = impact
                elif i == 1:
                    metrics.market_impact_100k = impact
                elif i == 2:
                    metrics.market_impact_1m = impact
            
            # Calculate price impact coefficient (Kyle's lambda)
            if metrics.total_depth > 0:
                # Simplified lambda estimation
                metrics.price_impact_coeff = 100000 / metrics.total_depth  # Impact per $100k
            
        except Exception as e:
            self.logger.error(f"Error estimating market impact: {e}")
    
    async def _calculate_order_book_impact(self, order_book, trade_size: float, 
                                         mid_price: float, side: str) -> float:
        """Calculate impact by walking through order book."""
        try:
            remaining_size = trade_size
            total_cost = 0.0
            
            # Choose appropriate side
            levels = order_book.asks if side == "buy" else order_book.bids
            
            for level in levels:
                if remaining_size <= 0:
                    break
                
                level_notional = level.size * level.price
                consumed_notional = min(remaining_size, level_notional)
                
                total_cost += consumed_notional
                remaining_size -= consumed_notional
            
            if trade_size > 0 and total_cost > 0:
                avg_price = total_cost / (trade_size - remaining_size) if remaining_size < trade_size else level.price
                impact = abs(avg_price - mid_price) / mid_price
                return impact
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating order book impact: {e}")
            return 0.0
    
    async def _calculate_flow_metrics(self, metrics: LiquidityMetrics, symbol: str, exchange: str):
        """Calculate order flow and imbalance metrics."""
        try:
            key = f"{exchange}_{symbol}"
            
            # Order flow imbalance
            if len(self.flow_history[key]) > 1:
                recent_metrics = list(self.flow_history[key])[-10:]
                
                # Calculate volume imbalance
                bid_volumes = [m.bid_depth for m in recent_metrics]
                ask_volumes = [m.ask_depth for m in recent_metrics]
                
                if bid_volumes and ask_volumes:
                    avg_bid = np.mean(bid_volumes)
                    avg_ask = np.mean(ask_volumes)
                    
                    if avg_bid + avg_ask > 0:
                        metrics.volume_imbalance = (avg_bid - avg_ask) / (avg_bid + avg_ask)
                    
                    # Order flow delta
                    spread_changes = np.diff([m.bid_ask_spread for m in recent_metrics])
                    if len(spread_changes) > 0:
                        metrics.order_flow_imbalance = np.mean(spread_changes)
            
            # Recovery time estimate (based on spread and depth)
            if metrics.bid_ask_spread > 0 and metrics.total_depth > 0:
                # Higher depth and tighter spread = faster recovery
                recovery_factor = metrics.total_depth / (metrics.bid_ask_spread * 1000000)
                metrics.recovery_time_estimate = max(0.5, 10 / recovery_factor)  # Minutes
            
        except Exception as e:
            self.logger.error(f"Error calculating flow metrics: {e}")
    
    def _calculate_liquidity_score(self, metrics: LiquidityMetrics) -> float:
        """Calculate overall liquidity score (0-1, higher is better)."""
        try:
            score_components = []
            
            # Spread component (tighter spreads = higher score)
            if metrics.spread_bps > 0:
                spread_score = max(0, 1 - (metrics.spread_bps / 50))  # 50 bps = 0 score
                score_components.append(spread_score * 0.3)
            
            # Depth component
            if metrics.depth_1_pct > 0:
                depth_score = min(1, metrics.depth_1_pct / 100000)  # $100k = perfect score
                score_components.append(depth_score * 0.4)
            
            # Impact component (lower impact = higher score)
            if metrics.market_impact_100k > 0:
                impact_score = max(0, 1 - (metrics.market_impact_100k * 100))  # 1% impact = 0 score
                score_components.append(impact_score * 0.3)
            
            return sum(score_components) if score_components else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    async def _order_flow_analyzer(self):
        """Analyze order flow patterns and generate signals."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        key = f"{exchange}_{symbol}"
                        
                        if key in self.liquidity_metrics:
                            signal = await self._generate_flow_signal(key, symbol, exchange)
                            if signal:
                                self.order_flow_signals[key] = signal
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in order flow analyzer: {e}")
                await asyncio.sleep(60)
    
    async def _generate_flow_signal(self, key: str, symbol: str, exchange: str) -> Optional[OrderFlowSignal]:
        """Generate order flow signal for symbol."""
        try:
            if len(self.flow_history[key]) < 5:
                return None
            
            current_metrics = self.liquidity_metrics[key]
            recent_history = list(self.flow_history[key])[-10:]
            
            # Analyze flow patterns
            volume_imbalances = [m.volume_imbalance for m in recent_history]
            spread_changes = np.diff([m.bid_ask_spread for m in recent_history])
            
            # Determine signal type and strength
            avg_imbalance = np.mean(volume_imbalances)
            spread_trend = np.mean(spread_changes) if len(spread_changes) > 0 else 0
            
            if avg_imbalance > 0.2:  # Strong buy pressure
                signal_type = "buy_pressure"
                strength = min(abs(avg_imbalance), 1.0)
            elif avg_imbalance < -0.2:  # Strong sell pressure
                signal_type = "sell_pressure"
                strength = min(abs(avg_imbalance), 1.0)
            elif abs(avg_imbalance) < 0.05 and abs(spread_trend) < 0.001:
                signal_type = "balanced"
                strength = 0.5
            else:
                signal_type = "stressed"
                strength = min(abs(spread_trend) * 1000, 1.0)
            
            # Calculate confidence
            consistency = 1.0 - np.std(volume_imbalances) if len(volume_imbalances) > 1 else 0.5
            confidence = min(consistency * strength, 1.0)
            
            # Generate execution recommendations
            execution_style, optimal_size = self._recommend_execution_style(
                current_metrics, signal_type, strength
            )
            
            signal = OrderFlowSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                order_flow_delta=spread_trend,
                volume_weighted_pressure=avg_imbalance,
                optimal_trade_size=optimal_size,
                recommended_execution_style=execution_style,
                estimated_impact=current_metrics.market_impact_100k
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating flow signal for {key}: {e}")
            return None
    
    def _recommend_execution_style(self, metrics: LiquidityMetrics, 
                                 signal_type: str, strength: float) -> Tuple[str, float]:
        """Recommend execution style and optimal trade size."""
        try:
            # Base recommendations on liquidity and flow conditions
            if signal_type == "balanced" and metrics.liquidity_score > 0.8:
                return "aggressive", min(metrics.depth_1_pct * 0.1, 50000)
            elif signal_type in ["buy_pressure", "sell_pressure"] and strength > 0.7:
                return "passive", min(metrics.depth_1_pct * 0.05, 25000)
            elif signal_type == "stressed":
                return "twap", min(metrics.depth_1_pct * 0.02, 10000)
            else:
                return "vwap", min(metrics.depth_1_pct * 0.08, 40000)
                
        except Exception as e:
            self.logger.error(f"Error recommending execution style: {e}")
            return "twap", 10000
    
    async def _market_impact_calibrator(self):
        """Calibrate market impact models based on observed data."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    if symbol in self.impact_models:
                        await self._recalibrate_impact_model(symbol)
                
                await asyncio.sleep(3600)  # Recalibrate every hour
                
            except Exception as e:
                self.logger.error(f"Error in market impact calibrator: {e}")
                await asyncio.sleep(1800)
    
    async def _recalibrate_impact_model(self, symbol: str):
        """Recalibrate impact model for symbol."""
        try:
            model = self.impact_models[symbol]
            
            # Collect recent impact observations
            impacts = []
            for exchange in self.exchanges:
                key = f"{exchange}_{symbol}"
                if key in self.flow_history and len(self.flow_history[key]) > 10:
                    recent_metrics = list(self.flow_history[key])[-50:]
                    for metrics in recent_metrics:
                        if metrics.market_impact_100k > 0:
                            impacts.append(metrics.market_impact_100k)
            
            if len(impacts) >= 10:
                # Update alpha based on observed impacts
                median_impact = np.median(impacts)
                model.alpha = median_impact * 0.8  # Conservative adjustment
                
                # Update volatility multiplier
                impact_volatility = np.std(impacts)
                model.volatility_multiplier = 1.0 + impact_volatility * 10
                
                model.last_calibrated = datetime.now()
                
                self.logger.debug(f"Recalibrated impact model for {symbol}: alpha={model.alpha:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error recalibrating impact model for {symbol}: {e}")
    
    async def _flow_signal_generator(self):
        """Generate composite flow signals and alerts."""
        while self.is_active:
            try:
                # Analyze cross-exchange flow patterns
                await self._detect_flow_anomalies()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Error in flow signal generator: {e}")
                await asyncio.sleep(120)
    
    async def _detect_flow_anomalies(self):
        """Detect unusual flow patterns across exchanges."""
        try:
            for symbol in self.symbols:
                symbol_signals = []
                
                for exchange in self.exchanges:
                    key = f"{exchange}_{symbol}"
                    if key in self.order_flow_signals:
                        symbol_signals.append(self.order_flow_signals[key])
                
                if len(symbol_signals) >= 2:
                    # Look for divergent signals
                    signal_types = [s.signal_type for s in symbol_signals]
                    strengths = [s.strength for s in symbol_signals]
                    
                    # Alert if signals are divergent or all showing stress
                    if len(set(signal_types)) > 2:  # Divergent signals
                        self.logger.warning(f"Divergent flow signals detected for {symbol}: {signal_types}")
                    
                    if all(s == "stressed" for s in signal_types):  # Market stress
                        avg_strength = np.mean(strengths)
                        if avg_strength > 0.7:
                            self.logger.warning(f"High market stress detected for {symbol}: {avg_strength:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error detecting flow anomalies: {e}")
    
    async def get_liquidity_metrics(self, symbol: Optional[str] = None, 
                                  exchange: Optional[str] = None) -> Dict[str, LiquidityMetrics]:
        """Get current liquidity metrics."""
        if symbol and exchange:
            key = f"{exchange}_{symbol}"
            return {key: self.liquidity_metrics[key]} if key in self.liquidity_metrics else {}
        elif symbol:
            return {k: v for k, v in self.liquidity_metrics.items() if symbol in k}
        else:
            return self.liquidity_metrics.copy()
    
    async def get_flow_signals(self, symbol: Optional[str] = None) -> Dict[str, OrderFlowSignal]:
        """Get current order flow signals."""
        if symbol:
            return {k: v for k, v in self.order_flow_signals.items() if symbol in k}
        return self.order_flow_signals.copy()
    
    async def estimate_execution_impact(self, symbol: str, trade_size: float, 
                                      side: str = "buy") -> Dict[str, float]:
        """Estimate execution impact for a proposed trade."""
        try:
            results = {}
            
            for exchange in self.exchanges:
                key = f"{exchange}_{symbol}"
                if key in self.liquidity_metrics:
                    metrics = self.liquidity_metrics[key]
                    
                    # Get current order book
                    order_book = await self.market_data.get_order_book(symbol, exchange)
                    if order_book:
                        current_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
                        impact = await self._calculate_order_book_impact(
                            order_book, trade_size, current_price, side
                        )
                        
                        results[exchange] = {
                            "estimated_impact": impact,
                            "liquidity_score": metrics.liquidity_score,
                            "recommended_execution": self.order_flow_signals.get(key, {}).get("recommended_execution_style", "twap")
                        }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error estimating execution impact: {e}")
            return {}
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "is_active": self.is_active,
            "liquidity_metrics_count": len(self.liquidity_metrics),
            "flow_signals_count": len(self.order_flow_signals),
            "impact_models_count": len(self.impact_models),
            "symbols": self.symbols,
            "exchanges": self.exchanges,
            "average_liquidity_score": np.mean([m.liquidity_score for m in self.liquidity_metrics.values()]) if self.liquidity_metrics else 0.0,
            "flow_signal_summary": {
                signal_type: sum(1 for s in self.order_flow_signals.values() if s.signal_type == signal_type)
                for signal_type in ["buy_pressure", "sell_pressure", "balanced", "stressed"]
            }
        }