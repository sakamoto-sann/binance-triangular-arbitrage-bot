"""
Grid Trading Bot v3.0 - Adaptive Grid Engine
Dynamic grid management with market-aware spacing and placement.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .market_analyzer import MarketAnalysis, MarketRegime
from ..utils.config_manager import BotConfig
from ..utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class GridType(Enum):
    """Grid type enumeration."""
    FAST = "fast"      # Short-term scalping grid
    MEDIUM = "medium"  # Medium-term grid
    SLOW = "slow"      # Long-term grid

@dataclass
class GridLevel:
    """Individual grid level definition."""
    level_id: str
    price: float
    side: str  # 'buy' or 'sell'
    quantity: float
    grid_type: GridType
    confidence: float  # Confidence in this level
    priority: int      # Execution priority (1 = highest)
    created_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level_id': self.level_id,
            'price': self.price,
            'side': self.side,
            'quantity': self.quantity,
            'grid_type': self.grid_type.value,
            'confidence': self.confidence,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

@dataclass
class GridConfiguration:
    """Grid configuration parameters."""
    base_spacing: float
    dynamic_spacing: float
    grid_count: int
    buy_levels: int
    sell_levels: int
    position_size: float
    max_position_size: float
    spacing_multiplier: float
    trend_bias: float  # -1 to 1, negative favors buy orders
    volatility_adjustment: float
    
@dataclass
class GridMetrics:
    """Grid performance and status metrics."""
    total_levels: int
    active_levels: int
    buy_levels: int
    sell_levels: int
    filled_levels: int
    avg_spacing: float
    current_bias: float
    portfolio_balance: float
    unrealized_pnl: float
    grid_efficiency: float  # 0-1 score
    last_rebalance: datetime

class AdaptiveGridEngine:
    """
    Advanced grid engine with dynamic spacing and market-aware placement.
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize adaptive grid engine.
        
        Args:
            config: Bot configuration.
        """
        self.config = config
        self.indicators = TechnicalIndicators()
        
        # Grid parameters
        self.base_spacing = config.trading.base_grid_interval
        self.base_position_size = config.trading.base_position_size
        self.grid_count = config.strategy.grid_count
        self.target_btc_ratio = config.trading.target_btc_ratio
        
        # Grid engine parameters - OPTIMIZED FOR ATR-BASED SPACING
        self.atr_spacing_multiplier_min = getattr(config.grid_engine, 'atr_spacing_multiplier_min', 1.0)
        self.atr_spacing_multiplier_max = getattr(config.grid_engine, 'atr_spacing_multiplier_max', 2.0)
        self.grid_levels_min = getattr(config.grid_engine, 'grid_levels_min', 15)
        self.grid_levels_max = getattr(config.grid_engine, 'grid_levels_max', 25)
        self.bull_bias = config.grid_engine.bull_bias
        self.bear_bias = config.grid_engine.bear_bias
        self.sideways_bias = config.grid_engine.sideways_bias
        self.adaptive_spacing = config.grid_engine.adaptive_spacing
        self.staggered_entry = getattr(config.grid_engine, 'staggered_entry', True)
        self.initial_levels = getattr(config.grid_engine, 'initial_levels', 3)
        self.spacing_smoothing = config.grid_engine.spacing_smoothing
        self.dynamic_adjustment_factor = getattr(config.grid_engine, 'dynamic_adjustment_factor', 0.15)
        
        # State management
        self.grid_levels: Dict[str, GridLevel] = {}
        self.grid_configurations: Dict[GridType, GridConfiguration] = {}
        self.last_market_analysis: Optional[MarketAnalysis] = None
        self.last_rebalance: datetime = datetime.now()
        self.spacing_ema: float = 1.0  # Exponential moving average of spacing multiplier
        
        # Performance tracking
        self.filled_levels_history: List[GridLevel] = []
        self.rebalance_history: List[Tuple[datetime, str]] = []
        
        logger.info("Adaptive grid engine initialized")
    
    def update_market_analysis(self, market_analysis: MarketAnalysis) -> None:
        """
        Update grid engine with latest market analysis.
        
        Args:
            market_analysis: Latest market analysis.
        """
        try:
            self.last_market_analysis = market_analysis
            
            # Check if grid recalibration is needed
            if self._should_recalibrate(market_analysis):
                logger.info("Market conditions changed significantly, recalibrating grids")
                self._recalibrate_grids(market_analysis)
            
            # Update grid configurations based on market analysis
            self._update_grid_configurations(market_analysis)
            
        except Exception as e:
            logger.error(f"Error updating market analysis in grid engine: {e}")
    
    def generate_grid_levels(self, current_price: float, 
                           available_balance: Dict[str, float],
                           market_analysis: Optional[MarketAnalysis] = None) -> List[GridLevel]:
        """
        Generate optimized grid levels based on current market conditions.
        
        Args:
            current_price: Current market price.
            available_balance: Available balances {'USDT': amount, 'BTC': amount}.
            market_analysis: Current market analysis.
            
        Returns:
            List of optimized grid levels.
        """
        try:
            if market_analysis is None:
                market_analysis = self.last_market_analysis
            
            if market_analysis is None:
                logger.warning("No market analysis available, using default grid configuration")
                return self._generate_default_grid(current_price, available_balance)
            
            # Calculate dynamic spacing
            spacing_multiplier = self._calculate_dynamic_spacing(market_analysis)
            dynamic_spacing = self.base_spacing * spacing_multiplier
            
            # Determine grid bias based on market regime
            buy_ratio, sell_ratio = self._calculate_grid_bias(market_analysis)
            
            # Calculate position sizes
            position_sizes = self._calculate_position_sizes(
                available_balance, market_analysis, buy_ratio, sell_ratio
            )
            
            # Generate multi-timeframe grids
            grid_levels = []
            
            # Fast grid (short-term scalping)
            fast_levels = self._generate_timeframe_grid(
                GridType.FAST, current_price, dynamic_spacing * 0.5,
                self.grid_count // 2, buy_ratio, sell_ratio,
                position_sizes['fast'], market_analysis
            )
            grid_levels.extend(fast_levels)
            
            # Medium grid (medium-term)
            medium_levels = self._generate_timeframe_grid(
                GridType.MEDIUM, current_price, dynamic_spacing,
                self.grid_count, buy_ratio, sell_ratio,
                position_sizes['medium'], market_analysis
            )
            grid_levels.extend(medium_levels)
            
            # Slow grid (long-term)
            slow_levels = self._generate_timeframe_grid(
                GridType.SLOW, current_price, dynamic_spacing * 2.0,
                self.grid_count // 3, buy_ratio, sell_ratio,
                position_sizes['slow'], market_analysis
            )
            grid_levels.extend(slow_levels)
            
            # Apply smart positioning near support/resistance
            grid_levels = self._apply_smart_positioning(grid_levels, market_analysis)
            
            # Update internal state
            self._update_grid_state(grid_levels)
            
            logger.info(f"Generated {len(grid_levels)} grid levels with {spacing_multiplier:.2f}x spacing")
            return grid_levels
            
        except Exception as e:
            logger.error(f"Error generating grid levels: {e}")
            return self._generate_default_grid(current_price, available_balance)
    
    def _calculate_dynamic_spacing(self, market_analysis: MarketAnalysis) -> float:
        """
        Calculate ATR-based dynamic spacing multiplier - OPTIMIZED VERSION.
        
        Args:
            market_analysis: Current market analysis.
            
        Returns:
            ATR-based spacing multiplier.
        """
        try:
            if not self.adaptive_spacing:
                return self.atr_spacing_multiplier_min
            
            # Get ATR value from market analysis
            atr_value = getattr(market_analysis.volatility_metrics, 'atr', 0.02)  # Default 2%
            
            # Base ATR multiplier - start with minimum
            atr_multiplier = self.atr_spacing_multiplier_min
            
            # Adjust ATR multiplier based on market regime
            regime = market_analysis.regime
            
            if regime == MarketRegime.SIDEWAYS:
                # Tighter grids in sideways markets for more trades
                atr_multiplier = self.atr_spacing_multiplier_min
            elif regime == MarketRegime.BREAKOUT:
                # Wider grids during breakouts to avoid whipsaws
                atr_multiplier = self.atr_spacing_multiplier_max
            elif regime in [MarketRegime.BULL, MarketRegime.BEAR]:
                # Medium spacing for trending markets
                trend_strength = abs(market_analysis.trend_metrics.trend_strength)
                atr_multiplier = self.atr_spacing_multiplier_min + (
                    trend_strength * (self.atr_spacing_multiplier_max - self.atr_spacing_multiplier_min)
                )
            
            # Adjust based on volatility regime for additional safety
            vol_metrics = market_analysis.volatility_metrics
            
            if vol_metrics.volatility_regime == "extreme":
                # Increase spacing in extreme volatility
                atr_multiplier *= 1.3
            elif vol_metrics.volatility_regime == "low":
                # Decrease spacing in low volatility for more opportunities
                atr_multiplier *= 0.8
            
            # Clamp to our defined range
            atr_multiplier = max(self.atr_spacing_multiplier_min, 
                               min(self.atr_spacing_multiplier_max, atr_multiplier))
            
            # Adjust based on trend momentum for additional refinement
            momentum = abs(market_analysis.trend_metrics.trend_momentum)
            atr_multiplier *= (1.0 + momentum * 0.2)  # Modest momentum adjustment
            
            # Final clamp after momentum adjustment
            atr_multiplier = max(self.atr_spacing_multiplier_min, 
                               min(self.atr_spacing_multiplier_max, atr_multiplier))
            
            # Apply exponential smoothing to reduce volatility
            self.spacing_ema = (self.spacing_smoothing * atr_multiplier + 
                              (1 - self.spacing_smoothing) * self.spacing_ema)
            
            return self.spacing_ema
            
        except Exception as e:
            logger.error(f"Error calculating dynamic spacing: {e}")
            return 1.0
    
    def _calculate_grid_bias(self, market_analysis: MarketAnalysis) -> Tuple[float, float]:
        """
        Calculate grid bias (buy vs sell order distribution).
        
        Args:
            market_analysis: Current market analysis.
            
        Returns:
            Tuple of (buy_ratio, sell_ratio).
        """
        try:
            regime = market_analysis.regime
            confidence = market_analysis.confidence
            
            # Base ratios based on regime
            if regime == MarketRegime.BULL:
                base_buy_ratio = self.bull_bias
            elif regime == MarketRegime.BEAR:
                base_buy_ratio = self.bear_bias
            else:  # SIDEWAYS, BREAKOUT, UNKNOWN
                base_buy_ratio = self.sideways_bias
            
            # Adjust based on confidence
            adjustment = (confidence - 0.5) * 0.2  # Max 10% adjustment
            
            if regime == MarketRegime.BULL:
                buy_ratio = max(0.1, min(0.4, base_buy_ratio - adjustment))
            elif regime == MarketRegime.BEAR:
                buy_ratio = max(0.6, min(0.9, base_buy_ratio + adjustment))
            else:
                buy_ratio = max(0.3, min(0.7, base_buy_ratio))
            
            sell_ratio = 1.0 - buy_ratio
            
            # Further adjust based on current portfolio balance
            # This helps with rebalancing toward target ratio
            # Implementation would require current portfolio state
            
            logger.debug(f"Grid bias calculated: {buy_ratio:.1%} buy, {sell_ratio:.1%} sell")
            return buy_ratio, sell_ratio
            
        except Exception as e:
            logger.error(f"Error calculating grid bias: {e}")
            return 0.5, 0.5
    
    def _calculate_position_sizes(self, available_balance: Dict[str, float],
                                market_analysis: MarketAnalysis,
                                buy_ratio: float, sell_ratio: float) -> Dict[str, float]:
        """
        Calculate optimal position sizes for different grid types.
        
        Args:
            available_balance: Available balances.
            market_analysis: Current market analysis.
            buy_ratio: Ratio of buy orders.
            sell_ratio: Ratio of sell orders.
            
        Returns:
            Position sizes by grid type.
        """
        try:
            base_size = self.base_position_size
            
            # Adjust base size based on volatility
            vol_regime = market_analysis.volatility_metrics.volatility_regime
            if vol_regime == "low":
                vol_multiplier = 1.2  # Larger positions in low volatility
            elif vol_regime == "high":
                vol_multiplier = 0.8  # Smaller positions in high volatility
            elif vol_regime == "extreme":
                vol_multiplier = 0.5  # Much smaller positions in extreme volatility
            else:
                vol_multiplier = 1.0
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + market_analysis.confidence * 0.5
            
            # Calculate adjusted base size
            adjusted_base = base_size * vol_multiplier * confidence_multiplier
            
            # Distribute across grid types
            position_sizes = {
                'fast': adjusted_base * 0.3,    # 30% for fast grid
                'medium': adjusted_base * 0.5,  # 50% for medium grid
                'slow': adjusted_base * 0.2     # 20% for slow grid
            }
            
            # Apply maximum position size limits
            max_size = self.config.risk_management.max_position_size
            for grid_type in position_sizes:
                position_sizes[grid_type] = min(position_sizes[grid_type], max_size)
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return {'fast': base_size * 0.5, 'medium': base_size, 'slow': base_size * 0.8}
    
    def _generate_timeframe_grid(self, grid_type: GridType, current_price: float,
                               spacing: float, level_count: int,
                               buy_ratio: float, sell_ratio: float,
                               position_size: float, 
                               market_analysis: MarketAnalysis) -> List[GridLevel]:
        """
        Generate grid levels for specific timeframe.
        
        Args:
            grid_type: Type of grid (fast/medium/slow).
            current_price: Current market price.
            spacing: Price spacing between levels.
            level_count: Total number of levels.
            buy_ratio: Ratio of buy orders.
            sell_ratio: Ratio of sell orders.
            position_size: Position size for this grid type.
            market_analysis: Current market analysis.
            
        Returns:
            List of grid levels.
        """
        try:
            levels = []
            
            # Calculate number of buy and sell levels
            buy_levels = max(1, int(level_count * buy_ratio))
            sell_levels = max(1, int(level_count * sell_ratio))
            
            # Adjust if total exceeds level_count
            total_levels = buy_levels + sell_levels
            if total_levels > level_count:
                if buy_levels > sell_levels:
                    buy_levels = level_count - sell_levels
                else:
                    sell_levels = level_count - buy_levels
            
            # Generate buy levels (below current price)
            for i in range(1, buy_levels + 1):
                price = current_price - (spacing * i)
                
                # Skip if price is too low
                if price <= 0:
                    continue
                
                # Calculate confidence based on distance and market analysis
                confidence = self._calculate_level_confidence(
                    'buy', price, current_price, market_analysis
                )
                
                # Calculate priority (closer = higher priority)
                priority = buy_levels - i + 1
                
                level = GridLevel(
                    level_id=f"{grid_type.value}_buy_{i}_{int(datetime.now().timestamp())}",
                    price=price,
                    side='buy',
                    quantity=position_size,
                    grid_type=grid_type,
                    confidence=confidence,
                    priority=priority,
                    created_at=datetime.now()
                )
                levels.append(level)
            
            # Generate sell levels (above current price)
            for i in range(1, sell_levels + 1):
                price = current_price + (spacing * i)
                
                # Calculate confidence
                confidence = self._calculate_level_confidence(
                    'sell', price, current_price, market_analysis
                )
                
                # Calculate priority
                priority = sell_levels - i + 1
                
                level = GridLevel(
                    level_id=f"{grid_type.value}_sell_{i}_{int(datetime.now().timestamp())}",
                    price=price,
                    side='sell',
                    quantity=position_size,
                    grid_type=grid_type,
                    confidence=confidence,
                    priority=priority,
                    created_at=datetime.now()
                )
                levels.append(level)
            
            return levels
            
        except Exception as e:
            logger.error(f"Error generating {grid_type.value} grid: {e}")
            return []
    
    def _calculate_level_confidence(self, side: str, level_price: float, 
                                  current_price: float, 
                                  market_analysis: MarketAnalysis) -> float:
        """
        Calculate confidence score for a grid level.
        
        Args:
            side: Order side ('buy' or 'sell').
            level_price: Price of the grid level.
            current_price: Current market price.
            market_analysis: Current market analysis.
            
        Returns:
            Confidence score (0-1).
        """
        try:
            base_confidence = 0.6
            
            # Distance factor (closer to current price = higher confidence)
            distance_ratio = abs(level_price - current_price) / current_price
            distance_factor = max(0.3, 1.0 - distance_ratio * 2)
            
            # Market regime factor
            regime = market_analysis.regime
            if side == 'buy':
                if regime == MarketRegime.BEAR:
                    regime_factor = 1.2  # Higher confidence for buy orders in bear market
                elif regime == MarketRegime.BULL:
                    regime_factor = 0.8  # Lower confidence for buy orders in bull market
                else:
                    regime_factor = 1.0
            else:  # sell
                if regime == MarketRegime.BULL:
                    regime_factor = 1.2  # Higher confidence for sell orders in bull market
                elif regime == MarketRegime.BEAR:
                    regime_factor = 0.8  # Lower confidence for sell orders in bear market
                else:
                    regime_factor = 1.0
            
            # Support/resistance factor
            sr_factor = 1.0
            if side == 'buy' and market_analysis.support_levels:
                # Check if level is near support
                closest_support = min(market_analysis.support_levels, 
                                    key=lambda x: abs(x - level_price))
                if abs(closest_support - level_price) / level_price < 0.02:  # Within 2%
                    sr_factor = 1.3
            elif side == 'sell' and market_analysis.resistance_levels:
                # Check if level is near resistance
                closest_resistance = min(market_analysis.resistance_levels,
                                       key=lambda x: abs(x - level_price))
                if abs(closest_resistance - level_price) / level_price < 0.02:  # Within 2%
                    sr_factor = 1.3
            
            # Trend momentum factor
            momentum = market_analysis.trend_metrics.trend_momentum
            if side == 'buy' and momentum < 0:  # Negative momentum favors buy orders
                momentum_factor = 1.1
            elif side == 'sell' and momentum > 0:  # Positive momentum favors sell orders
                momentum_factor = 1.1
            else:
                momentum_factor = 0.95
            
            # Calculate final confidence
            confidence = (base_confidence * distance_factor * regime_factor * 
                         sr_factor * momentum_factor)
            
            # Apply market analysis confidence
            confidence *= market_analysis.confidence
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating level confidence: {e}")
            return 0.5
    
    def _apply_smart_positioning(self, grid_levels: List[GridLevel], 
                               market_analysis: MarketAnalysis) -> List[GridLevel]:
        """
        Apply smart positioning adjustments near support/resistance levels.
        
        Args:
            grid_levels: Initial grid levels.
            market_analysis: Current market analysis.
            
        Returns:
            Adjusted grid levels.
        """
        try:
            adjusted_levels = []
            
            for level in grid_levels:
                adjusted_level = level
                
                # Check for support/resistance proximity
                if level.side == 'buy' and market_analysis.support_levels:
                    # Find closest support level
                    closest_support = min(market_analysis.support_levels,
                                        key=lambda x: abs(x - level.price))
                    
                    # If close to support, adjust price and increase confidence
                    distance = abs(closest_support - level.price) / level.price
                    if distance < 0.015:  # Within 1.5%
                        adjusted_level.price = closest_support * 1.001  # Slightly above support
                        adjusted_level.confidence = min(0.95, level.confidence * 1.2)
                        adjusted_level.priority = min(10, level.priority + 2)
                
                elif level.side == 'sell' and market_analysis.resistance_levels:
                    # Find closest resistance level
                    closest_resistance = min(market_analysis.resistance_levels,
                                           key=lambda x: abs(x - level.price))
                    
                    # If close to resistance, adjust price and increase confidence
                    distance = abs(closest_resistance - level.price) / level.price
                    if distance < 0.015:  # Within 1.5%
                        adjusted_level.price = closest_resistance * 0.999  # Slightly below resistance
                        adjusted_level.confidence = min(0.95, level.confidence * 1.2)
                        adjusted_level.priority = min(10, level.priority + 2)
                
                adjusted_levels.append(adjusted_level)
            
            return adjusted_levels
            
        except Exception as e:
            logger.error(f"Error applying smart positioning: {e}")
            return grid_levels
    
    def _should_recalibrate(self, market_analysis: MarketAnalysis) -> bool:
        """
        Determine if grid recalibration is needed.
        
        Args:
            market_analysis: Current market analysis.
            
        Returns:
            Whether recalibration is needed.
        """
        try:
            # Always recalibrate if no previous analysis
            if self.last_market_analysis is None:
                return True
            
            # Check time since last rebalance
            time_since_rebalance = datetime.now() - self.last_rebalance
            rebalance_frequency = timedelta(hours=self.config.strategy.rebalance_frequency)
            
            if time_since_rebalance >= rebalance_frequency:
                return True
            
            # Check for regime change
            if market_analysis.regime != self.last_market_analysis.regime:
                return True
            
            # Check for significant confidence change
            confidence_change = abs(market_analysis.confidence - 
                                  self.last_market_analysis.confidence)
            if confidence_change > 0.3:
                return True
            
            # Check for significant volatility change
            vol_change = abs(
                market_analysis.volatility_metrics.current_volatility -
                self.last_market_analysis.volatility_metrics.current_volatility
            )
            if vol_change > 0.02:  # 2% volatility change
                return True
            
            # Check for trend reversal
            current_trend = market_analysis.trend_metrics.trend_strength
            previous_trend = self.last_market_analysis.trend_metrics.trend_strength
            if current_trend * previous_trend < 0:  # Sign change = trend reversal
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking recalibration conditions: {e}")
            return False
    
    def _recalibrate_grids(self, market_analysis: MarketAnalysis) -> None:
        """
        Recalibrate all grid configurations.
        
        Args:
            market_analysis: Current market analysis.
        """
        try:
            # Clear existing grid levels
            self.grid_levels.clear()
            
            # Update configurations
            self._update_grid_configurations(market_analysis)
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
            # Record recalibration
            self.rebalance_history.append((
                datetime.now(),
                f"Recalibrated due to {market_analysis.regime.value} regime"
            ))
            
            # Keep history limited
            if len(self.rebalance_history) > 100:
                self.rebalance_history = self.rebalance_history[-100:]
            
            logger.info(f"Grid recalibrated for {market_analysis.regime.value} market regime")
            
        except Exception as e:
            logger.error(f"Error recalibrating grids: {e}")
    
    def _update_grid_configurations(self, market_analysis: MarketAnalysis) -> None:
        """
        Update grid configurations based on market analysis.
        
        Args:
            market_analysis: Current market analysis.
        """
        try:
            # Calculate dynamic parameters
            spacing_multiplier = self._calculate_dynamic_spacing(market_analysis)
            buy_ratio, sell_ratio = self._calculate_grid_bias(market_analysis)
            
            # Update configurations for each grid type
            for grid_type in GridType:
                if grid_type == GridType.FAST:
                    spacing = self.base_spacing * spacing_multiplier * 0.5
                    count = self.grid_count // 2
                elif grid_type == GridType.MEDIUM:
                    spacing = self.base_spacing * spacing_multiplier
                    count = self.grid_count
                else:  # SLOW
                    spacing = self.base_spacing * spacing_multiplier * 2.0
                    count = self.grid_count // 3
                
                config = GridConfiguration(
                    base_spacing=self.base_spacing,
                    dynamic_spacing=spacing,
                    grid_count=count,
                    buy_levels=max(1, int(count * buy_ratio)),
                    sell_levels=max(1, int(count * sell_ratio)),
                    position_size=self.base_position_size,
                    max_position_size=self.config.risk_management.max_position_size,
                    spacing_multiplier=spacing_multiplier,
                    trend_bias=market_analysis.trend_metrics.trend_strength,
                    volatility_adjustment=market_analysis.volatility_metrics.current_volatility
                )
                
                self.grid_configurations[grid_type] = config
            
        except Exception as e:
            logger.error(f"Error updating grid configurations: {e}")
    
    def _update_grid_state(self, grid_levels: List[GridLevel]) -> None:
        """
        Update internal grid state.
        
        Args:
            grid_levels: New grid levels.
        """
        try:
            # Clear old levels
            self.grid_levels.clear()
            
            # Add new levels
            for level in grid_levels:
                self.grid_levels[level.level_id] = level
            
        except Exception as e:
            logger.error(f"Error updating grid state: {e}")
    
    def _generate_default_grid(self, current_price: float, 
                             available_balance: Dict[str, float]) -> List[GridLevel]:
        """
        Generate default grid when market analysis is unavailable.
        
        Args:
            current_price: Current market price.
            available_balance: Available balances.
            
        Returns:
            Default grid levels.
        """
        try:
            levels = []
            spacing = self.base_spacing
            position_size = self.base_position_size
            
            # Generate balanced grid (50/50 buy/sell)
            half_count = self.grid_count // 2
            
            # Buy levels
            for i in range(1, half_count + 1):
                price = current_price - (spacing * i)
                if price > 0:
                    level = GridLevel(
                        level_id=f"default_buy_{i}_{int(datetime.now().timestamp())}",
                        price=price,
                        side='buy',
                        quantity=position_size,
                        grid_type=GridType.MEDIUM,
                        confidence=0.5,
                        priority=half_count - i + 1,
                        created_at=datetime.now()
                    )
                    levels.append(level)
            
            # Sell levels
            for i in range(1, half_count + 1):
                price = current_price + (spacing * i)
                level = GridLevel(
                    level_id=f"default_sell_{i}_{int(datetime.now().timestamp())}",
                    price=price,
                    side='sell',
                    quantity=position_size,
                    grid_type=GridType.MEDIUM,
                    confidence=0.5,
                    priority=half_count - i + 1,
                    created_at=datetime.now()
                )
                levels.append(level)
            
            logger.info(f"Generated {len(levels)} default grid levels")
            return levels
            
        except Exception as e:
            logger.error(f"Error generating default grid: {e}")
            return []
    
    def mark_level_filled(self, level_id: str, filled_quantity: float, 
                         fill_price: float) -> bool:
        """
        Mark a grid level as filled.
        
        Args:
            level_id: ID of the filled level.
            filled_quantity: Quantity that was filled.
            fill_price: Price at which it was filled.
            
        Returns:
            Success status.
        """
        try:
            if level_id in self.grid_levels:
                level = self.grid_levels[level_id]
                level.is_active = False
                
                # Add to filled history
                self.filled_levels_history.append(level)
                
                # Keep history limited
                if len(self.filled_levels_history) > 1000:
                    self.filled_levels_history = self.filled_levels_history[-1000:]
                
                # Remove from active levels
                del self.grid_levels[level_id]
                
                logger.info(f"Grid level {level_id} marked as filled: {filled_quantity} at {fill_price}")
                return True
            else:
                logger.warning(f"Grid level {level_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error marking level as filled: {e}")
            return False
    
    def get_active_levels(self, grid_type: Optional[GridType] = None) -> List[GridLevel]:
        """
        Get currently active grid levels.
        
        Args:
            grid_type: Filter by grid type (optional).
            
        Returns:
            List of active grid levels.
        """
        try:
            levels = list(self.grid_levels.values())
            
            if grid_type is not None:
                levels = [level for level in levels if level.grid_type == grid_type]
            
            # Sort by priority (highest first)
            levels.sort(key=lambda x: x.priority, reverse=True)
            
            return levels
            
        except Exception as e:
            logger.error(f"Error getting active levels: {e}")
            return []
    
    def get_grid_metrics(self) -> GridMetrics:
        """
        Get current grid performance metrics.
        
        Returns:
            Grid metrics.
        """
        try:
            active_levels = list(self.grid_levels.values())
            buy_levels = [l for l in active_levels if l.side == 'buy']
            sell_levels = [l for l in active_levels if l.side == 'sell']
            
            # Calculate average spacing
            if len(active_levels) > 1:
                prices = sorted([level.price for level in active_levels])
                spacings = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
                avg_spacing = np.mean(spacings) if spacings else 0
            else:
                avg_spacing = 0
            
            # Calculate current bias
            total_levels = len(active_levels)
            current_bias = len(buy_levels) / total_levels if total_levels > 0 else 0.5
            
            # Simple efficiency metric (placeholder for more complex calculation)
            grid_efficiency = min(1.0, len(self.filled_levels_history) / 100) if self.filled_levels_history else 0
            
            return GridMetrics(
                total_levels=total_levels,
                active_levels=len(active_levels),
                buy_levels=len(buy_levels),
                sell_levels=len(sell_levels),
                filled_levels=len(self.filled_levels_history),
                avg_spacing=avg_spacing,
                current_bias=current_bias,
                portfolio_balance=0,  # Would need portfolio state
                unrealized_pnl=0,     # Would need portfolio state
                grid_efficiency=grid_efficiency,
                last_rebalance=self.last_rebalance
            )
            
        except Exception as e:
            logger.error(f"Error calculating grid metrics: {e}")
            return GridMetrics(0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, datetime.now())
    
    def optimize_grid_parameters(self, performance_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Optimize grid parameters based on performance history.
        
        Args:
            performance_history: Historical performance data.
            
        Returns:
            Optimized parameters.
        """
        try:
            if not performance_history:
                return {}
            
            # Simple optimization based on recent performance
            # In a production system, this could use more sophisticated methods
            
            optimizations = {}
            
            # Analyze spacing effectiveness
            recent_performance = performance_history[-20:] if len(performance_history) >= 20 else performance_history
            
            # If recent performance is poor, suggest adjustments
            avg_return = np.mean([p.get('return_pct', 0) for p in recent_performance])
            
            if avg_return < 0:
                # Poor performance - suggest wider spacing
                optimizations['spacing_multiplier'] = min(self.max_spacing_multiplier, 
                                                        self.spacing_ema * 1.1)
            elif avg_return > 0.1:
                # Good performance - can try tighter spacing
                optimizations['spacing_multiplier'] = max(self.min_spacing_multiplier,
                                                        self.spacing_ema * 0.95)
            
            # Analyze grid bias effectiveness
            # This would require more detailed performance tracking by grid side
            
            logger.info(f"Grid optimization suggested: {optimizations}")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing grid parameters: {e}")
            return {}