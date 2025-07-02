"""
Grid Trading Bot v3.0 - Adaptive Grid Strategy
Main grid trading strategy with market-aware adaptation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, StrategyConfig, StrategySignal, SignalType, SignalStrength
from ..core.market_analyzer import MarketAnalysis, MarketRegime
from ..core.grid_engine import AdaptiveGridEngine, GridLevel, GridType
from ..core.risk_manager import RiskMetrics, RiskLevel
from ..utils.config_manager import BotConfig
from ..utils.data_manager import PositionData

logger = logging.getLogger(__name__)

class AdaptiveGridStrategy(BaseStrategy):
    """
    Adaptive grid trading strategy that adjusts to market conditions.
    """
    
    def __init__(self, config: BotConfig, strategy_config: StrategyConfig):
        """
        Initialize adaptive grid strategy.
        
        Args:
            config: Bot configuration.
            strategy_config: Strategy-specific configuration.
        """
        super().__init__(config, strategy_config)
        
        # Initialize grid engine
        self.grid_engine = AdaptiveGridEngine(config)
        
        # Strategy parameters
        self.min_signal_confidence = strategy_config.parameters.get('min_signal_confidence', 0.6)
        self.rebalance_threshold = strategy_config.parameters.get('rebalance_threshold', 0.3)
        self.max_position_ratio = strategy_config.parameters.get('max_position_ratio', 0.8)
        self.volatility_adjustment = strategy_config.parameters.get('volatility_adjustment', True)
        self.regime_adaptation = strategy_config.parameters.get('regime_adaptation', True)
        
        # Grid state
        self.active_grid_levels: List[GridLevel] = []
        self.last_rebalance: datetime = datetime.now()
        self.rebalance_frequency = timedelta(hours=config.strategy.rebalance_frequency)
        
        # Performance tracking
        self.grid_performance_history: List[Dict[str, Any]] = []
        self.regime_performance: Dict[str, float] = {}
        
        # Risk management
        self.max_grid_exposure = config.risk_management.max_position_size * 0.8
        self.emergency_stop_threshold = 0.15  # 15% drawdown triggers emergency stop
        
        logger.info("Adaptive grid strategy initialized")
    
    def generate_signal(self, price_data: pd.DataFrame,
                       market_analysis: MarketAnalysis,
                       risk_metrics: RiskMetrics,
                       current_positions: List[PositionData]) -> Optional[StrategySignal]:
        """
        Generate trading signal based on current market conditions.
        
        Args:
            price_data: Historical price data.
            market_analysis: Current market analysis.
            risk_metrics: Current risk metrics.
            current_positions: Current portfolio positions.
            
        Returns:
            Trading signal or None.
        """
        try:
            # Update market state
            self.update_market_state(price_data, market_analysis, risk_metrics)
            
            # Check if strategy should be active
            if not self._should_trade(market_analysis, risk_metrics):
                return None
            
            current_price = price_data.iloc[-1]['close']
            
            # Check for emergency stop
            if self._should_emergency_stop(risk_metrics):
                return self._create_emergency_stop_signal(current_price)
            
            # Check for rebalancing
            if self.should_rebalance(market_analysis, self.performance_metrics):
                return self._create_rebalance_signal(current_price, market_analysis)
            
            # Generate grid-based signals
            signal = self._generate_grid_signal(
                current_price, market_analysis, risk_metrics, current_positions
            )
            
            if signal:
                # Validate signal before returning
                is_valid, reason = self.validate_signal(signal)
                if not is_valid:
                    logger.warning(f"Signal validation failed: {reason}")
                    return None
                
                # Record the signal
                self.record_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def calculate_position_size(self, signal: StrategySignal,
                              available_capital: float,
                              risk_metrics: RiskMetrics) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk management.
        
        Args:
            signal: Trading signal.
            available_capital: Available trading capital.
            risk_metrics: Current risk metrics.
            
        Returns:
            Optimal position size.
        """
        try:
            # Base position size from configuration
            base_size = self.config.trading.base_position_size
            
            # Adjust based on signal strength and confidence
            strength_multiplier = self._get_strength_multiplier(signal.strength)
            confidence_multiplier = signal.confidence
            
            # Adjust based on market volatility
            volatility_multiplier = 1.0
            if self.volatility_adjustment and self.last_market_analysis:
                vol_regime = self.last_market_analysis.volatility_metrics.volatility_regime
                if vol_regime == "low":
                    volatility_multiplier = 1.2
                elif vol_regime == "high":
                    volatility_multiplier = 0.8
                elif vol_regime == "extreme":
                    volatility_multiplier = 0.5
            
            # Adjust based on risk level
            risk_multiplier = 1.0
            if risk_metrics.risk_level == RiskLevel.HIGH:
                risk_multiplier = 0.7
            elif risk_metrics.risk_level == RiskLevel.EXTREME:
                risk_multiplier = 0.4
            
            # Calculate adjusted position size
            adjusted_size = (base_size * strength_multiplier * confidence_multiplier * 
                           volatility_multiplier * risk_multiplier)
            
            # Apply maximum position size limits
            max_size = min(self.max_grid_exposure, available_capital * self.max_position_ratio)
            final_size = min(adjusted_size, max_size)
            
            # Ensure minimum size
            min_size = self.config.trading.min_order_size
            if final_size < min_size:
                final_size = 0.0
            
            logger.debug(f"Position size calculated: {final_size:.6f} (base: {base_size}, "
                        f"strength: {strength_multiplier}, confidence: {confidence_multiplier})")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config.trading.base_position_size * 0.5
    
    def should_rebalance(self, market_analysis: MarketAnalysis,
                        performance_metrics: Dict[str, Any]) -> bool:
        """
        Determine if grid should be rebalanced.
        
        Args:
            market_analysis: Current market analysis.
            performance_metrics: Current performance metrics.
            
        Returns:
            Whether to rebalance.
        """
        try:
            # Time-based rebalancing
            time_since_rebalance = datetime.now() - self.last_rebalance
            if time_since_rebalance >= self.rebalance_frequency:
                return True
            
            # Market regime change
            if (self.last_market_analysis and 
                market_analysis.regime != self.last_market_analysis.regime):
                return True
            
            # Significant confidence change
            if (self.last_market_analysis and
                abs(market_analysis.confidence - self.last_market_analysis.confidence) > self.rebalance_threshold):
                return True
            
            # Poor grid performance
            if self._is_grid_performance_poor():
                return True
            
            # High volatility change
            if (self.last_market_analysis and
                abs(market_analysis.volatility_metrics.current_volatility - 
                    self.last_market_analysis.volatility_metrics.current_volatility) > 0.02):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalance conditions: {e}")
            return False
    
    def update_grid_levels(self, available_balance: Dict[str, float]) -> List[GridLevel]:
        """
        Update grid levels based on current market conditions.
        
        Args:
            available_balance: Available trading balances.
            
        Returns:
            Updated grid levels.
        """
        try:
            if not self.last_market_analysis:
                logger.warning("No market analysis available for grid update")
                return []
            
            # Update grid engine with latest market analysis
            self.grid_engine.update_market_analysis(self.last_market_analysis)
            
            # Generate new grid levels
            new_grid_levels = self.grid_engine.generate_grid_levels(
                self.current_price, available_balance, self.last_market_analysis
            )
            
            # Update active grid levels
            self.active_grid_levels = new_grid_levels
            
            # Record grid update
            self._record_grid_update(new_grid_levels)
            
            logger.info(f"Grid levels updated: {len(new_grid_levels)} levels generated")
            return new_grid_levels
            
        except Exception as e:
            logger.error(f"Error updating grid levels: {e}")
            return []
    
    def _generate_grid_signal(self, current_price: float,
                            market_analysis: MarketAnalysis,
                            risk_metrics: RiskMetrics,
                            current_positions: List[PositionData]) -> Optional[StrategySignal]:
        """Generate grid-based trading signal."""
        try:
            # Check if we need to establish new grid levels
            if not self.active_grid_levels:
                return self._create_establish_grid_signal(current_price, market_analysis)
            
            # Check for grid level fills and replacements
            fill_signal = self._check_grid_fills(current_price, market_analysis)
            if fill_signal:
                return fill_signal
            
            # Check for grid optimization opportunities
            optimization_signal = self._check_grid_optimization(
                current_price, market_analysis, risk_metrics
            )
            if optimization_signal:
                return optimization_signal
            
            # No signal needed
            return None
            
        except Exception as e:
            logger.error(f"Error generating grid signal: {e}")
            return None
    
    def _create_establish_grid_signal(self, current_price: float,
                                    market_analysis: MarketAnalysis) -> StrategySignal:
        """Create signal to establish initial grid."""
        try:
            # Determine signal strength based on market conditions
            if market_analysis.regime == MarketRegime.SIDEWAYS:
                strength = SignalStrength.STRONG
            elif market_analysis.regime in [MarketRegime.BULL, MarketRegime.BEAR]:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Base quantity for grid establishment
            base_quantity = self.config.trading.base_position_size
            
            return StrategySignal(
                signal_type=SignalType.BUY,  # Establish grid with buy order
                strength=strength,
                confidence=market_analysis.confidence,
                symbol=self.config.trading.symbol,
                price=current_price,
                quantity=base_quantity,
                reason=f"Establish grid in {market_analysis.regime.value} market",
                timestamp=datetime.now(),
                metadata={
                    'signal_source': 'grid_establishment',
                    'market_regime': market_analysis.regime.value,
                    'volatility_regime': market_analysis.volatility_metrics.volatility_regime
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating establish grid signal: {e}")
            return None
    
    def _create_rebalance_signal(self, current_price: float,
                               market_analysis: MarketAnalysis) -> StrategySignal:
        """Create rebalancing signal."""
        try:
            return StrategySignal(
                signal_type=SignalType.REBALANCE,
                strength=SignalStrength.MODERATE,
                confidence=0.8,
                symbol=self.config.trading.symbol,
                price=current_price,
                quantity=0.0,  # Rebalance doesn't specify quantity
                reason=f"Rebalance for {market_analysis.regime.value} regime",
                timestamp=datetime.now(),
                metadata={
                    'signal_source': 'rebalance',
                    'previous_regime': self.last_market_analysis.regime.value if self.last_market_analysis else 'unknown',
                    'new_regime': market_analysis.regime.value,
                    'confidence_change': abs(market_analysis.confidence - 
                                           (self.last_market_analysis.confidence if self.last_market_analysis else 0.5))
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating rebalance signal: {e}")
            return None
    
    def _create_emergency_stop_signal(self, current_price: float) -> StrategySignal:
        """Create emergency stop signal."""
        try:
            return StrategySignal(
                signal_type=SignalType.CLOSE_ALL,
                strength=SignalStrength.VERY_STRONG,
                confidence=1.0,
                symbol=self.config.trading.symbol,
                price=current_price,
                quantity=0.0,
                reason="Emergency stop due to high risk",
                timestamp=datetime.now(),
                metadata={
                    'signal_source': 'emergency_stop',
                    'risk_level': self.last_risk_metrics.risk_level.value if self.last_risk_metrics else 'unknown',
                    'drawdown': self.last_risk_metrics.current_drawdown if self.last_risk_metrics else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating emergency stop signal: {e}")
            return None
    
    def _check_grid_fills(self, current_price: float,
                         market_analysis: MarketAnalysis) -> Optional[StrategySignal]:
        """Check for grid level fills and create replacement signals."""
        try:
            # This would integrate with order manager to check for filled orders
            # For now, return None as this requires order status integration
            return None
            
        except Exception as e:
            logger.error(f"Error checking grid fills: {e}")
            return None
    
    def _check_grid_optimization(self, current_price: float,
                               market_analysis: MarketAnalysis,
                               risk_metrics: RiskMetrics) -> Optional[StrategySignal]:
        """Check for grid optimization opportunities."""
        try:
            # Calculate grid efficiency
            grid_metrics = self.grid_engine.get_grid_metrics()
            
            # If efficiency is low, suggest optimization
            if grid_metrics.grid_efficiency < 0.3:  # 30% efficiency threshold
                return StrategySignal(
                    signal_type=SignalType.REBALANCE,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    symbol=self.config.trading.symbol,
                    price=current_price,
                    quantity=0.0,
                    reason=f"Grid optimization needed (efficiency: {grid_metrics.grid_efficiency:.1%})",
                    timestamp=datetime.now(),
                    metadata={
                        'signal_source': 'grid_optimization',
                        'current_efficiency': grid_metrics.grid_efficiency,
                        'total_levels': grid_metrics.total_levels,
                        'active_levels': grid_metrics.active_levels
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking grid optimization: {e}")
            return None
    
    def _should_trade(self, market_analysis: MarketAnalysis,
                     risk_metrics: RiskMetrics) -> bool:
        """Determine if strategy should trade in current conditions."""
        try:
            # Don't trade if strategy is not active
            if not self.is_active:
                return False
            
            # Don't trade in extreme risk conditions
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                return False
            
            # Don't trade if confidence is too low
            if market_analysis.confidence < self.min_signal_confidence:
                return False
            
            # Don't trade during high volatility spikes
            if market_analysis.volatility_metrics.volatility_regime == "extreme":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {e}")
            return False
    
    def _should_emergency_stop(self, risk_metrics: RiskMetrics) -> bool:
        """Check if emergency stop should be triggered."""
        try:
            # High drawdown
            if risk_metrics.current_drawdown > self.emergency_stop_threshold:
                return True
            
            # Extreme risk level
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                return True
            
            # Circuit breaker active (would need integration with risk manager)
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency stop conditions: {e}")
            return False
    
    def _get_strength_multiplier(self, strength: SignalStrength) -> float:
        """Get position size multiplier based on signal strength."""
        multipliers = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.STRONG: 1.5,
            SignalStrength.VERY_STRONG: 2.0
        }
        return multipliers.get(strength, 1.0)
    
    def _is_grid_performance_poor(self) -> bool:
        """Check if grid performance is poor and needs rebalancing."""
        try:
            if not self.grid_performance_history:
                return False
            
            # Check recent performance
            recent_performance = self.grid_performance_history[-10:]  # Last 10 records
            if len(recent_performance) < 5:
                return False
            
            # Calculate average efficiency
            avg_efficiency = np.mean([p.get('efficiency', 0) for p in recent_performance])
            
            # Poor performance threshold
            return avg_efficiency < 0.4  # 40% efficiency threshold
            
        except Exception as e:
            logger.error(f"Error checking grid performance: {e}")
            return False
    
    def _record_grid_update(self, grid_levels: List[GridLevel]) -> None:
        """Record grid update for performance tracking."""
        try:
            grid_metrics = self.grid_engine.get_grid_metrics()
            
            performance_record = {
                'timestamp': datetime.now(),
                'total_levels': len(grid_levels),
                'buy_levels': len([l for l in grid_levels if l.side == 'buy']),
                'sell_levels': len([l for l in grid_levels if l.side == 'sell']),
                'avg_confidence': np.mean([l.confidence for l in grid_levels]),
                'efficiency': grid_metrics.grid_efficiency,
                'market_regime': self.last_market_analysis.regime.value if self.last_market_analysis else 'unknown'
            }
            
            self.grid_performance_history.append(performance_record)
            
            # Keep limited history
            if len(self.grid_performance_history) > 1000:
                self.grid_performance_history = self.grid_performance_history[-1000:]
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
        except Exception as e:
            logger.error(f"Error recording grid update: {e}")
    
    def get_grid_status(self) -> Dict[str, Any]:
        """
        Get current grid status.
        
        Returns:
            Grid status information.
        """
        try:
            grid_metrics = self.grid_engine.get_grid_metrics()
            
            return {
                'active_levels': len(self.active_grid_levels),
                'grid_metrics': {
                    'total_levels': grid_metrics.total_levels,
                    'active_levels': grid_metrics.active_levels,
                    'buy_levels': grid_metrics.buy_levels,
                    'sell_levels': grid_metrics.sell_levels,
                    'efficiency': grid_metrics.grid_efficiency,
                    'current_bias': grid_metrics.current_bias
                },
                'last_rebalance': self.last_rebalance.isoformat(),
                'next_rebalance': (self.last_rebalance + self.rebalance_frequency).isoformat(),
                'performance_summary': {
                    'recent_efficiency': np.mean([p.get('efficiency', 0) 
                                                for p in self.grid_performance_history[-10:]]) if self.grid_performance_history else 0,
                    'total_updates': len(self.grid_performance_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting grid status: {e}")
            return {}
    
    def optimize_grid_parameters(self) -> Dict[str, float]:
        """
        Optimize grid parameters based on historical performance.
        
        Returns:
            Optimized parameters.
        """
        try:
            if not self.grid_performance_history:
                return {}
            
            # Use grid engine's optimization
            return self.grid_engine.optimize_grid_parameters(self.grid_performance_history)
            
        except Exception as e:
            logger.error(f"Error optimizing grid parameters: {e}")
            return {}
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """
        Get strategy insights and analytics.
        
        Returns:
            Strategy insights.
        """
        try:
            insights = {
                'strategy_type': 'adaptive_grid',
                'current_configuration': {
                    'min_signal_confidence': self.min_signal_confidence,
                    'rebalance_threshold': self.rebalance_threshold,
                    'max_position_ratio': self.max_position_ratio,
                    'volatility_adjustment': self.volatility_adjustment,
                    'regime_adaptation': self.regime_adaptation
                },
                'grid_status': self.get_grid_status(),
                'performance_by_regime': self.regime_performance,
                'recent_signals': [s.to_dict() for s in self.signal_history[-10:]],
                'optimization_suggestions': self.optimize_grid_parameters()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting strategy insights: {e}")
            return {}