"""
Grid Trading Bot v3.0 - Base Strategy
Abstract base class for all trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd

from ..core.market_analyzer import MarketAnalysis
from ..core.grid_engine import GridLevel
from ..core.risk_manager import RiskMetrics
from ..utils.config_manager import BotConfig

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_ALL = "close_all"
    REBALANCE = "rebalance"

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class StrategySignal:
    """Trading strategy signal."""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1 scale
    symbol: str
    price: float
    quantity: float
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'symbol': self.symbol,
            'price': self.price,
            'quantity': self.quantity,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    name: str
    version: str
    enabled: bool
    parameters: Dict[str, Any]
    risk_parameters: Dict[str, float]
    performance_targets: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create strategy config from dictionary."""
        return cls(
            name=data.get('name', 'unknown'),
            version=data.get('version', '1.0.0'),
            enabled=data.get('enabled', True),
            parameters=data.get('parameters', {}),
            risk_parameters=data.get('risk_parameters', {}),
            performance_targets=data.get('performance_targets', {})
        )

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, config: BotConfig, strategy_config: StrategyConfig):
        """
        Initialize base strategy.
        
        Args:
            config: Bot configuration.
            strategy_config: Strategy-specific configuration.
        """
        self.config = config
        self.strategy_config = strategy_config
        
        # Strategy state
        self.is_active: bool = strategy_config.enabled
        self.last_signal: Optional[StrategySignal] = None
        self.signal_history: List[StrategySignal] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Market state
        self.last_market_analysis: Optional[MarketAnalysis] = None
        self.last_risk_metrics: Optional[RiskMetrics] = None
        self.current_price: float = 0.0
        
        # Strategy metrics
        self.total_signals: int = 0
        self.successful_signals: int = 0
        self.strategy_pnl: float = 0.0
        self.max_strategy_drawdown: float = 0.0
        
        logger.info(f"Strategy initialized: {strategy_config.name} v{strategy_config.version}")
    
    @abstractmethod
    def generate_signal(self, price_data: pd.DataFrame, 
                       market_analysis: MarketAnalysis,
                       risk_metrics: RiskMetrics,
                       current_positions: List[Any]) -> Optional[StrategySignal]:
        """
        Generate trading signal based on market conditions.
        
        Args:
            price_data: Historical price data.
            market_analysis: Current market analysis.
            risk_metrics: Current risk metrics.
            current_positions: Current portfolio positions.
            
        Returns:
            Trading signal or None if no signal.
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal,
                              available_capital: float,
                              risk_metrics: RiskMetrics) -> float:
        """
        Calculate optimal position size for signal.
        
        Args:
            signal: Trading signal.
            available_capital: Available trading capital.
            risk_metrics: Current risk metrics.
            
        Returns:
            Optimal position size.
        """
        pass
    
    @abstractmethod
    def should_rebalance(self, market_analysis: MarketAnalysis,
                        performance_metrics: Dict[str, Any]) -> bool:
        """
        Determine if strategy should rebalance.
        
        Args:
            market_analysis: Current market analysis.
            performance_metrics: Current performance metrics.
            
        Returns:
            Whether to rebalance.
        """
        pass
    
    def update_market_state(self, price_data: pd.DataFrame,
                          market_analysis: MarketAnalysis,
                          risk_metrics: RiskMetrics) -> None:
        """
        Update strategy's market state.
        
        Args:
            price_data: Latest price data.
            market_analysis: Current market analysis.
            risk_metrics: Current risk metrics.
        """
        try:
            self.last_market_analysis = market_analysis
            self.last_risk_metrics = risk_metrics
            
            if len(price_data) > 0:
                self.current_price = price_data.iloc[-1]['close']
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
    
    def record_signal(self, signal: StrategySignal) -> None:
        """
        Record a generated signal.
        
        Args:
            signal: Trading signal to record.
        """
        try:
            self.last_signal = signal
            self.signal_history.append(signal)
            self.total_signals += 1
            
            # Keep limited history
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            logger.debug(f"Signal recorded: {signal.signal_type.value} {signal.symbol} @ {signal.price}")
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
    
    def record_trade_outcome(self, signal_id: str, pnl: float, success: bool) -> None:
        """
        Record outcome of a trade based on strategy signal.
        
        Args:
            signal_id: ID of the signal that generated the trade.
            pnl: Profit/loss of the trade.
            success: Whether the trade was successful.
        """
        try:
            self.strategy_pnl += pnl
            
            if success:
                self.successful_signals += 1
            
            # Update strategy drawdown
            if pnl < 0:
                current_dd = abs(pnl) / max(abs(self.strategy_pnl), 1000)  # Relative to strategy capital
                self.max_strategy_drawdown = max(self.max_strategy_drawdown, current_dd)
            
            logger.debug(f"Trade outcome recorded: P&L ${pnl:.2f}, Success: {success}")
            
        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Strategy performance dictionary.
        """
        try:
            win_rate = self.successful_signals / self.total_signals if self.total_signals > 0 else 0
            
            return {
                'strategy_name': self.strategy_config.name,
                'total_signals': self.total_signals,
                'successful_signals': self.successful_signals,
                'win_rate': win_rate,
                'strategy_pnl': self.strategy_pnl,
                'max_drawdown': self.max_strategy_drawdown,
                'is_active': self.is_active,
                'last_signal_time': self.last_signal.timestamp if self.last_signal else None,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def validate_signal(self, signal: StrategySignal) -> Tuple[bool, str]:
        """
        Validate a trading signal before execution.
        
        Args:
            signal: Signal to validate.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        try:
            # Basic validation
            if signal.confidence < 0 or signal.confidence > 1:
                return False, "Invalid confidence level"
            
            if signal.price <= 0:
                return False, "Invalid price"
            
            if signal.quantity <= 0:
                return False, "Invalid quantity"
            
            if not signal.symbol:
                return False, "Missing symbol"
            
            # Strategy-specific validation
            if not self.is_active:
                return False, "Strategy is not active"
            
            # Risk validation
            if self.last_risk_metrics:
                if self.last_risk_metrics.current_drawdown > 0.2:  # 20% drawdown limit
                    return False, "Portfolio drawdown too high"
            
            return True, "Signal is valid"
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Validation error: {e}"
    
    def _update_performance_metrics(self) -> None:
        """Update internal performance metrics."""
        try:
            if self.last_market_analysis and self.last_risk_metrics:
                self.performance_metrics.update({
                    'current_regime': self.last_market_analysis.regime.value,
                    'regime_confidence': self.last_market_analysis.confidence,
                    'portfolio_drawdown': self.last_risk_metrics.current_drawdown,
                    'risk_level': self.last_risk_metrics.risk_level.value,
                    'last_update': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def reset_strategy_state(self) -> None:
        """Reset strategy to initial state."""
        try:
            self.signal_history.clear()
            self.last_signal = None
            self.total_signals = 0
            self.successful_signals = 0
            self.strategy_pnl = 0.0
            self.max_strategy_drawdown = 0.0
            self.performance_metrics.clear()
            
            logger.info(f"Strategy state reset: {self.strategy_config.name}")
            
        except Exception as e:
            logger.error(f"Error resetting strategy state: {e}")
    
    def activate_strategy(self) -> None:
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Strategy activated: {self.strategy_config.name}")
    
    def deactivate_strategy(self) -> None:
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Strategy deactivated: {self.strategy_config.name}")
    
    def update_configuration(self, new_config: StrategyConfig) -> None:
        """
        Update strategy configuration.
        
        Args:
            new_config: New strategy configuration.
        """
        try:
            old_name = self.strategy_config.name
            self.strategy_config = new_config
            self.is_active = new_config.enabled
            
            logger.info(f"Strategy configuration updated: {old_name} -> {new_config.name}")
            
        except Exception as e:
            logger.error(f"Error updating strategy configuration: {e}")
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get signal generation statistics.
        
        Returns:
            Signal statistics.
        """
        try:
            if not self.signal_history:
                return {}
            
            # Count signals by type
            signal_counts = {}
            confidence_sum = 0
            
            for signal in self.signal_history:
                signal_type = signal.signal_type.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                confidence_sum += signal.confidence
            
            avg_confidence = confidence_sum / len(self.signal_history)
            
            # Recent signals (last 24 hours)
            recent_cutoff = datetime.now() - pd.Timedelta(hours=24)
            recent_signals = [s for s in self.signal_history if s.timestamp > recent_cutoff]
            
            return {
                'total_signals': len(self.signal_history),
                'signal_distribution': signal_counts,
                'average_confidence': avg_confidence,
                'recent_signals_24h': len(recent_signals),
                'last_signal_time': self.signal_history[-1].timestamp.isoformat(),
                'win_rate': self.successful_signals / self.total_signals if self.total_signals > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}
    
    def export_strategy_data(self) -> Dict[str, Any]:
        """
        Export strategy data for analysis.
        
        Returns:
            Complete strategy data.
        """
        try:
            return {
                'strategy_config': {
                    'name': self.strategy_config.name,
                    'version': self.strategy_config.version,
                    'enabled': self.strategy_config.enabled,
                    'parameters': self.strategy_config.parameters
                },
                'performance': self.get_strategy_performance(),
                'signal_statistics': self.get_signal_statistics(),
                'signal_history': [signal.to_dict() for signal in self.signal_history[-100:]],  # Last 100 signals
                'current_state': {
                    'is_active': self.is_active,
                    'current_price': self.current_price,
                    'last_update': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error exporting strategy data: {e}")
            return {}