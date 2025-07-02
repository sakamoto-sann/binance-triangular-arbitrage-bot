"""
Grid Trading Bot v3.0 - Risk Manager
Advanced risk management with Kelly Criterion, stop losses, and circuit breakers.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .market_analyzer import MarketAnalysis, MarketRegime
from .grid_engine import GridLevel
from ..utils.config_manager import BotConfig
from ..utils.data_manager import OrderData, PositionData

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class RiskEvent(Enum):
    """Risk event types."""
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    CIRCUIT_BREAKER_ACTIVATED = "circuit_breaker_activated"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_SPIKE = "correlation_spike"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_exposure: float
    leverage: float
    current_drawdown: float
    max_drawdown: float
    var_1d: float  # 1-day Value at Risk
    var_5d: float  # 5-day Value at Risk
    sharpe_ratio: float
    sortino_ratio: float
    risk_level: RiskLevel
    last_updated: datetime

@dataclass
class PositionRisk:
    """Individual position risk assessment."""
    symbol: str
    position_size: float
    unrealized_pnl: float
    stop_loss_price: Optional[float]
    risk_amount: float
    risk_percentage: float
    confidence: float
    time_held: timedelta
    risk_level: RiskLevel

@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    event_type: RiskEvent
    severity: RiskLevel
    message: str
    timestamp: datetime
    portfolio_impact: float
    recommended_action: str
    is_active: bool = True

class RiskManager:
    """
    Advanced risk management system with Kelly Criterion and dynamic controls.
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize risk manager.
        
        Args:
            config: Bot configuration.
        """
        self.config = config
        
        # Risk parameters - OPTIMIZED FOR AGGRESSIVE TRADING
        self.max_portfolio_drawdown = config.strategy.max_portfolio_drawdown
        self.circuit_breaker_volatility = config.strategy.circuit_breaker_volatility
        self.kelly_multiplier = config.strategy.kelly_multiplier
        self.max_position_size = config.risk_management.max_position_size
        self.trailing_stop_pct = getattr(config.risk_management, 'trailing_stop_pct', 0.04)
        self.profit_target_pct = getattr(config.risk_management, 'profit_target_pct', 0.08)
        self.max_daily_trades = config.risk_management.max_daily_trades
        self.max_open_orders = config.risk_management.max_open_orders
        self.risk_free_rate = config.risk_management.risk_free_rate
        
        # Tiered position sizing parameters
        self.position_size_high_confidence = getattr(config.risk_management, 'position_size_high_confidence', 0.08)
        self.position_size_medium_confidence = getattr(config.risk_management, 'position_size_medium_confidence', 0.05)
        self.position_size_low_confidence = getattr(config.risk_management, 'position_size_low_confidence', 0.02)
        
        # Advanced risk controls
        self.correlation_limit = getattr(config.risk_management, 'correlation_limit', 0.6)
        self.transaction_cost_rate = getattr(config.risk_management, 'transaction_cost_rate', 0.001)
        
        # State management
        self.portfolio_peak: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.daily_trades: int = 0
        self.last_trade_date: datetime = datetime.now().date()
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.risk_history: List[RiskMetrics] = []
        self.stop_losses: Dict[str, float] = {}  # symbol -> stop_loss_price
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_until: Optional[datetime] = None
        
        # Performance tracking for Kelly Criterion
        self.trade_history: List[Dict[str, Any]] = []
        self.win_rate: float = 0.5
        self.avg_win: float = 0.01
        self.avg_loss: float = 0.01
        
        logger.info("Risk manager initialized")
    
    def assess_portfolio_risk(self, portfolio_value: float, positions: List[PositionData],
                            open_orders: List[OrderData], market_analysis: MarketAnalysis) -> RiskMetrics:
        """
        Assess overall portfolio risk.
        
        Args:
            portfolio_value: Current portfolio value.
            positions: Current positions.
            open_orders: Open orders.
            market_analysis: Current market analysis.
            
        Returns:
            Comprehensive risk metrics.
        """
        try:
            # Update portfolio peak and drawdown
            if portfolio_value > self.portfolio_peak:
                self.portfolio_peak = portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Calculate total exposure
            total_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in positions)
            
            # Calculate unrealized P&L
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            realized_pnl = sum(pos.realized_pnl for pos in positions)
            
            # Calculate leverage
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate VaR (Value at Risk)
            var_1d, var_5d = self._calculate_var(positions, market_analysis)
            
            # Calculate Sharpe and Sortino ratios
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            
            # Determine risk level
            risk_level = self._assess_risk_level(
                self.current_drawdown, leverage, market_analysis.volatility_metrics.volatility_regime
            )
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_exposure=total_exposure,
                leverage=leverage,
                current_drawdown=self.current_drawdown,
                max_drawdown=self.max_drawdown,
                var_1d=var_1d,
                var_5d=var_5d,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                risk_level=risk_level,
                last_updated=datetime.now()
            )
            
            # Update risk history
            self.risk_history.append(risk_metrics)
            if len(self.risk_history) > 1000:  # Keep last 1000 records
                self.risk_history = self.risk_history[-1000:]
            
            # Check for risk alerts
            self._check_risk_alerts(risk_metrics, market_analysis)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return self._create_default_risk_metrics(portfolio_value)
    
    def calculate_kelly_position_size(self, signal_strength: float, 
                                    current_price: float, volatility: float,
                                    available_capital: float, confidence: float = 0.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion with tiered confidence sizing.
        
        Args:
            signal_strength: Signal strength (-1 to 1).
            current_price: Current market price.
            volatility: Current volatility.
            available_capital: Available capital for trading.
            confidence: Signal confidence (0.0 to 1.0).
            
        Returns:
            Optimal position size.
        """
        try:
            # TIERED POSITION SIZING BASED ON CONFIDENCE
            if confidence >= 0.8:
                # High confidence: Use larger position size
                base_position_size = self.position_size_high_confidence
            elif confidence >= 0.6:
                # Medium confidence: Use medium position size
                base_position_size = self.position_size_medium_confidence
            else:
                # Low confidence: Use smaller position size
                base_position_size = self.position_size_low_confidence
            
            # Update trading statistics
            self._update_kelly_statistics()
            
            # Kelly formula: f = (bp - q) / b
            # where:
            # f = fraction of capital to wager
            # b = odds of winning (avg_win)
            # p = probability of winning (win_rate)
            # q = probability of losing (1 - win_rate)
            
            if self.avg_win <= 0 or self.win_rate <= 0:
                return base_position_size * available_capital  # Use base sizing if no Kelly data
            
            # Calculate Kelly fraction
            kelly_fraction = ((self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)) / self.avg_win
            
            # Apply Kelly multiplier (typically 0.25 for conservative approach)
            kelly_fraction *= self.kelly_multiplier
            
            # Adjust for signal strength
            kelly_fraction *= abs(signal_strength)
            
            # Adjust for volatility (reduce size in high volatility)
            volatility_adjustment = max(0.3, 1.0 - (volatility * 10))
            kelly_fraction *= volatility_adjustment
            
            # Calculate position size
            position_value = available_capital * kelly_fraction
            position_size = position_value / current_price if current_price > 0 else 0
            
            # Apply maximum position size limit
            position_size = min(position_size, self.max_position_size)
            
            # Ensure minimum meaningful size
            min_size = 0.0001  # Minimum position size
            if position_size < min_size:
                position_size = 0.0
            
            logger.debug(f"Kelly position size calculated: {position_size:.6f} "
                        f"(kelly_fraction: {kelly_fraction:.4f}, signal: {signal_strength:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return self.config.trading.base_position_size * 0.5
    
    def check_position_limits(self, new_position_size: float, symbol: str,
                            current_positions: List[PositionData]) -> Tuple[bool, str]:
        """
        Check if new position exceeds risk limits.
        
        Args:
            new_position_size: Proposed new position size.
            symbol: Trading symbol.
            current_positions: Current positions.
            
        Returns:
            Tuple of (is_allowed, reason).
        """
        try:
            # Check individual position size limit
            if abs(new_position_size) > self.max_position_size:
                return False, f"Position size {new_position_size:.6f} exceeds maximum {self.max_position_size:.6f}"
            
            # Check total exposure
            current_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in current_positions)
            new_exposure = current_exposure + abs(new_position_size * self._get_current_price(symbol))
            
            # Rough exposure limit (could be configurable)
            max_exposure = self.portfolio_peak * 2.0  # 2x portfolio value
            if new_exposure > max_exposure:
                return False, f"Total exposure {new_exposure:.2f} would exceed limit {max_exposure:.2f}"
            
            # Check daily trade limit
            if self._is_new_trade_day():
                self.daily_trades = 0
                self.last_trade_date = datetime.now().date()
            
            if self.daily_trades >= self.max_daily_trades:
                return False, f"Daily trade limit {self.max_daily_trades} exceeded"
            
            # Check circuit breaker
            if self.circuit_breaker_active:
                if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
                    return False, f"Circuit breaker active until {self.circuit_breaker_until}"
                else:
                    self._deactivate_circuit_breaker()
            
            return True, "Position allowed"
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False, f"Error checking limits: {e}"
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                          volatility: float, confidence: float) -> float:
        """
        Calculate dynamic stop loss price.
        
        Args:
            entry_price: Entry price of position.
            side: Position side ('buy' or 'sell').
            volatility: Current market volatility.
            confidence: Confidence in the position.
            
        Returns:
            Stop loss price.
        """
        try:
            # Base stop loss percentage
            base_stop_pct = self.stop_loss_pct
            
            # Adjust based on volatility
            volatility_multiplier = max(0.5, min(3.0, 1.0 + volatility * 10))
            
            # Adjust based on confidence (higher confidence = tighter stop)
            confidence_multiplier = max(0.7, 2.0 - confidence)
            
            # Calculate adjusted stop percentage
            stop_pct = base_stop_pct * volatility_multiplier * confidence_multiplier
            
            # Calculate stop price
            if side.lower() == 'buy':
                stop_price = entry_price * (1.0 - stop_pct)
            else:  # sell
                stop_price = entry_price * (1.0 + stop_pct)
            
            logger.debug(f"Stop loss calculated: {stop_price:.2f} for {side} at {entry_price:.2f} "
                        f"(stop_pct: {stop_pct:.3f})")
            
            return stop_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Fallback to basic stop loss
            if side.lower() == 'buy':
                return entry_price * (1.0 - self.stop_loss_pct)
            else:
                return entry_price * (1.0 + self.stop_loss_pct)
    
    def check_stop_losses(self, positions: List[PositionData], 
                         current_price: float) -> List[Dict[str, Any]]:
        """
        Check if any positions should trigger stop losses.
        
        Args:
            positions: Current positions.
            current_price: Current market price.
            
        Returns:
            List of positions that should be closed.
        """
        try:
            stop_loss_triggers = []
            
            for position in positions:
                symbol = position.symbol
                
                if symbol in self.stop_losses:
                    stop_price = self.stop_losses[symbol]
                    
                    # Determine if stop should trigger
                    should_trigger = False
                    
                    if position.quantity > 0:  # Long position
                        if current_price <= stop_price:
                            should_trigger = True
                    else:  # Short position
                        if current_price >= stop_price:
                            should_trigger = True
                    
                    if should_trigger:
                        stop_loss_triggers.append({
                            'symbol': symbol,
                            'position': position,
                            'stop_price': stop_price,
                            'current_price': current_price,
                            'reason': 'stop_loss_triggered'
                        })
                        
                        # Create risk alert
                        self._create_risk_alert(
                            RiskEvent.STOP_LOSS_TRIGGERED,
                            RiskLevel.HIGH,
                            f"Stop loss triggered for {symbol} at {current_price:.2f}",
                            abs(position.unrealized_pnl)
                        )
            
            return stop_loss_triggers
            
        except Exception as e:
            logger.error(f"Error checking stop losses: {e}")
            return []
    
    def check_circuit_breakers(self, market_analysis: MarketAnalysis, 
                             portfolio_metrics: RiskMetrics) -> bool:
        """
        Check if circuit breakers should be activated.
        
        Args:
            market_analysis: Current market analysis.
            portfolio_metrics: Current portfolio metrics.
            
        Returns:
            Whether circuit breaker was activated.
        """
        try:
            should_activate = False
            reason = ""
            
            # Check portfolio drawdown
            if portfolio_metrics.current_drawdown > self.max_portfolio_drawdown:
                should_activate = True
                reason = f"Portfolio drawdown {portfolio_metrics.current_drawdown:.2%} exceeds limit {self.max_portfolio_drawdown:.2%}"
            
            # Check volatility spike
            elif market_analysis.volatility_metrics.current_volatility > self.circuit_breaker_volatility:
                should_activate = True
                reason = f"Volatility {market_analysis.volatility_metrics.current_volatility:.2%} exceeds limit {self.circuit_breaker_volatility:.2%}"
            
            # Check extreme market conditions
            elif market_analysis.volatility_metrics.volatility_regime == "extreme":
                should_activate = True
                reason = "Extreme market volatility detected"
            
            # Check leverage
            elif portfolio_metrics.leverage > 3.0:  # Configurable leverage limit
                should_activate = True
                reason = f"Leverage {portfolio_metrics.leverage:.2f} exceeds safe limit"
            
            if should_activate and not self.circuit_breaker_active:
                self._activate_circuit_breaker(reason)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return False
    
    def _calculate_var(self, positions: List[PositionData], 
                      market_analysis: MarketAnalysis) -> Tuple[float, float]:
        """
        Calculate Value at Risk for 1-day and 5-day periods.
        
        Args:
            positions: Current positions.
            market_analysis: Current market analysis.
            
        Returns:
            Tuple of (var_1d, var_5d).
        """
        try:
            if not positions:
                return 0.0, 0.0
            
            # Simple VaR calculation using volatility
            # In production, this would use historical simulation or Monte Carlo
            
            total_value = sum(abs(pos.quantity * pos.avg_price) for pos in positions)
            current_volatility = market_analysis.volatility_metrics.current_volatility
            
            # 95% confidence level (1.645 standard deviations)
            confidence_level = 1.645
            
            # 1-day VaR
            var_1d = total_value * current_volatility * confidence_level / np.sqrt(365)
            
            # 5-day VaR (assuming independence)
            var_5d = total_value * current_volatility * confidence_level * np.sqrt(5) / np.sqrt(365)
            
            return var_1d, var_5d
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0, 0.0
    
    def _calculate_sharpe_ratio(self, lookback_days: int = 30) -> float:
        """
        Calculate Sharpe ratio based on recent performance.
        
        Args:
            lookback_days: Number of days to look back.
            
        Returns:
            Sharpe ratio.
        """
        try:
            if len(self.risk_history) < 2:
                return 0.0
            
            # Get recent risk metrics
            recent_metrics = self.risk_history[-min(lookback_days, len(self.risk_history)):]
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(recent_metrics)):
                prev_value = recent_metrics[i-1].portfolio_value
                curr_value = recent_metrics[i].portfolio_value
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate Sharpe ratio
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_daily = self.risk_free_rate / 365
            
            if std_return > 0:
                sharpe_ratio = (avg_return - risk_free_daily) / std_return * np.sqrt(365)
            else:
                sharpe_ratio = 0.0
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, lookback_days: int = 30) -> float:
        """
        Calculate Sortino ratio based on recent performance.
        
        Args:
            lookback_days: Number of days to look back.
            
        Returns:
            Sortino ratio.
        """
        try:
            if len(self.risk_history) < 2:
                return 0.0
            
            # Get recent risk metrics
            recent_metrics = self.risk_history[-min(lookback_days, len(self.risk_history)):]
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(recent_metrics)):
                prev_value = recent_metrics[i-1].portfolio_value
                curr_value = recent_metrics[i].portfolio_value
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < 0]
            
            if len(negative_returns) == 0:
                return float('inf')  # No negative returns
            
            downside_deviation = np.std(negative_returns)
            avg_return = np.mean(returns)
            risk_free_daily = self.risk_free_rate / 365
            
            if downside_deviation > 0:
                sortino_ratio = (avg_return - risk_free_daily) / downside_deviation * np.sqrt(365)
            else:
                sortino_ratio = 0.0
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _assess_risk_level(self, drawdown: float, leverage: float, 
                          volatility_regime: str) -> RiskLevel:
        """
        Assess overall risk level.
        
        Args:
            drawdown: Current drawdown.
            leverage: Current leverage.
            volatility_regime: Current volatility regime.
            
        Returns:
            Risk level assessment.
        """
        try:
            risk_score = 0
            
            # Drawdown component
            if drawdown > 0.15:  # 15%
                risk_score += 3
            elif drawdown > 0.10:  # 10%
                risk_score += 2
            elif drawdown > 0.05:  # 5%
                risk_score += 1
            
            # Leverage component
            if leverage > 2.0:
                risk_score += 3
            elif leverage > 1.5:
                risk_score += 2
            elif leverage > 1.0:
                risk_score += 1
            
            # Volatility component
            if volatility_regime == "extreme":
                risk_score += 3
            elif volatility_regime == "high":
                risk_score += 2
            elif volatility_regime == "normal":
                risk_score += 1
            
            # Map score to risk level
            if risk_score >= 7:
                return RiskLevel.EXTREME
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _update_kelly_statistics(self) -> None:
        """Update Kelly Criterion statistics from trade history."""
        try:
            if len(self.trade_history) < 10:  # Need minimum trades
                return
            
            recent_trades = self.trade_history[-100:]  # Last 100 trades
            
            wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
            
            if len(recent_trades) > 0:
                self.win_rate = len(wins) / len(recent_trades)
            
            if len(wins) > 0:
                self.avg_win = np.mean([t['pnl'] for t in wins])
            
            if len(losses) > 0:
                self.avg_loss = abs(np.mean([t['pnl'] for t in losses]))
            
        except Exception as e:
            logger.error(f"Error updating Kelly statistics: {e}")
    
    def _check_risk_alerts(self, risk_metrics: RiskMetrics, 
                          market_analysis: MarketAnalysis) -> None:
        """
        Check for and create risk alerts.
        
        Args:
            risk_metrics: Current risk metrics.
            market_analysis: Current market analysis.
        """
        try:
            # Check drawdown alert
            if risk_metrics.current_drawdown > self.max_portfolio_drawdown * 0.8:  # 80% of limit
                self._create_risk_alert(
                    RiskEvent.DRAWDOWN_LIMIT_EXCEEDED,
                    RiskLevel.HIGH,
                    f"Drawdown approaching limit: {risk_metrics.current_drawdown:.2%}",
                    risk_metrics.unrealized_pnl
                )
            
            # Check volatility alert
            if market_analysis.volatility_metrics.volatility_regime == "extreme":
                self._create_risk_alert(
                    RiskEvent.VOLATILITY_SPIKE,
                    RiskLevel.HIGH,
                    f"Extreme volatility detected: {market_analysis.volatility_metrics.current_volatility:.2%}",
                    0
                )
            
            # Clean up old alerts
            self._cleanup_old_alerts()
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    def _create_risk_alert(self, event_type: RiskEvent, severity: RiskLevel,
                          message: str, portfolio_impact: float) -> None:
        """
        Create a new risk alert.
        
        Args:
            event_type: Type of risk event.
            severity: Alert severity.
            message: Alert message.
            portfolio_impact: Impact on portfolio.
        """
        try:
            alert = RiskAlert(
                alert_id=f"{event_type.value}_{int(datetime.now().timestamp())}",
                event_type=event_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                portfolio_impact=portfolio_impact,
                recommended_action=self._get_recommended_action(event_type, severity),
                is_active=True
            )
            
            self.risk_alerts.append(alert)
            
            logger.warning(f"Risk alert created: {event_type.value} - {message}")
            
        except Exception as e:
            logger.error(f"Error creating risk alert: {e}")
    
    def _get_recommended_action(self, event_type: RiskEvent, severity: RiskLevel) -> str:
        """
        Get recommended action for risk event.
        
        Args:
            event_type: Type of risk event.
            severity: Event severity.
            
        Returns:
            Recommended action.
        """
        action_map = {
            RiskEvent.STOP_LOSS_TRIGGERED: "Close position immediately",
            RiskEvent.CIRCUIT_BREAKER_ACTIVATED: "Halt all trading",
            RiskEvent.POSITION_LIMIT_EXCEEDED: "Reduce position size",
            RiskEvent.DRAWDOWN_LIMIT_EXCEEDED: "Reduce risk exposure",
            RiskEvent.VOLATILITY_SPIKE: "Widen stop losses and reduce position sizes",
            RiskEvent.CORRELATION_SPIKE: "Diversify positions"
        }
        
        base_action = action_map.get(event_type, "Monitor closely")
        
        if severity == RiskLevel.EXTREME:
            return f"URGENT: {base_action}"
        elif severity == RiskLevel.HIGH:
            return f"HIGH PRIORITY: {base_action}"
        else:
            return base_action
    
    def _activate_circuit_breaker(self, reason: str) -> None:
        """
        Activate circuit breaker.
        
        Args:
            reason: Reason for activation.
        """
        try:
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(hours=1)  # 1 hour cooldown
            
            self._create_risk_alert(
                RiskEvent.CIRCUIT_BREAKER_ACTIVATED,
                RiskLevel.EXTREME,
                f"Circuit breaker activated: {reason}",
                0
            )
            
            logger.warning(f"Circuit breaker activated: {reason}")
            
        except Exception as e:
            logger.error(f"Error activating circuit breaker: {e}")
    
    def _deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker."""
        try:
            self.circuit_breaker_active = False
            self.circuit_breaker_until = None
            
            logger.info("Circuit breaker deactivated")
            
        except Exception as e:
            logger.error(f"Error deactivating circuit breaker: {e}")
    
    def _cleanup_old_alerts(self, max_age_hours: int = 24) -> None:
        """
        Clean up old risk alerts.
        
        Args:
            max_age_hours: Maximum age of alerts to keep.
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            self.risk_alerts = [alert for alert in self.risk_alerts 
                              if alert.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    def _is_new_trade_day(self) -> bool:
        """Check if it's a new trading day."""
        return datetime.now().date() > self.last_trade_date
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Current price (placeholder implementation).
        """
        # This would integrate with price feed in production
        return 50000.0  # Placeholder price
    
    def _create_default_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """
        Create default risk metrics for error cases.
        
        Args:
            portfolio_value: Current portfolio value.
            
        Returns:
            Default risk metrics.
        """
        return RiskMetrics(
            portfolio_value=portfolio_value,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_exposure=0.0,
            leverage=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            var_1d=0.0,
            var_5d=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            risk_level=RiskLevel.LOW,
            last_updated=datetime.now()
        )
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """
        Get currently active risk alerts.
        
        Returns:
            List of active alerts.
        """
        return [alert for alert in self.risk_alerts if alert.is_active]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a risk alert.
        
        Args:
            alert_id: ID of alert to acknowledge.
            
        Returns:
            Success status.
        """
        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.is_active = False
                    logger.info(f"Risk alert acknowledged: {alert_id}")
                    return True
            
            logger.warning(f"Risk alert not found: {alert_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def add_trade_to_history(self, trade_data: Dict[str, Any]) -> None:
        """
        Add completed trade to history for Kelly Criterion calculations.
        
        Args:
            trade_data: Trade data including pnl.
        """
        try:
            self.trade_history.append(trade_data)
            
            # Keep only recent trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            # Update daily trade count
            if self._is_new_trade_day():
                self.daily_trades = 1
                self.last_trade_date = datetime.now().date()
            else:
                self.daily_trades += 1
            
        except Exception as e:
            logger.error(f"Error adding trade to history: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            Risk summary dictionary.
        """
        try:
            if not self.risk_history:
                return {}
            
            latest_metrics = self.risk_history[-1]
            
            return {
                'current_risk_level': latest_metrics.risk_level.value,
                'portfolio_value': latest_metrics.portfolio_value,
                'current_drawdown': latest_metrics.current_drawdown,
                'max_drawdown': latest_metrics.max_drawdown,
                'leverage': latest_metrics.leverage,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'var_1d': latest_metrics.var_1d,
                'active_alerts': len(self.get_active_alerts()),
                'circuit_breaker_active': self.circuit_breaker_active,
                'daily_trades': self.daily_trades,
                'win_rate': self.win_rate,
                'kelly_stats': {
                    'win_rate': self.win_rate,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {}