"""
Advanced Trading System - Dynamic Risk Manager
Portfolio-level risk oversight with real-time monitoring and adaptive controls.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertType(Enum):
    """Types of risk alerts."""
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    POSITION_LIMIT_BREACH = "POSITION_LIMIT_BREACH"
    VAR_LIMIT_BREACH = "VAR_LIMIT_BREACH"
    LIQUIDITY_WARNING = "LIQUIDITY_WARNING"
    STRATEGY_ERROR = "STRATEGY_ERROR"
    MARKET_STRESS = "MARKET_STRESS"
    FUNDING_RATE_SPIKE = "FUNDING_RATE_SPIKE"

@dataclass
class RiskAlert:
    """Risk alert details."""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    affected_strategies: List[str]
    current_value: float
    threshold_value: float
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""
    # VaR and tail risk
    var_95_1d: float = 0.0  # 1-day 95% VaR
    var_99_1d: float = 0.0  # 1-day 99% VaR
    cvar_95: float = 0.0    # Conditional VaR (Expected Shortfall)
    max_loss_1d: float = 0.0  # Maximum 1-day loss
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown_12m: float = 0.0
    time_underwater_days: float = 0.0
    
    # Concentration and correlation
    strategy_concentration: float = 0.0  # Herfindahl index
    max_strategy_correlation: float = 0.0
    avg_strategy_correlation: float = 0.0
    portfolio_beta: float = 0.0
    
    # Leverage and exposure
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    total_notional: float = 0.0
    
    # Volatility metrics
    realized_volatility_30d: float = 0.0
    forward_volatility_estimate: float = 0.0
    volatility_of_volatility: float = 0.0
    
    # Liquidity metrics
    liquidity_score: float = 1.0  # 1.0 = fully liquid, 0.0 = illiquid
    time_to_liquidate: float = 0.0  # Hours to liquidate 95% of positions
    
    # Market stress indicators
    crypto_fear_greed_index: float = 50.0
    funding_rate_stress: float = 0.0
    basis_stress: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimits:
    """Portfolio-wide risk limits."""
    max_portfolio_var_95: float = 0.02  # 2% daily VaR
    max_portfolio_var_99: float = 0.05  # 5% daily VaR
    max_drawdown: float = 0.10  # 10% maximum drawdown
    max_strategy_correlation: float = 0.75
    max_concentration: float = 0.4  # 40% max allocation to single strategy
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 1.5
    min_liquidity_score: float = 0.7
    max_time_to_liquidate: float = 24.0  # 24 hours
    
    # Emergency thresholds
    emergency_drawdown: float = 0.15  # 15% emergency stop
    emergency_var_multiplier: float = 2.0  # 2x normal VaR triggers emergency

class DynamicRiskManager:
    """
    Advanced portfolio risk management system.
    
    Provides real-time risk monitoring, limit enforcement, and adaptive
    risk controls for a multi-strategy trading portfolio.
    """
    
    def __init__(self, total_capital: float, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize the dynamic risk manager.
        
        Args:
            total_capital: Total portfolio capital
            risk_limits: Custom risk limits (uses defaults if None)
        """
        self.total_capital = total_capital
        self.risk_limits = risk_limits or RiskLimits()
        
        # Risk state
        self.current_risk_level = RiskLevel.LOW
        self.active_alerts: List[RiskAlert] = []
        self.metrics_history: List[PortfolioRiskMetrics] = []
        
        # Risk monitoring
        self.portfolio_metrics = PortfolioRiskMetrics()
        self.returns_history: List[float] = []
        self.nav_history: List[float] = []
        
        # Emergency controls
        self.emergency_stop_triggered = False
        self.circuit_breaker_active = False
        
        # Portfolio-Level Drawdown Control (Feature 1) - Claude's Conservative Parameters
        self.portfolio_high_water_mark = total_capital
        self.drawdown_halt_triggered = False
        self.drawdown_halt_threshold = 0.20  # 20% drawdown threshold (conservative)
        self.recovery_threshold = 0.10  # Resume when drawdown < 10% (conservative)
        
        # Settings
        self.lookback_periods = 252  # ~1 year of daily data
        self.var_confidence_levels = [0.95, 0.99]
        self.stress_test_scenarios = self._initialize_stress_scenarios()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DynamicRiskManager initialized with ${total_capital:,.0f} capital")
    
    def _initialize_stress_scenarios(self) -> List[Dict[str, float]]:
        """Initialize stress test scenarios."""
        return [
            {"name": "2020_covid_crash", "btc_shock": -0.4, "vol_shock": 2.0, "correlation_shock": 0.3},
            {"name": "2022_luna_collapse", "btc_shock": -0.3, "vol_shock": 1.5, "correlation_shock": 0.2},
            {"name": "2018_crypto_winter", "btc_shock": -0.8, "vol_shock": 1.8, "correlation_shock": 0.4},
            {"name": "flash_crash", "btc_shock": -0.2, "vol_shock": 3.0, "correlation_shock": 0.1},
            {"name": "funding_rate_crisis", "funding_shock": 0.01, "vol_shock": 1.2, "correlation_shock": 0.15}
        ]
    
    async def assess_portfolio_risk(self, strategies: Dict[str, Any], portfolio_metrics: Any) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment.
        
        Args:
            strategies: Dictionary of active strategies
            portfolio_metrics: Current portfolio metrics
            
        Returns:
            Dict with risk assessment results
        """
        try:
            # Update portfolio risk metrics
            await self._calculate_portfolio_risk_metrics(strategies, portfolio_metrics)
            
            # Check all risk limits
            risk_violations = await self._check_risk_limits()
            
            # Update risk level
            new_risk_level = self._calculate_overall_risk_level()
            if new_risk_level != self.current_risk_level:
                self.logger.info(f"Risk level changed: {self.current_risk_level.value} -> {new_risk_level.value}")
                self.current_risk_level = new_risk_level
            
            # Generate alerts for violations
            for violation in risk_violations:
                await self._generate_risk_alert(violation)
            
            # Check for emergency conditions
            emergency_needed = await self._check_emergency_conditions()
            
            # Store metrics history
            self.metrics_history.append(self.portfolio_metrics)
            if len(self.metrics_history) > self.lookback_periods:
                self.metrics_history = self.metrics_history[-self.lookback_periods:]
            
            return {
                "risk_level": self.current_risk_level.value,
                "active_alerts": len(self.active_alerts),
                "risk_violations": risk_violations,
                "emergency_needed": emergency_needed,
                "critical_risk": self.current_risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY],
                "portfolio_var_95": self.portfolio_metrics.var_95_1d,
                "portfolio_drawdown": self.portfolio_metrics.current_drawdown,
                "correlation_risk": self.portfolio_metrics.max_strategy_correlation
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            return {"error": str(e), "critical_risk": True}
    
    async def _calculate_portfolio_risk_metrics(self, strategies: Dict[str, Any], portfolio_metrics: Any):
        """Calculate comprehensive portfolio risk metrics."""
        try:
            # Update NAV history
            current_nav = portfolio_metrics.net_asset_value
            self.nav_history.append(current_nav)
            
            # Calculate returns if we have history
            if len(self.nav_history) > 1:
                return_pct = (current_nav - self.nav_history[-2]) / self.nav_history[-2]
                self.returns_history.append(return_pct)
            
            # Trim history for memory management
            if len(self.nav_history) > self.lookback_periods:
                self.nav_history = self.nav_history[-self.lookback_periods:]
                self.returns_history = self.returns_history[-self.lookback_periods:]
            
            # Calculate VaR and tail risk metrics
            if len(self.returns_history) >= 30:
                await self._calculate_var_metrics()
            
            # Calculate drawdown metrics
            await self._calculate_drawdown_metrics()
            
            # Calculate correlation and concentration
            await self._calculate_correlation_metrics(strategies)
            
            # Calculate leverage and exposure
            await self._calculate_leverage_metrics(strategies)
            
            # Calculate volatility metrics
            await self._calculate_volatility_metrics()
            
            # Calculate liquidity metrics
            await self._calculate_liquidity_metrics(strategies)
            
            # Update market stress indicators
            await self._update_market_stress_indicators(strategies)
            
            self.portfolio_metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
    
    async def _calculate_var_metrics(self):
        """Calculate Value at Risk and tail risk metrics."""
        try:
            if len(self.returns_history) < 30:
                return
            
            returns_array = np.array(self.returns_history)
            
            # Historical VaR
            self.portfolio_metrics.var_95_1d = np.percentile(returns_array, 5)  # 5th percentile
            self.portfolio_metrics.var_99_1d = np.percentile(returns_array, 1)  # 1st percentile
            
            # Conditional VaR (Expected Shortfall)
            var_95_threshold = self.portfolio_metrics.var_95_1d
            tail_returns = returns_array[returns_array <= var_95_threshold]
            if len(tail_returns) > 0:
                self.portfolio_metrics.cvar_95 = np.mean(tail_returns)
            
            # Maximum loss
            self.portfolio_metrics.max_loss_1d = np.min(returns_array)
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR metrics: {e}")
    
    async def _calculate_drawdown_metrics(self):
        """Calculate drawdown and underwater metrics."""
        try:
            if len(self.nav_history) < 2:
                return
            
            nav_array = np.array(self.nav_history)
            
            # Calculate running maximum (peak)
            peak = np.maximum.accumulate(nav_array)
            
            # Calculate drawdown
            drawdown = (peak - nav_array) / peak
            
            # Current drawdown
            self.portfolio_metrics.current_drawdown = drawdown[-1]
            
            # Maximum drawdown over period
            self.portfolio_metrics.max_drawdown_12m = np.max(drawdown)
            
            # Time underwater (days since peak)
            current_peak = np.max(nav_array)
            peak_index = np.argmax(nav_array)
            days_since_peak = len(nav_array) - peak_index - 1
            self.portfolio_metrics.time_underwater_days = days_since_peak / 24  # Assuming hourly data
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
    
    async def _calculate_correlation_metrics(self, strategies: Dict[str, Any]):
        """Calculate strategy correlation and concentration metrics."""
        try:
            if len(strategies) < 2:
                self.portfolio_metrics.strategy_concentration = 1.0
                self.portfolio_metrics.max_strategy_correlation = 0.0
                self.portfolio_metrics.avg_strategy_correlation = 0.0
                return
            
            # Calculate concentration (Herfindahl index)
            total_capital = sum(s.capital_allocation for s in strategies.values())
            if total_capital > 0:
                weights_squared = sum((s.capital_allocation / total_capital) ** 2 for s in strategies.values())
                self.portfolio_metrics.strategy_concentration = weights_squared
            
            # Calculate strategy correlations
            strategy_returns = {}
            min_length = float('inf')
            
            for strategy_id, strategy in strategies.items():
                if len(strategy.pnl_history) > 1:
                    returns = np.diff(strategy.pnl_history) / np.array(strategy.pnl_history[:-1])
                    strategy_returns[strategy_id] = returns[-100:]  # Last 100 returns
                    min_length = min(min_length, len(strategy_returns[strategy_id]))
            
            if len(strategy_returns) >= 2 and min_length >= 10:
                # Align return series
                aligned_returns = {}
                for strategy_id, returns in strategy_returns.items():
                    aligned_returns[strategy_id] = returns[-min_length:]
                
                # Calculate correlation matrix
                strategy_ids = list(aligned_returns.keys())
                returns_matrix = np.array([aligned_returns[sid] for sid in strategy_ids])
                
                if returns_matrix.shape[0] >= 2:
                    corr_matrix = np.corrcoef(returns_matrix)
                    
                    # Extract off-diagonal correlations
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    correlations = corr_matrix[mask]
                    
                    self.portfolio_metrics.max_strategy_correlation = np.max(np.abs(correlations))
                    self.portfolio_metrics.avg_strategy_correlation = np.mean(np.abs(correlations))
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation metrics: {e}")
    
    async def _calculate_leverage_metrics(self, strategies: Dict[str, Any]):
        """Calculate leverage and exposure metrics."""
        try:
            total_long_exposure = 0.0
            total_short_exposure = 0.0
            total_notional = 0.0
            
            for strategy in strategies.values():
                for position in strategy.positions.values():
                    if position.net_exposure > 0:
                        total_long_exposure += position.net_exposure
                    else:
                        total_short_exposure += abs(position.net_exposure)
                    
                    total_notional += abs(position.net_exposure)
            
            # Calculate leverage ratios
            if self.total_capital > 0:
                self.portfolio_metrics.gross_leverage = total_notional / self.total_capital
                net_exposure = abs(total_long_exposure - total_short_exposure)
                self.portfolio_metrics.net_leverage = net_exposure / self.total_capital
            
            self.portfolio_metrics.total_notional = total_notional
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage metrics: {e}")
    
    async def _calculate_volatility_metrics(self):
        """Calculate volatility and volatility-of-volatility metrics."""
        try:
            if len(self.returns_history) < 30:
                return
            
            returns_array = np.array(self.returns_history)
            
            # 30-day realized volatility (annualized)
            recent_returns = returns_array[-30:]
            self.portfolio_metrics.realized_volatility_30d = np.std(recent_returns) * np.sqrt(365)
            
            # Rolling volatility for vol-of-vol calculation
            if len(returns_array) >= 60:
                rolling_vols = []
                window = 30
                for i in range(window, len(returns_array)):
                    window_returns = returns_array[i-window:i]
                    rolling_vol = np.std(window_returns)
                    rolling_vols.append(rolling_vol)
                
                if len(rolling_vols) > 1:
                    self.portfolio_metrics.volatility_of_volatility = np.std(rolling_vols)
            
            # Forward volatility estimate (simple EWMA)
            if len(returns_array) >= 10:
                decay_factor = 0.94
                weights = np.array([decay_factor ** i for i in range(len(recent_returns))])
                weights = weights / np.sum(weights)
                
                weighted_variance = np.sum(weights * (recent_returns ** 2))
                self.portfolio_metrics.forward_volatility_estimate = np.sqrt(weighted_variance * 365)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
    
    async def _calculate_liquidity_metrics(self, strategies: Dict[str, Any]):
        """Calculate portfolio liquidity metrics."""
        try:
            # This is a simplified calculation
            # In practice, would need detailed market impact models
            
            total_positions = sum(len(s.positions) for s in strategies.values())
            if total_positions == 0:
                self.portfolio_metrics.liquidity_score = 1.0
                self.portfolio_metrics.time_to_liquidate = 0.0
                return
            
            # Estimate based on position sizes and market depth
            # For crypto markets, assume reasonable liquidity for moderate sizes
            avg_position_size = self.portfolio_metrics.total_notional / max(total_positions, 1)
            
            if avg_position_size < 10000:  # Small positions
                self.portfolio_metrics.liquidity_score = 0.95
                self.portfolio_metrics.time_to_liquidate = 0.5  # 30 minutes
            elif avg_position_size < 50000:  # Medium positions
                self.portfolio_metrics.liquidity_score = 0.85
                self.portfolio_metrics.time_to_liquidate = 2.0  # 2 hours
            elif avg_position_size < 200000:  # Large positions
                self.portfolio_metrics.liquidity_score = 0.75
                self.portfolio_metrics.time_to_liquidate = 8.0  # 8 hours
            else:  # Very large positions
                self.portfolio_metrics.liquidity_score = 0.6
                self.portfolio_metrics.time_to_liquidate = 24.0  # 24 hours
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity metrics: {e}")
    
    async def _update_market_stress_indicators(self, strategies: Dict[str, Any]):
        """Update market stress and regime indicators."""
        try:
            # Calculate funding rate stress
            funding_rates = []
            for strategy in strategies.values():
                specific_metrics = await strategy.get_strategy_specific_metrics()
                current_funding = specific_metrics.get('current_funding_rate', 0.0)
                if current_funding != 0:
                    funding_rates.append(current_funding)
            
            if funding_rates:
                avg_funding = np.mean(funding_rates)
                funding_std = np.std(funding_rates) if len(funding_rates) > 1 else 0.0
                
                # Normalize funding stress (0 = normal, 1 = extreme)
                normal_funding_std = 0.0005  # 0.05% normal std
                self.portfolio_metrics.funding_rate_stress = min(funding_std / normal_funding_std, 1.0)
            
            # Simple fear/greed proxy based on recent volatility
            if self.portfolio_metrics.realized_volatility_30d > 0:
                normal_vol = 0.6  # 60% normal annual vol for crypto
                vol_ratio = self.portfolio_metrics.realized_volatility_30d / normal_vol
                
                # Convert to fear/greed scale (0 = extreme fear, 100 = extreme greed)
                if vol_ratio > 1.5:  # High volatility = fear
                    self.portfolio_metrics.crypto_fear_greed_index = max(20 - (vol_ratio - 1.5) * 10, 0)
                elif vol_ratio < 0.5:  # Low volatility = greed
                    self.portfolio_metrics.crypto_fear_greed_index = min(80 + (0.5 - vol_ratio) * 40, 100)
                else:
                    self.portfolio_metrics.crypto_fear_greed_index = 50  # Neutral
            
        except Exception as e:
            self.logger.error(f"Error updating market stress indicators: {e}")
    
    async def _check_risk_limits(self) -> List[Dict[str, Any]]:
        """Check all risk limits and return violations."""
        violations = []
        
        try:
            # VaR limit checks
            if abs(self.portfolio_metrics.var_95_1d) > self.risk_limits.max_portfolio_var_95:
                violations.append({
                    "type": "var_95_breach",
                    "current": abs(self.portfolio_metrics.var_95_1d),
                    "limit": self.risk_limits.max_portfolio_var_95,
                    "severity": "high"
                })
            
            if abs(self.portfolio_metrics.var_99_1d) > self.risk_limits.max_portfolio_var_99:
                violations.append({
                    "type": "var_99_breach",
                    "current": abs(self.portfolio_metrics.var_99_1d),
                    "limit": self.risk_limits.max_portfolio_var_99,
                    "severity": "critical"
                })
            
            # Drawdown limit check
            if self.portfolio_metrics.current_drawdown > self.risk_limits.max_drawdown:
                violations.append({
                    "type": "drawdown_breach",
                    "current": self.portfolio_metrics.current_drawdown,
                    "limit": self.risk_limits.max_drawdown,
                    "severity": "critical"
                })
            
            # Correlation limit check
            if self.portfolio_metrics.max_strategy_correlation > self.risk_limits.max_strategy_correlation:
                violations.append({
                    "type": "correlation_breach",
                    "current": self.portfolio_metrics.max_strategy_correlation,
                    "limit": self.risk_limits.max_strategy_correlation,
                    "severity": "medium"
                })
            
            # Concentration limit check
            if self.portfolio_metrics.strategy_concentration > self.risk_limits.max_concentration:
                violations.append({
                    "type": "concentration_breach",
                    "current": self.portfolio_metrics.strategy_concentration,
                    "limit": self.risk_limits.max_concentration,
                    "severity": "medium"
                })
            
            # Leverage limit checks
            if self.portfolio_metrics.gross_leverage > self.risk_limits.max_gross_leverage:
                violations.append({
                    "type": "gross_leverage_breach",
                    "current": self.portfolio_metrics.gross_leverage,
                    "limit": self.risk_limits.max_gross_leverage,
                    "severity": "high"
                })
            
            if self.portfolio_metrics.net_leverage > self.risk_limits.max_net_leverage:
                violations.append({
                    "type": "net_leverage_breach",
                    "current": self.portfolio_metrics.net_leverage,
                    "limit": self.risk_limits.max_net_leverage,
                    "severity": "high"
                })
            
            # Liquidity limit checks
            if self.portfolio_metrics.liquidity_score < self.risk_limits.min_liquidity_score:
                violations.append({
                    "type": "liquidity_breach",
                    "current": self.portfolio_metrics.liquidity_score,
                    "limit": self.risk_limits.min_liquidity_score,
                    "severity": "medium"
                })
            
            if self.portfolio_metrics.time_to_liquidate > self.risk_limits.max_time_to_liquidate:
                violations.append({
                    "type": "liquidation_time_breach",
                    "current": self.portfolio_metrics.time_to_liquidate,
                    "limit": self.risk_limits.max_time_to_liquidate,
                    "severity": "medium"
                })
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
        
        return violations
    
    def _calculate_overall_risk_level(self) -> RiskLevel:
        """Calculate overall portfolio risk level."""
        try:
            risk_score = 0
            
            # Drawdown contribution
            if self.portfolio_metrics.current_drawdown > 0.15:
                risk_score += 4  # Emergency
            elif self.portfolio_metrics.current_drawdown > 0.10:
                risk_score += 3  # Critical
            elif self.portfolio_metrics.current_drawdown > 0.05:
                risk_score += 2  # High
            elif self.portfolio_metrics.current_drawdown > 0.02:
                risk_score += 1  # Medium
            
            # VaR contribution
            if abs(self.portfolio_metrics.var_95_1d) > 0.08:
                risk_score += 3
            elif abs(self.portfolio_metrics.var_95_1d) > 0.05:
                risk_score += 2
            elif abs(self.portfolio_metrics.var_95_1d) > 0.03:
                risk_score += 1
            
            # Correlation contribution
            if self.portfolio_metrics.max_strategy_correlation > 0.9:
                risk_score += 2
            elif self.portfolio_metrics.max_strategy_correlation > 0.8:
                risk_score += 1
            
            # Volatility contribution
            if self.portfolio_metrics.realized_volatility_30d > 1.0:
                risk_score += 2
            elif self.portfolio_metrics.realized_volatility_30d > 0.8:
                risk_score += 1
            
            # Market stress contribution
            if self.portfolio_metrics.funding_rate_stress > 0.8:
                risk_score += 1
            
            # Map score to risk level
            if risk_score >= 7:
                return RiskLevel.EMERGENCY
            elif risk_score >= 5:
                return RiskLevel.CRITICAL
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {e}")
            return RiskLevel.HIGH  # Conservative default
    
    async def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met."""
        try:
            # Emergency drawdown threshold
            if self.portfolio_metrics.current_drawdown > self.risk_limits.emergency_drawdown:
                self.logger.critical(f"Emergency drawdown threshold breached: {self.portfolio_metrics.current_drawdown:.1%}")
                return True
            
            # Emergency VaR threshold
            var_threshold = self.risk_limits.max_portfolio_var_99 * self.risk_limits.emergency_var_multiplier
            if abs(self.portfolio_metrics.var_99_1d) > var_threshold:
                self.logger.critical(f"Emergency VaR threshold breached: {abs(self.portfolio_metrics.var_99_1d):.1%}")
                return True
            
            # Check for multiple critical violations
            critical_violations = sum(1 for alert in self.active_alerts 
                                    if alert.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY])
            if critical_violations >= 3:
                self.logger.critical(f"Multiple critical violations: {critical_violations}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
            return True  # Conservative default
    
    def update_portfolio_value(self, current_portfolio_value: float) -> bool:
        """
        Update portfolio value and check drawdown control.
        
        Args:
            current_portfolio_value: Current total portfolio value
            
        Returns:
            True if trading should continue, False if halted due to drawdown
        """
        try:
            # Update high water mark
            if current_portfolio_value > self.portfolio_high_water_mark:
                self.portfolio_high_water_mark = current_portfolio_value
                self.logger.info(f"New portfolio high water mark: ${self.portfolio_high_water_mark:,.2f}")
            
            # Calculate current drawdown
            current_drawdown = (self.portfolio_high_water_mark - current_portfolio_value) / self.portfolio_high_water_mark
            
            # Check if we should halt trading due to drawdown
            if not self.drawdown_halt_triggered and current_drawdown >= self.drawdown_halt_threshold:
                self.drawdown_halt_triggered = True
                self.logger.critical(f"DRAWDOWN HALT TRIGGERED: {current_drawdown:.2%} exceeds threshold {self.drawdown_halt_threshold:.2%}")
                self.logger.critical(f"Portfolio value: ${current_portfolio_value:,.2f}, High water mark: ${self.portfolio_high_water_mark:,.2f}")
                
                # Generate emergency alert
                alert = RiskAlert(
                    alert_type=AlertType.DRAWDOWN_WARNING,
                    risk_level=RiskLevel.EMERGENCY,
                    message=f"Portfolio drawdown halt triggered: {current_drawdown:.2%}",
                    affected_strategies=["ALL"],
                    current_value=current_drawdown,
                    threshold_value=self.drawdown_halt_threshold,
                    recommendation="Halt all trading until drawdown recovers"
                )
                self.active_alerts.append(alert)
                return False
            
            # Check if we should resume trading after recovery
            elif self.drawdown_halt_triggered and current_drawdown <= self.recovery_threshold:
                self.drawdown_halt_triggered = False
                self.logger.info(f"DRAWDOWN RECOVERY: {current_drawdown:.2%} below recovery threshold {self.recovery_threshold:.2%}")
                self.logger.info("Trading resumed after drawdown recovery")
                
                # Acknowledge recovery alert
                for alert in self.active_alerts:
                    if alert.alert_type == AlertType.DRAWDOWN_WARNING and not alert.acknowledged:
                        alert.acknowledged = True
                
                return True
            
            # Return current trading status
            return not self.drawdown_halt_triggered
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
            return False  # Conservative: halt trading on error
    
    def is_trading_halted(self) -> bool:
        """
        Check if trading is currently halted due to risk controls.
        
        Returns:
            True if trading is halted, False otherwise
        """
        return self.drawdown_halt_triggered or self.emergency_stop_triggered or self.circuit_breaker_active
    
    def get_drawdown_status(self) -> Dict[str, Any]:
        """
        Get current drawdown control status.
        
        Returns:
            Dictionary with drawdown status information
        """
        try:
            current_nav = self.nav_history[-1] if self.nav_history else self.total_capital
            current_drawdown = (self.portfolio_high_water_mark - current_nav) / self.portfolio_high_water_mark
            
            return {
                'high_water_mark': self.portfolio_high_water_mark,
                'current_portfolio_value': current_nav,
                'current_drawdown': current_drawdown,
                'drawdown_threshold': self.drawdown_halt_threshold,
                'recovery_threshold': self.recovery_threshold,
                'trading_halted': self.drawdown_halt_triggered,
                'days_since_peak': len([nav for nav in self.nav_history[::-1] if nav < self.portfolio_high_water_mark]),
                'status': 'HALTED' if self.drawdown_halt_triggered else 'ACTIVE'
            }
        except Exception as e:
            self.logger.error(f"Error getting drawdown status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _generate_risk_alert(self, violation: Dict[str, Any]):
        """Generate and store risk alert."""
        try:
            alert_type_map = {
                "var_95_breach": AlertType.VAR_LIMIT_BREACH,
                "var_99_breach": AlertType.VAR_LIMIT_BREACH,
                "drawdown_breach": AlertType.DRAWDOWN_WARNING,
                "correlation_breach": AlertType.CORRELATION_SPIKE,
                "concentration_breach": AlertType.POSITION_LIMIT_BREACH,
                "gross_leverage_breach": AlertType.POSITION_LIMIT_BREACH,
                "net_leverage_breach": AlertType.POSITION_LIMIT_BREACH,
                "liquidity_breach": AlertType.LIQUIDITY_WARNING,
                "liquidation_time_breach": AlertType.LIQUIDITY_WARNING
            }
            
            alert_type = alert_type_map.get(violation["type"], AlertType.POSITION_LIMIT_BREACH)
            
            severity_map = {
                "low": RiskLevel.LOW,
                "medium": RiskLevel.MEDIUM,
                "high": RiskLevel.HIGH,
                "critical": RiskLevel.CRITICAL
            }
            
            risk_level = severity_map.get(violation["severity"], RiskLevel.MEDIUM)
            
            # Generate recommendation
            recommendations = {
                AlertType.VAR_LIMIT_BREACH: "Reduce position sizes or increase hedging",
                AlertType.DRAWDOWN_WARNING: "Consider emergency stop or position reduction",
                AlertType.CORRELATION_SPIKE: "Diversify strategy allocation or pause correlated strategies",
                AlertType.POSITION_LIMIT_BREACH: "Reduce position sizes to comply with limits",
                AlertType.LIQUIDITY_WARNING: "Reduce position sizes in illiquid markets"
            }
            
            alert = RiskAlert(
                alert_type=alert_type,
                risk_level=risk_level,
                message=f"{violation['type']}: {violation['current']:.3f} exceeds limit {violation['limit']:.3f}",
                affected_strategies=[],  # Would need strategy mapping
                current_value=violation["current"],
                threshold_value=violation["limit"],
                recommendation=recommendations.get(alert_type, "Review risk controls")
            )
            
            # Add to active alerts (avoid duplicates)
            existing_alert = next((a for a in self.active_alerts 
                                 if a.alert_type == alert_type and not a.acknowledged), None)
            
            if not existing_alert:
                self.active_alerts.append(alert)
                self.logger.warning(f"Risk alert generated: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error generating risk alert: {e}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data."""
        return {
            "risk_overview": {
                "current_risk_level": self.current_risk_level.value,
                "active_alerts": len(self.active_alerts),
                "emergency_stop_triggered": self.emergency_stop_triggered,
                "circuit_breaker_active": self.circuit_breaker_active
            },
            "portfolio_metrics": {
                "current_drawdown": self.portfolio_metrics.current_drawdown,
                "max_drawdown_12m": self.portfolio_metrics.max_drawdown_12m,
                "var_95_1d": self.portfolio_metrics.var_95_1d,
                "var_99_1d": self.portfolio_metrics.var_99_1d,
                "gross_leverage": self.portfolio_metrics.gross_leverage,
                "net_leverage": self.portfolio_metrics.net_leverage,
                "liquidity_score": self.portfolio_metrics.liquidity_score,
                "correlation_risk": self.portfolio_metrics.max_strategy_correlation
            },
            "market_conditions": {
                "realized_volatility": self.portfolio_metrics.realized_volatility_30d,
                "funding_rate_stress": self.portfolio_metrics.funding_rate_stress,
                "crypto_fear_greed": self.portfolio_metrics.crypto_fear_greed_index
            },
            "active_alerts": [
                {
                    "type": alert.alert_type.value,
                    "level": alert.risk_level.value,
                    "message": alert.message,
                    "recommendation": alert.recommendation,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts if not alert.acknowledged
            ],
            "risk_limits": {
                "max_drawdown": self.risk_limits.max_drawdown,
                "max_var_95": self.risk_limits.max_portfolio_var_95,
                "max_correlation": self.risk_limits.max_strategy_correlation,
                "max_gross_leverage": self.risk_limits.max_gross_leverage
            }
        }
    
    def acknowledge_alert(self, alert_type: AlertType):
        """Acknowledge a risk alert."""
        for alert in self.active_alerts:
            if alert.alert_type == alert_type and not alert.acknowledged:
                alert.acknowledged = True
                self.logger.info(f"Risk alert acknowledged: {alert_type.value}")
                break