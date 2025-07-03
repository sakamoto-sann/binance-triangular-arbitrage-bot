"""
Advanced Trading System - Base Strategy Interface
Abstract base class for all trading strategies in the portfolio system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    """Strategy execution status."""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class StrategyType(Enum):
    """Types of trading strategies."""
    DELTA_NEUTRAL_GRID = "DELTA_NEUTRAL_GRID"
    FUNDING_ARBITRAGE = "FUNDING_ARBITRAGE"
    BASIS_TRADING = "BASIS_TRADING"
    VOLATILITY_HARVESTING = "VOLATILITY_HARVESTING"
    CROSS_EXCHANGE_ARB = "CROSS_EXCHANGE_ARB"
    LIQUIDITY_MINING = "LIQUIDITY_MINING"

@dataclass
class StrategyMetrics:
    """Real-time strategy performance metrics."""
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    active_positions: int = 0
    capital_utilization: float = 0.0
    last_updated: datetime = None

@dataclass
class PositionExposure:
    """Current position exposure details."""
    symbol: str
    net_exposure: float  # Net USD exposure
    long_exposure: float
    short_exposure: float
    delta_exposure: float
    gamma_exposure: float = 0.0
    vega_exposure: float = 0.0
    theta_exposure: float = 0.0

@dataclass
class RiskLimits:
    """Risk management limits for strategies."""
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    max_leverage: float
    max_correlation: float = 0.7
    var_limit_95: float = 0.02  # 2% VaR limit
    var_limit_99: float = 0.05  # 5% VaR limit

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies in the portfolio system must inherit from this class
    and implement the required abstract methods. This ensures consistent
    interfaces for the PortfolioManager to orchestrate multiple strategies.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 strategy_type: StrategyType,
                 capital_allocation: float,
                 risk_limits: RiskLimits,
                 config: Dict[str, Any]):
        """
        Initialize base strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            strategy_type: Type of strategy (from StrategyType enum)
            capital_allocation: Amount of capital allocated to this strategy
            risk_limits: Risk management limits
            config: Strategy-specific configuration parameters
        """
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.capital_allocation = capital_allocation
        self.risk_limits = risk_limits
        self.config = config
        
        # Strategy state
        self.status = StrategyStatus.INACTIVE
        self.is_emergency_stopped = False
        self.last_execution_time = None
        self.error_count = 0
        self.max_consecutive_errors = 5
        
        # Performance tracking
        self.metrics = StrategyMetrics()
        self.pnl_history: List[float] = []
        self.position_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Current positions
        self.positions: Dict[str, PositionExposure] = {}
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
        
        self.logger.info(f"Strategy initialized: {strategy_id} ({strategy_type.value})")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the strategy. Called once before starting execution.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute_cycle(self, market_data: Dict[str, Any]) -> bool:
        """
        Execute one cycle of the strategy logic.
        
        Args:
            market_data: Current market data including prices, order books, etc.
            
        Returns:
            bool: True if execution successful, False if error occurred
        """
        pass
    
    @abstractmethod
    async def calculate_positions(self) -> Dict[str, PositionExposure]:
        """
        Calculate current position exposures.
        
        Returns:
            Dict mapping symbol to PositionExposure
        """
        pass
    
    @abstractmethod
    async def calculate_pnl(self) -> Tuple[float, float, float]:
        """
        Calculate current P&L.
        
        Returns:
            Tuple of (total_pnl, realized_pnl, unrealized_pnl)
        """
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """
        Emergency stop - close all positions and halt strategy.
        
        Returns:
            bool: True if emergency stop successful
        """
        pass
    
    @abstractmethod
    async def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """
        Get strategy-specific metrics beyond standard ones.
        
        Returns:
            Dict of strategy-specific metrics
        """
        pass
    
    # Common implementation methods
    
    async def start(self) -> bool:
        """Start the strategy execution."""
        try:
            if await self.initialize():
                self.status = StrategyStatus.ACTIVE
                self.logger.info(f"Strategy {self.strategy_id} started successfully")
                return True
            else:
                self.status = StrategyStatus.ERROR
                self.logger.error(f"Strategy {self.strategy_id} failed to initialize")
                return False
        except Exception as e:
            self.status = StrategyStatus.ERROR
            self.logger.error(f"Error starting strategy {self.strategy_id}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the strategy execution."""
        try:
            self.status = StrategyStatus.INACTIVE
            self.logger.info(f"Strategy {self.strategy_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping strategy {self.strategy_id}: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause the strategy execution."""
        try:
            if self.status == StrategyStatus.ACTIVE:
                self.status = StrategyStatus.PAUSED
                self.logger.info(f"Strategy {self.strategy_id} paused")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error pausing strategy {self.strategy_id}: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the strategy execution."""
        try:
            if self.status == StrategyStatus.PAUSED:
                self.status = StrategyStatus.ACTIVE
                self.logger.info(f"Strategy {self.strategy_id} resumed")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error resuming strategy {self.strategy_id}: {e}")
            return False
    
    async def update_metrics(self):
        """Update strategy performance metrics."""
        try:
            # Calculate current P&L
            total_pnl, realized_pnl, unrealized_pnl = await self.calculate_pnl()
            
            self.metrics.total_pnl = total_pnl
            self.metrics.realized_pnl = realized_pnl
            self.metrics.unrealized_pnl = unrealized_pnl
            self.metrics.last_updated = datetime.now()
            
            # Update P&L history
            self.pnl_history.append(total_pnl)
            
            # Calculate performance ratios if we have enough data
            if len(self.pnl_history) > 30:  # At least 30 data points
                self._calculate_performance_ratios()
            
            # Update positions
            self.positions = await self.calculate_positions()
            self.metrics.active_positions = len([p for p in self.positions.values() if abs(p.net_exposure) > 1.0])
            
            # Calculate capital utilization
            total_exposure = sum(abs(p.net_exposure) for p in self.positions.values())
            self.metrics.capital_utilization = min(total_exposure / self.capital_allocation, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics for {self.strategy_id}: {e}")
    
    def _calculate_performance_ratios(self):
        """Calculate Sharpe, Sortino, and other performance ratios."""
        if len(self.pnl_history) < 2:
            return
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.pnl_history)):
            if self.pnl_history[i-1] != 0:
                ret = (self.pnl_history[i] - self.pnl_history[i-1]) / abs(self.pnl_history[i-1])
                returns.append(ret)
        
        if not returns:
            return
        
        import numpy as np
        
        returns_array = np.array(returns)
        
        # Sharpe ratio (assuming 2% risk-free rate, annualized)
        if np.std(returns_array) > 0:
            self.metrics.sharpe_ratio = (np.mean(returns_array) - 0.02/365) / np.std(returns_array) * np.sqrt(365)
        
        # Sortino ratio (downside deviation)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                self.metrics.sortino_ratio = (np.mean(returns_array) - 0.02/365) / downside_std * np.sqrt(365)
        
        # Max drawdown
        peak = self.pnl_history[0]
        max_dd = 0.0
        for pnl in self.pnl_history:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        self.metrics.max_drawdown = max_dd
        
        # Current drawdown
        current_peak = max(self.pnl_history[-30:])  # Peak in last 30 periods
        current_pnl = self.pnl_history[-1]
        self.metrics.current_drawdown = (current_peak - current_pnl) / current_peak if current_peak > 0 else 0.0
    
    async def check_risk_limits(self) -> bool:
        """
        Check if strategy is within risk limits.
        
        Returns:
            bool: True if within limits, False if breach detected
        """
        try:
            # Check drawdown limit
            if self.metrics.current_drawdown > self.risk_limits.max_drawdown:
                self.logger.warning(f"Drawdown limit breached: {self.metrics.current_drawdown:.1%} > {self.risk_limits.max_drawdown:.1%}")
                return False
            
            # Check daily loss limit
            if len(self.pnl_history) > 0:
                daily_pnl = self.pnl_history[-1] - (self.pnl_history[-2] if len(self.pnl_history) > 1 else 0)
                if daily_pnl < -self.risk_limits.max_daily_loss:
                    self.logger.warning(f"Daily loss limit breached: ${daily_pnl:.2f} < ${-self.risk_limits.max_daily_loss:.2f}")
                    return False
            
            # Check position size limits
            total_exposure = sum(abs(p.net_exposure) for p in self.positions.values())
            if total_exposure > self.risk_limits.max_position_size:
                self.logger.warning(f"Position size limit breached: ${total_exposure:.2f} > ${self.risk_limits.max_position_size:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def handle_error(self, error: Exception):
        """Handle strategy execution errors."""
        self.error_count += 1
        self.logger.error(f"Strategy error ({self.error_count}/{self.max_consecutive_errors}): {error}")
        
        if self.error_count >= self.max_consecutive_errors:
            self.logger.critical(f"Too many consecutive errors, triggering emergency stop")
            await self.emergency_stop()
            self.status = StrategyStatus.EMERGENCY_STOP
    
    def reset_error_count(self):
        """Reset error count after successful execution."""
        if self.error_count > 0:
            self.error_count = 0
            self.logger.info("Error count reset after successful execution")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy status summary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "capital_allocation": self.capital_allocation,
            "metrics": {
                "total_pnl": self.metrics.total_pnl,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "active_positions": self.metrics.active_positions,
                "capital_utilization": self.metrics.capital_utilization
            },
            "risk_status": {
                "error_count": self.error_count,
                "emergency_stopped": self.is_emergency_stopped,
                "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None
            },
            "positions": {symbol: {
                "net_exposure": pos.net_exposure,
                "delta_exposure": pos.delta_exposure
            } for symbol, pos in self.positions.items()}
        }