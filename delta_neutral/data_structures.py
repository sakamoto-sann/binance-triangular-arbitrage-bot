"""
Delta-Neutral Market Making Grid Bot - Core Data Structures
Professional-grade data structures for institutional market making.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class GridType(Enum):
    """Grid type enumeration."""
    SPOT_PROFIT = "SPOT_PROFIT"      # Dense profit-generating grid
    FUTURES_HEDGE = "FUTURES_HEDGE"  # Sparse delta-hedging ladder

@dataclass
class DeltaNeutralPosition:
    """Core delta-neutral position tracking."""
    spot_position: float = 0.0           # BTC spot position size
    futures_position: float = 0.0        # BTC futures position size
    net_delta: float = 0.0               # Combined delta exposure
    current_funding_rate: float = 0.0    # Current 8-hour funding rate
    basis_spread: float = 0.0            # Spot - Futures price spread
    last_rebalance: datetime = field(default_factory=datetime.now)
    total_pnl: float = 0.0               # Cumulative P&L
    
    def calculate_delta(self) -> float:
        """Calculate current net delta exposure."""
        # Spot has delta = 1.0, Futures has delta = -1.0 (short)
        self.net_delta = self.spot_position + self.futures_position
        return self.net_delta
    
    def needs_rebalance(self, threshold: float = 0.04) -> bool:
        """Check if position needs delta rebalancing."""
        return abs(self.calculate_delta()) > threshold

@dataclass
class GridLevel:
    """Individual grid level configuration."""
    price: float
    quantity: float
    side: PositionSide
    grid_type: GridType
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    success: bool
    order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    partial_fill: bool = False
    slippage: float = 0.0
    commission: float = 0.0
    error_message: Optional[str] = None
    emergency_hedge_needed: bool = False
    execution_time_ms: int = 0

@dataclass
class FundingRateData:
    """Funding rate information."""
    symbol: str
    funding_rate: float            # Current funding rate (8-hour)
    funding_time: datetime         # Next funding time
    predicted_rate: float          # Predicted next funding rate
    historical_avg: float = 0.0    # 7-day historical average
    is_positive: bool = True       # Whether funding is positive
    
    def is_attractive_for_long(self, threshold: float = 0.001) -> bool:
        """Check if funding rate is attractive for long positions."""
        return self.funding_rate > threshold

@dataclass
class BasisData:
    """Spot vs Futures basis information."""
    spot_price: float
    futures_price: float
    basis_points: float            # (Spot - Futures) / Spot * 10000
    basis_percentage: float        # (Spot - Futures) / Spot * 100
    is_contango: bool             # Futures > Spot
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_basis(self):
        """Calculate basis metrics."""
        if self.spot_price > 0:
            basis_abs = self.futures_price - self.spot_price
            self.basis_percentage = (basis_abs / self.spot_price) * 100
            self.basis_points = self.basis_percentage * 100
            self.is_contango = self.futures_price > self.spot_price

@dataclass
class MarketNeutralMetrics:
    """Performance metrics for market-neutral strategy."""
    total_pnl: float = 0.0
    grid_pnl: float = 0.0          # Profit from grid rebalancing
    funding_pnl: float = 0.0       # Profit from funding rates
    basis_pnl: float = 0.0         # Profit from basis trading
    spread_pnl: float = 0.0        # Profit from market making spreads
    
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    delta_exposure: float = 0.0     # Current delta exposure
    basis_exposure: float = 0.0     # Current basis risk
    
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    
    daily_funding_income: float = 0.0
    annualized_funding_rate: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimits:
    """Risk management limits and thresholds."""
    max_delta_exposure: float = 0.08        # 8% maximum delta
    max_drawdown: float = 0.05              # 5% maximum drawdown
    max_basis_risk: float = 0.015           # 1.5% maximum basis risk
    
    # Position sizing limits
    max_spot_position: float = 1.0          # Maximum spot position (BTC)
    max_futures_position: float = 1.0       # Maximum futures position (BTC)
    max_position_usd: float = 100000.0      # Maximum position value (USD)
    
    # Execution limits
    max_slippage: float = 0.001             # 0.1% maximum slippage
    max_execution_time_ms: int = 5000       # 5 second max execution time
    
    # API and system limits
    api_rate_limit_buffer: float = 0.2      # 20% buffer on rate limits
    max_failed_executions: int = 3          # Max consecutive failures
    
    # Emergency thresholds
    emergency_hedge_timeout_ms: int = 30000 # 30 second emergency timeout
    circuit_breaker_drawdown: float = 0.05 # 5% circuit breaker trigger

@dataclass
class GridConfiguration:
    """Configuration for grid trading parameters."""
    # Spot Profit Grid Configuration
    spot_grid_levels: int = 40              # Dense grid for max rebalancing
    spot_grid_spacing_pct: float = 0.005    # 0.5% spacing (tight)
    spot_position_size_pct: float = 0.02    # 2% position size per level
    
    # Futures Hedge Ladder Configuration  
    futures_grid_levels: int = 12           # Sparse ladder for efficiency
    futures_grid_spacing_pct: float = 0.025 # 2.5% spacing (wide)
    futures_hedge_ratio: float = 1.0        # Perfect hedge ratio
    
    # Dynamic adjustments
    volatility_scaling: bool = True         # Scale spacing with volatility
    funding_bias: bool = True               # Bias based on funding rates
    
    # ATR-based spacing
    atr_multiplier_min: float = 0.3         # Minimum ATR multiplier
    atr_multiplier_max: float = 0.8         # Maximum ATR multiplier
    
    # Rebalancing parameters
    delta_rebalance_threshold: float = 0.04 # 4% delta triggers rebalance
    min_rebalance_interval_seconds: int = 300 # 5 minute minimum between rebalances

@dataclass
class SystemState:
    """Overall system state and health."""
    is_active: bool = False
    is_emergency_mode: bool = False
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Connection states
    spot_exchange_connected: bool = False
    futures_exchange_connected: bool = False
    
    # System performance
    avg_execution_time_ms: float = 0.0
    failed_executions_count: int = 0
    api_rate_limit_usage: float = 0.0
    
    # Risk status
    current_risk_level: str = "LOW"         # LOW, MEDIUM, HIGH, CRITICAL
    active_alerts: List[str] = field(default_factory=list)
    
    def update_heartbeat(self):
        """Update system heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def add_alert(self, alert: str):
        """Add a system alert."""
        if alert not in self.active_alerts:
            self.active_alerts.append(alert)
    
    def clear_alert(self, alert: str):
        """Clear a system alert."""
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)