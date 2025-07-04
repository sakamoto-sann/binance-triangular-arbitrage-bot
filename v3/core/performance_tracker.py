"""
Grid Trading Bot v3.0 - Performance Tracker
Real-time performance tracking, metrics calculation, reporting, and benchmarking.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

from .market_analyzer import MarketAnalysis
from .grid_engine import GridMetrics, GridLevel
from .risk_manager import RiskMetrics, RiskAlert
from ..utils.config_manager import BotConfig
from ..utils.data_manager import DataManager, OrderData, PositionData, TradeData

logger = logging.getLogger(__name__)

class PerformancePeriod(Enum):
    """Performance analysis periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"

class PerformanceAlert(Enum):
    """Performance alert types."""
    NEW_EQUITY_HIGH = "new_equity_high"
    NEW_DRAWDOWN_HIGH = "new_drawdown_high"
    POOR_PERFORMANCE = "poor_performance"
    EXCELLENT_PERFORMANCE = "excellent_performance"
    BENCHMARK_UNDERPERFORMANCE = "benchmark_underperformance"
    RISK_ADJUSTED_UNDERPERFORMANCE = "risk_adjusted_underperformance"

@dataclass
class TradePerformance:
    """Individual trade performance metrics."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    commission: float
    duration_minutes: int
    grid_type: Optional[str] = None
    market_regime: Optional[str] = None
    
@dataclass
class PeriodPerformance:
    """Performance metrics for a specific period."""
    period: str
    start_date: datetime
    end_date: datetime
    starting_value: float
    ending_value: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration: float
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    
@dataclass
class GridPerformance:
    """Grid-specific performance metrics."""
    grid_type: str
    total_levels: int
    filled_levels: int
    fill_rate: float
    total_pnl: float
    avg_pnl_per_level: float
    best_level_pnl: float
    worst_level_pnl: float
    avg_fill_time: float
    efficiency_score: float
    
@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot."""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    active_positions: int
    benchmark_performance: float
    risk_adjusted_return: float

class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis system.
    """
    
    def __init__(self, config: BotConfig, data_manager: DataManager):
        """
        Initialize performance tracker.
        
        Args:
            config: Bot configuration.
            data_manager: Data manager instance.
        """
        self.config = config
        self.data_manager = data_manager
        
        # Configuration parameters
        self.initial_capital = config.trading.initial_balance
        self.risk_free_rate = config.risk_management.risk_free_rate
        self.benchmark_symbol = config.performance.get('benchmark_symbol', 'BTCUSDT')
        
        # Performance tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[TradePerformance] = []
        self.period_performance: Dict[str, PeriodPerformance] = {}
        self.grid_performance: Dict[str, GridPerformance] = {}
        
        # Current state
        self.current_portfolio_value: float = self.initial_capital
        self.portfolio_peak: float = self.initial_capital
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_duration: int = 0
        self.drawdown_start: Optional[datetime] = None
        
        # Benchmark tracking
        self.benchmark_data: Dict[str, float] = {}
        self.benchmark_initial_price: Optional[float] = None
        
        # Performance alerts
        self.performance_alerts: List[Dict[str, Any]] = []
        self.last_equity_high: float = self.initial_capital
        self.last_alert_check: datetime = datetime.now()
        
        # Statistics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_commission: float = 0.0
        self.total_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        
        # Attribution analysis
        self.strategy_attribution: Dict[str, float] = {}
        self.regime_attribution: Dict[str, float] = {}
        self.timeframe_attribution: Dict[str, float] = {}
        
        logger.info("Performance tracker initialized")
    
    def update_portfolio_value(self, portfolio_value: float, positions: List[PositionData],
                             market_analysis: Optional[MarketAnalysis] = None) -> None:
        """
        Update current portfolio value and calculate performance metrics.
        
        Args:
            portfolio_value: Current total portfolio value.
            positions: Current positions.
            market_analysis: Current market analysis.
        """
        try:
            timestamp = datetime.now()
            
            # Update current values
            self.current_portfolio_value = portfolio_value
            
            # Calculate unrealized P&L
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            # Update peak and drawdown
            if portfolio_value > self.portfolio_peak:
                self.portfolio_peak = portfolio_value
                self.current_drawdown = 0.0
                self.drawdown_start = None
            else:
                self.current_drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
                if self.drawdown_start is None:
                    self.drawdown_start = timestamp
                
                # Update max drawdown
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl()
            
            # Create portfolio snapshot
            snapshot = {
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'total_pnl': portfolio_value - self.initial_capital,
                'unrealized_pnl': self.unrealized_pnl,
                'realized_pnl': self.realized_pnl,
                'daily_pnl': daily_pnl,
                'drawdown': self.current_drawdown,
                'positions': len(positions),
                'market_regime': market_analysis.regime.value if market_analysis else 'unknown'
            }
            
            # Add to history
            self.portfolio_history.append(snapshot)
            
            # Keep limited history
            if len(self.portfolio_history) > 10000:
                self.portfolio_history = self.portfolio_history[-10000:]
            
            # Save to database
            self.data_manager.save_performance_snapshot(snapshot)
            
            # Check for performance alerts
            self._check_performance_alerts(snapshot)
            
            logger.debug(f"Portfolio updated: ${portfolio_value:,.2f}, Drawdown: {self.current_drawdown:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def record_trade(self, trade_data: TradeData, grid_type: Optional[str] = None,
                    market_regime: Optional[str] = None) -> None:
        """
        Record a completed trade for performance analysis.
        
        Args:
            trade_data: Trade execution data.
            grid_type: Type of grid that generated the trade.
            market_regime: Market regime during trade.
        """
        try:
            # Calculate trade metrics
            pnl = trade_data.exit_value - trade_data.entry_value - trade_data.commission
            pnl_pct = pnl / trade_data.entry_value if trade_data.entry_value > 0 else 0
            duration = int((trade_data.exit_time - trade_data.entry_time).total_seconds() / 60)
            
            # Create trade performance record
            trade_perf = TradePerformance(
                trade_id=trade_data.trade_id,
                symbol=trade_data.symbol,
                entry_time=trade_data.entry_time,
                exit_time=trade_data.exit_time,
                entry_price=trade_data.entry_price,
                exit_price=trade_data.exit_price,
                quantity=trade_data.quantity,
                side=trade_data.side,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=trade_data.commission,
                duration_minutes=duration,
                grid_type=grid_type,
                market_regime=market_regime
            )
            
            # Add to trade history
            self.trade_history.append(trade_perf)
            
            # Update statistics
            self.total_trades += 1
            self.total_commission += trade_data.commission
            self.realized_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update attribution analysis
            self._update_attribution_analysis(trade_perf)
            
            # Keep limited history
            if len(self.trade_history) > 10000:
                self.trade_history = self.trade_history[-10000:]
            
            # Save to database
            self.data_manager.save_trade_performance(asdict(trade_perf))
            
            logger.info(f"Trade recorded: {trade_data.symbol} {trade_data.side} P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def calculate_period_performance(self, period: PerformancePeriod,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> PeriodPerformance:
        """
        Calculate performance metrics for a specific period.
        
        Args:
            period: Performance period type.
            start_date: Custom start date (optional).
            end_date: Custom end date (optional).
            
        Returns:
            Period performance metrics.
        """
        try:
            # Determine date range
            if start_date is None or end_date is None:
                start_date, end_date = self._get_period_dates(period)
            
            # Filter data for period
            period_history = [
                snapshot for snapshot in self.portfolio_history
                if start_date <= snapshot['timestamp'] <= end_date
            ]
            
            period_trades = [
                trade for trade in self.trade_history
                if start_date <= trade.exit_time <= end_date
            ]
            
            if not period_history:
                return self._create_empty_period_performance(period.value, start_date, end_date)
            
            # Calculate basic metrics
            starting_value = period_history[0]['portfolio_value']
            ending_value = period_history[-1]['portfolio_value']
            total_return = ending_value - starting_value
            total_return_pct = total_return / starting_value if starting_value > 0 else 0
            
            # Calculate time-based metrics
            period_days = (end_date - start_date).days
            if period_days > 0:
                annualized_return = ((ending_value / starting_value) ** (365 / period_days)) - 1
            else:
                annualized_return = 0
            
            # Calculate volatility and risk metrics
            returns = self._calculate_period_returns(period_history)
            volatility = np.std(returns) * np.sqrt(365) if len(returns) > 1 else 0
            
            # Calculate Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
            else:
                sharpe_ratio = 0
            
            # Calculate Sortino ratio
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_deviation = np.std(negative_returns) * np.sqrt(365)
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation
            else:
                sortino_ratio = float('inf') if annualized_return > self.risk_free_rate else 0
            
            # Calculate drawdown metrics
            max_dd, max_dd_duration = self._calculate_period_drawdown(period_history)
            
            # Calculate Calmar ratio
            calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics(period_trades)
            
            # Calculate benchmark performance
            benchmark_return = self._calculate_benchmark_performance(start_date, end_date)
            
            # Calculate alpha and beta
            alpha, beta = self._calculate_alpha_beta(returns, benchmark_return)
            
            # Calculate information ratio
            information_ratio = self._calculate_information_ratio(returns, benchmark_return)
            
            period_perf = PeriodPerformance(
                period=period.value,
                start_date=start_date,
                end_date=end_date,
                starting_value=starting_value,
                ending_value=ending_value,
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_dd,
                max_drawdown_duration=max_dd_duration,
                calmar_ratio=calmar_ratio,
                benchmark_return=benchmark_return,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                **trade_stats
            )
            
            # Cache the result
            self.period_performance[period.value] = period_perf
            
            return period_perf
            
        except Exception as e:
            logger.error(f"Error calculating period performance: {e}")
            return self._create_empty_period_performance(period.value, start_date or datetime.now(), end_date or datetime.now())
    
    def calculate_grid_performance(self, grid_levels: List[GridLevel],
                                 filled_levels: List[GridLevel]) -> Dict[str, GridPerformance]:
        """
        Calculate performance metrics for each grid type.
        
        Args:
            grid_levels: All grid levels.
            filled_levels: Filled grid levels.
            
        Returns:
            Grid performance by type.
        """
        try:
            grid_performance = {}
            
            # Group by grid type
            grid_types = set(level.grid_type.value for level in grid_levels)
            
            for grid_type in grid_types:
                type_levels = [l for l in grid_levels if l.grid_type.value == grid_type]
                type_filled = [l for l in filled_levels if l.grid_type.value == grid_type]
                
                if not type_levels:
                    continue
                
                # Calculate metrics
                total_levels = len(type_levels)
                filled_count = len(type_filled)
                fill_rate = filled_count / total_levels if total_levels > 0 else 0
                
                # Calculate P&L from trades associated with these levels
                total_pnl = 0
                fill_times = []
                
                for level in type_filled:
                    # Find associated trades
                    level_trades = [
                        t for t in self.trade_history
                        if t.grid_type == grid_type and 
                        abs(t.entry_price - level.price) / level.price < 0.01  # Within 1%
                    ]
                    
                    for trade in level_trades:
                        total_pnl += trade.pnl
                        fill_times.append(trade.duration_minutes)
                
                avg_pnl_per_level = total_pnl / filled_count if filled_count > 0 else 0
                avg_fill_time = np.mean(fill_times) if fill_times else 0
                
                # Find best and worst performing levels
                level_pnls = []
                for level in type_filled:
                    level_trades = [
                        t for t in self.trade_history
                        if t.grid_type == grid_type and 
                        abs(t.entry_price - level.price) / level.price < 0.01
                    ]
                    if level_trades:
                        level_pnl = sum(t.pnl for t in level_trades)
                        level_pnls.append(level_pnl)
                
                best_level_pnl = max(level_pnls) if level_pnls else 0
                worst_level_pnl = min(level_pnls) if level_pnls else 0
                
                # Calculate efficiency score (fill rate * profitability)
                profitability_score = min(1.0, max(-1.0, total_pnl / abs(total_pnl) if total_pnl != 0 else 0))
                efficiency_score = fill_rate * (0.5 + profitability_score * 0.5)
                
                grid_perf = GridPerformance(
                    grid_type=grid_type,
                    total_levels=total_levels,
                    filled_levels=filled_count,
                    fill_rate=fill_rate,
                    total_pnl=total_pnl,
                    avg_pnl_per_level=avg_pnl_per_level,
                    best_level_pnl=best_level_pnl,
                    worst_level_pnl=worst_level_pnl,
                    avg_fill_time=avg_fill_time,
                    efficiency_score=efficiency_score
                )
                
                grid_performance[grid_type] = grid_perf
            
            # Update cached performance
            self.grid_performance.update(grid_performance)
            
            return grid_performance
            
        except Exception as e:
            logger.error(f"Error calculating grid performance: {e}")
            return {}
    
    def generate_performance_report(self, period: PerformancePeriod = PerformancePeriod.ALL_TIME,
                                  include_trades: bool = True,
                                  include_attribution: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            period: Period for the report.
            include_trades: Whether to include trade details.
            include_attribution: Whether to include attribution analysis.
            
        Returns:
            Comprehensive performance report.
        """
        try:
            # Calculate period performance
            period_perf = self.calculate_period_performance(period)
            
            # Current portfolio snapshot
            current_snapshot = self.get_current_performance_snapshot()
            
            # Risk metrics
            risk_summary = self._calculate_risk_summary()
            
            # Base report
            report = {
                'report_timestamp': datetime.now(),
                'period': period.value,
                'portfolio_summary': {
                    'current_value': self.current_portfolio_value,
                    'initial_capital': self.initial_capital,
                    'total_return': self.current_portfolio_value - self.initial_capital,
                    'total_return_pct': (self.current_portfolio_value - self.initial_capital) / self.initial_capital,
                    'unrealized_pnl': self.unrealized_pnl,
                    'realized_pnl': self.realized_pnl,
                    'total_commission': self.total_commission
                },
                'period_performance': asdict(period_perf),
                'current_snapshot': asdict(current_snapshot),
                'risk_metrics': risk_summary,
                'grid_performance': {k: asdict(v) for k, v in self.grid_performance.items()}
            }
            
            # Add trade analysis if requested
            if include_trades:
                report['trade_analysis'] = self._generate_trade_analysis()
            
            # Add attribution analysis if requested
            if include_attribution:
                report['attribution_analysis'] = {
                    'strategy_attribution': self.strategy_attribution,
                    'regime_attribution': self.regime_attribution,
                    'timeframe_attribution': self.timeframe_attribution
                }
            
            # Add performance alerts
            report['recent_alerts'] = self.performance_alerts[-10:]  # Last 10 alerts
            
            # Add benchmark comparison
            report['benchmark_comparison'] = self._generate_benchmark_comparison()
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def get_current_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Get current performance snapshot.
        
        Returns:
            Current performance snapshot.
        """
        try:
            # Calculate current metrics
            total_pnl = self.current_portfolio_value - self.initial_capital
            total_return_pct = total_pnl / self.initial_capital
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl()
            
            # Calculate current Sharpe ratio
            sharpe_ratio = self._calculate_current_sharpe_ratio()
            
            # Calculate win rate
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            # Get benchmark performance
            benchmark_perf = self._get_current_benchmark_performance()
            
            # Calculate risk-adjusted return
            risk_adjusted_return = sharpe_ratio * np.sqrt(252)  # Annualized
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=self.current_portfolio_value,
                total_pnl=total_pnl,
                unrealized_pnl=self.unrealized_pnl,
                realized_pnl=self.realized_pnl,
                daily_pnl=daily_pnl,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self.max_drawdown,
                win_rate=win_rate,
                total_trades=self.total_trades,
                active_positions=0,  # Would need position data
                benchmark_performance=benchmark_perf,
                risk_adjusted_return=risk_adjusted_return
            )
            
        except Exception as e:
            logger.error(f"Error creating performance snapshot: {e}")
            return PerformanceSnapshot(
                datetime.now(), self.current_portfolio_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
    
    def update_benchmark_price(self, symbol: str, price: float) -> None:
        """
        Update benchmark price for performance comparison.
        
        Args:
            symbol: Benchmark symbol.
            price: Current benchmark price.
        """
        try:
            if self.benchmark_initial_price is None:
                self.benchmark_initial_price = price
            
            self.benchmark_data[symbol] = price
            
        except Exception as e:
            logger.error(f"Error updating benchmark price: {e}")
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Find yesterday's end value
            yesterday_snapshots = [
                s for s in self.portfolio_history
                if s['timestamp'] < today_start
            ]
            
            if not yesterday_snapshots:
                return 0.0
            
            yesterday_end_value = yesterday_snapshots[-1]['portfolio_value']
            return self.current_portfolio_value - yesterday_end_value
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _calculate_current_sharpe_ratio(self) -> float:
        """Calculate current Sharpe ratio."""
        try:
            if len(self.portfolio_history) < 30:  # Need at least 30 data points
                return 0.0
            
            # Calculate daily returns for last 30 days
            recent_history = self.portfolio_history[-30:]
            returns = []
            
            for i in range(1, len(recent_history)):
                prev_value = recent_history[i-1]['portfolio_value']
                curr_value = recent_history[i]['portfolio_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return 0.0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_daily = self.risk_free_rate / 365
            
            if std_return > 0:
                return (avg_return - risk_free_daily) / std_return * np.sqrt(365)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _check_performance_alerts(self, snapshot: Dict[str, Any]) -> None:
        """Check for performance alerts."""
        try:
            current_value = snapshot['portfolio_value']
            
            # New equity high alert
            if current_value > self.last_equity_high:
                self.last_equity_high = current_value
                self._create_performance_alert(
                    PerformanceAlert.NEW_EQUITY_HIGH,
                    f"New portfolio high: ${current_value:,.2f}",
                    snapshot
                )
            
            # High drawdown alert
            if self.current_drawdown > 0.1:  # 10% drawdown
                self._create_performance_alert(
                    PerformanceAlert.NEW_DRAWDOWN_HIGH,
                    f"High drawdown: {self.current_drawdown:.1%}",
                    snapshot
                )
            
            # Performance alerts (check daily)
            if (datetime.now() - self.last_alert_check).days >= 1:
                self._check_daily_performance_alerts()
                self.last_alert_check = datetime.now()
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _create_performance_alert(self, alert_type: PerformanceAlert, 
                                message: str, snapshot: Dict[str, Any]) -> None:
        """Create a performance alert."""
        try:
            alert = {
                'alert_id': f"{alert_type.value}_{int(datetime.now().timestamp())}",
                'type': alert_type.value,
                'message': message,
                'timestamp': datetime.now(),
                'portfolio_value': snapshot['portfolio_value'],
                'total_pnl': snapshot['total_pnl'],
                'drawdown': snapshot['drawdown']
            }
            
            self.performance_alerts.append(alert)
            
            # Keep limited alerts
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]
            
            logger.info(f"Performance alert: {message}")
            
        except Exception as e:
            logger.error(f"Error creating performance alert: {e}")
    
    def _check_daily_performance_alerts(self) -> None:
        """Check daily performance metrics for alerts."""
        try:
            # Calculate daily performance
            daily_perf = self.calculate_period_performance(PerformancePeriod.DAILY)
            
            # Poor performance alert
            if daily_perf.total_return_pct < -0.05:  # -5% daily loss
                self._create_performance_alert(
                    PerformanceAlert.POOR_PERFORMANCE,
                    f"Poor daily performance: {daily_perf.total_return_pct:.1%}",
                    {'portfolio_value': self.current_portfolio_value, 'total_pnl': daily_perf.total_return, 'drawdown': self.current_drawdown}
                )
            
            # Excellent performance alert
            elif daily_perf.total_return_pct > 0.05:  # +5% daily gain
                self._create_performance_alert(
                    PerformanceAlert.EXCELLENT_PERFORMANCE,
                    f"Excellent daily performance: {daily_perf.total_return_pct:.1%}",
                    {'portfolio_value': self.current_portfolio_value, 'total_pnl': daily_perf.total_return, 'drawdown': self.current_drawdown}
                )
            
        except Exception as e:
            logger.error(f"Error checking daily performance alerts: {e}")
    
    def _update_attribution_analysis(self, trade: TradePerformance) -> None:
        """Update attribution analysis with new trade."""
        try:
            # Strategy attribution
            if trade.grid_type:
                if trade.grid_type not in self.strategy_attribution:
                    self.strategy_attribution[trade.grid_type] = 0
                self.strategy_attribution[trade.grid_type] += trade.pnl
            
            # Market regime attribution
            if trade.market_regime:
                if trade.market_regime not in self.regime_attribution:
                    self.regime_attribution[trade.market_regime] = 0
                self.regime_attribution[trade.market_regime] += trade.pnl
            
            # Timeframe attribution (based on trade duration)
            if trade.duration_minutes < 60:
                timeframe = "short_term"
            elif trade.duration_minutes < 1440:  # 24 hours
                timeframe = "medium_term"
            else:
                timeframe = "long_term"
            
            if timeframe not in self.timeframe_attribution:
                self.timeframe_attribution[timeframe] = 0
            self.timeframe_attribution[timeframe] += trade.pnl
            
        except Exception as e:
            logger.error(f"Error updating attribution analysis: {e}")
    
    def _get_period_dates(self, period: PerformancePeriod) -> Tuple[datetime, datetime]:
        """Get start and end dates for a period."""
        end_date = datetime.now()
        
        if period == PerformancePeriod.DAILY:
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.WEEKLY:
            days_since_monday = end_date.weekday()
            start_date = end_date - timedelta(days=days_since_monday)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.MONTHLY:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.QUARTERLY:
            quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
            start_date = end_date.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.YEARLY:
            start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # ALL_TIME
            start_date = datetime(2020, 1, 1)  # Or first trade date
        
        return start_date, end_date
    
    def _calculate_period_returns(self, period_history: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns for a period."""
        returns = []
        for i in range(1, len(period_history)):
            prev_value = period_history[i-1]['portfolio_value']
            curr_value = period_history[i]['portfolio_value']
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        return returns
    
    def _calculate_period_drawdown(self, period_history: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration for a period."""
        max_drawdown = 0
        max_duration = 0
        current_duration = 0
        peak = 0
        
        for snapshot in period_history:
            value = snapshot['portfolio_value']
            
            if value > peak:
                peak = value
                current_duration = 0
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        
        return max_drawdown, max_duration
    
    def _calculate_trade_statistics(self, trades: List[TradePerformance]) -> Dict[str, Any]:
        """Calculate trade statistics for a period."""
        if not trades:
            return {
                'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'largest_win': 0, 'largest_loss': 0, 'consecutive_wins': 0,
                'consecutive_losses': 0, 'avg_trade_duration': 0
            }
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        avg_duration = np.mean([t.duration_minutes for t in trades])
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'avg_trade_duration': avg_duration
        }
    
    def _calculate_benchmark_performance(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate benchmark performance for period."""
        # Placeholder implementation
        # In production, this would fetch actual benchmark data
        return 0.0
    
    def _calculate_alpha_beta(self, returns: List[float], benchmark_return: float) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark."""
        # Placeholder implementation
        # In production, this would calculate actual alpha/beta
        return 0.0, 1.0
    
    def _calculate_information_ratio(self, returns: List[float], benchmark_return: float) -> float:
        """Calculate information ratio."""
        # Placeholder implementation
        return 0.0
    
    def _calculate_risk_summary(self) -> Dict[str, Any]:
        """Calculate risk summary metrics."""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'volatility': self._calculate_current_volatility(),
            'var_95': self._calculate_var(0.95),
            'sharpe_ratio': self._calculate_current_sharpe_ratio()
        }
    
    def _calculate_current_volatility(self) -> float:
        """Calculate current portfolio volatility."""
        if len(self.portfolio_history) < 30:
            return 0.0
        
        returns = self._calculate_period_returns(self.portfolio_history[-30:])
        return np.std(returns) * np.sqrt(365) if returns else 0.0
    
    def _calculate_var(self, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(self.portfolio_history) < 30:
            return 0.0
        
        returns = self._calculate_period_returns(self.portfolio_history[-30:])
        if not returns:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100) * self.current_portfolio_value
    
    def _generate_trade_analysis(self) -> Dict[str, Any]:
        """Generate detailed trade analysis."""
        recent_trades = self.trade_history[-100:]  # Last 100 trades
        
        return {
            'total_trades': len(self.trade_history),
            'recent_trades_count': len(recent_trades),
            'recent_performance': self._calculate_trade_statistics(recent_trades),
            'trade_distribution': self._analyze_trade_distribution(),
            'duration_analysis': self._analyze_trade_durations(),
            'profitability_by_regime': self._analyze_regime_profitability()
        }
    
    def _analyze_trade_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of trade outcomes."""
        if not self.trade_history:
            return {}
        
        pnls = [t.pnl for t in self.trade_history]
        
        return {
            'mean_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'std_pnl': np.std(pnls),
            'skewness': self._calculate_skewness(pnls),
            'kurtosis': self._calculate_kurtosis(pnls)
        }
    
    def _analyze_trade_durations(self) -> Dict[str, Any]:
        """Analyze trade duration patterns."""
        if not self.trade_history:
            return {}
        
        durations = [t.duration_minutes for t in self.trade_history]
        
        return {
            'avg_duration_minutes': np.mean(durations),
            'median_duration_minutes': np.median(durations),
            'min_duration_minutes': min(durations),
            'max_duration_minutes': max(durations)
        }
    
    def _analyze_regime_profitability(self) -> Dict[str, float]:
        """Analyze profitability by market regime."""
        regime_pnl = {}
        
        for trade in self.trade_history:
            if trade.market_regime:
                if trade.market_regime not in regime_pnl:
                    regime_pnl[trade.market_regime] = []
                regime_pnl[trade.market_regime].append(trade.pnl)
        
        return {regime: np.mean(pnls) for regime, pnls in regime_pnl.items()}
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean([((x - mean) / std) ** 3 for x in data])
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean([((x - mean) / std) ** 4 for x in data]) - 3
    
    def _get_current_benchmark_performance(self) -> float:
        """Get current benchmark performance."""
        if not self.benchmark_data or self.benchmark_initial_price is None:
            return 0.0
        
        current_price = list(self.benchmark_data.values())[-1]
        return (current_price - self.benchmark_initial_price) / self.benchmark_initial_price
    
    def _generate_benchmark_comparison(self) -> Dict[str, Any]:
        """Generate benchmark comparison analysis."""
        return {
            'benchmark_symbol': self.benchmark_symbol,
            'benchmark_return': self._get_current_benchmark_performance(),
            'portfolio_return': (self.current_portfolio_value - self.initial_capital) / self.initial_capital,
            'outperformance': ((self.current_portfolio_value - self.initial_capital) / self.initial_capital) - self._get_current_benchmark_performance(),
            'tracking_error': 0.0,  # Placeholder
            'correlation': 0.0  # Placeholder
        }
    
    def _create_empty_period_performance(self, period: str, start_date: datetime, end_date: datetime) -> PeriodPerformance:
        """Create empty period performance for error cases."""
        return PeriodPerformance(
            period=period,
            start_date=start_date,
            end_date=end_date,
            starting_value=self.initial_capital,
            ending_value=self.current_portfolio_value,
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            calmar_ratio=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            largest_win=0,
            largest_loss=0,
            consecutive_wins=0,
            consecutive_losses=0,
            avg_trade_duration=0,
            benchmark_return=0,
            alpha=0,
            beta=1,
            information_ratio=0
        )
    
    def export_performance_data(self, filepath: str) -> bool:
        """
        Export performance data to file.
        
        Args:
            filepath: Path to save the data.
            
        Returns:
            Success status.
        """
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'current_value': self.current_portfolio_value,
                    'total_trades': self.total_trades
                },
                'portfolio_history': self.portfolio_history,
                'trade_history': [asdict(trade) for trade in self.trade_history],
                'performance_summary': self.generate_performance_report(),
                'attribution_analysis': {
                    'strategy_attribution': self.strategy_attribution,
                    'regime_attribution': self.regime_attribution,
                    'timeframe_attribution': self.timeframe_attribution
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Performance data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get concise performance summary.
        
        Returns:
            Performance summary.
        """
        try:
            return {
                'portfolio_value': self.current_portfolio_value,
                'total_return': self.current_portfolio_value - self.initial_capital,
                'total_return_pct': (self.current_portfolio_value - self.initial_capital) / self.initial_capital,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self._calculate_current_sharpe_ratio(),
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                'unrealized_pnl': self.unrealized_pnl,
                'realized_pnl': self.realized_pnl,
                'benchmark_outperformance': 0.0,  # Placeholder
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}