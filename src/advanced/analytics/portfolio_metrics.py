"""
Advanced Trading System - Portfolio Metrics
Comprehensive performance calculations and risk-adjusted returns analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns and P&L
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    upside_deviation: float = 0.0
    
    # Risk-adjusted ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Drawdown analysis
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0  # Days
    recovery_factor: float = 0.0
    pain_index: float = 0.0
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Activity metrics
    total_trades: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0
    turnover_ratio: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RiskDecomposition:
    """Risk decomposition analysis."""
    total_portfolio_risk: float = 0.0
    
    # Component contributions
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    correlation_risk: float = 0.0
    
    # Strategy contributions
    strategy_risk_contributions: Dict[str, float] = field(default_factory=dict)
    marginal_risk_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Factor exposures
    market_beta: float = 0.0
    momentum_exposure: float = 0.0
    volatility_exposure: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class BenchmarkComparison:
    """Benchmark comparison metrics."""
    benchmark_name: str = "BTC"
    
    # Relative performance
    excess_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    # Risk comparison
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    
    # Capture ratios
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    capture_ratio: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

class PortfolioMetricsCalculator:
    """
    Comprehensive portfolio metrics calculation engine.
    
    Calculates risk-adjusted returns, drawdown analysis, and comparative
    performance metrics for portfolio strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        
        # Data storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.returns_series: deque = deque(maxlen=5000)
        self.nav_series: deque = deque(maxlen=5000)
        self.benchmark_returns: deque = deque(maxlen=5000)
        
        # Performance tracking
        self.peak_nav = 0.0
        self.trough_nav = 0.0
        self.drawdown_start = None
        self.trades_history: List[Dict[str, Any]] = []
        
        # Settings
        self.min_periods = 30  # Minimum periods for reliable metrics
        self.trading_days_per_year = 365  # Crypto trades 24/7
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("PortfolioMetricsCalculator initialized")
    
    async def calculate_portfolio_metrics(self, portfolio_state: Dict[str, Any]) -> PerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            portfolio_state: Current portfolio state including NAV, positions, trades
            
        Returns:
            PerformanceMetrics with all calculated ratios and statistics
        """
        try:
            metrics = PerformanceMetrics()
            
            # Update data series
            current_nav = portfolio_state.get('net_asset_value', 0.0)
            await self._update_data_series(current_nav, portfolio_state)
            
            if len(self.nav_series) < 2:
                return metrics
            
            # Calculate return and P&L metrics
            await self._calculate_return_metrics(metrics, portfolio_state)
            
            # Calculate risk metrics
            await self._calculate_risk_metrics(metrics)
            
            # Calculate risk-adjusted ratios
            await self._calculate_risk_adjusted_ratios(metrics)
            
            # Calculate drawdown analysis
            await self._calculate_drawdown_metrics(metrics)
            
            # Calculate distribution metrics
            await self._calculate_distribution_metrics(metrics)
            
            # Calculate trading metrics
            await self._calculate_trading_metrics(metrics, portfolio_state)
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            metrics.last_updated = datetime.now()
            
            self.logger.debug(f"Portfolio metrics calculated: Sharpe={metrics.sharpe_ratio:.2f}, DD={metrics.max_drawdown:.1%}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return PerformanceMetrics()
    
    async def _update_data_series(self, current_nav: float, portfolio_state: Dict[str, Any]):
        """Update NAV and returns series."""
        try:
            self.nav_series.append(current_nav)
            
            # Calculate return if we have previous NAV
            if len(self.nav_series) > 1:
                prev_nav = self.nav_series[-2]
                if prev_nav > 0:
                    current_return = (current_nav - prev_nav) / prev_nav
                    self.returns_series.append(current_return)
            
            # Update peak tracking for drawdown
            if current_nav > self.peak_nav:
                self.peak_nav = current_nav
                self.drawdown_start = None
            elif self.drawdown_start is None and current_nav < self.peak_nav:
                self.drawdown_start = datetime.now()
            
            # Update trades history
            new_trades = portfolio_state.get('recent_trades', [])
            self.trades_history.extend(new_trades)
            if len(self.trades_history) > 10000:
                self.trades_history = self.trades_history[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error updating data series: {e}")
    
    async def _calculate_return_metrics(self, metrics: PerformanceMetrics, portfolio_state: Dict[str, Any]):
        """Calculate return and P&L metrics."""
        try:
            if len(self.nav_series) < 2:
                return
            
            nav_array = np.array(list(self.nav_series))
            initial_nav = nav_array[0]
            current_nav = nav_array[-1]
            
            # Total return
            if initial_nav > 0:
                metrics.total_return = (current_nav - initial_nav) / initial_nav
            
            # Annualized return
            if len(self.nav_series) > 1:
                periods = len(self.nav_series) - 1
                days = periods / (24 * 60)  # Assuming minute-level data
                years = days / 365
                
                if years > 0 and initial_nav > 0:
                    metrics.annualized_return = ((current_nav / initial_nav) ** (1 / years)) - 1
            
            # P&L metrics
            metrics.cumulative_pnl = current_nav - initial_nav
            metrics.realized_pnl = portfolio_state.get('realized_pnl', 0.0)
            metrics.unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating return metrics: {e}")
    
    async def _calculate_risk_metrics(self, metrics: PerformanceMetrics):
        """Calculate volatility and risk metrics."""
        try:
            if len(self.returns_series) < self.min_periods:
                return
            
            returns_array = np.array(list(self.returns_series))
            
            # Volatility
            metrics.volatility = np.std(returns_array)
            metrics.annualized_volatility = metrics.volatility * np.sqrt(self.trading_days_per_year * 24 * 60)
            
            # Downside and upside deviation
            negative_returns = returns_array[returns_array < 0]
            positive_returns = returns_array[returns_array > 0]
            
            metrics.downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            metrics.upside_deviation = np.std(positive_returns) if len(positive_returns) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
    
    async def _calculate_risk_adjusted_ratios(self, metrics: PerformanceMetrics):
        """Calculate risk-adjusted performance ratios."""
        try:
            if len(self.returns_series) < self.min_periods:
                return
            
            returns_array = np.array(list(self.returns_series))
            mean_return = np.mean(returns_array)
            
            # Convert to annualized values for ratios
            annualized_mean_return = mean_return * self.trading_days_per_year * 24 * 60
            daily_risk_free = self.risk_free_rate / self.trading_days_per_year
            
            # Sharpe Ratio
            if metrics.volatility > 0:
                excess_return = mean_return - (daily_risk_free / (24 * 60))
                metrics.sharpe_ratio = (excess_return / metrics.volatility) * np.sqrt(self.trading_days_per_year * 24 * 60)
            
            # Sortino Ratio
            if metrics.downside_deviation > 0:
                excess_return = mean_return - (daily_risk_free / (24 * 60))
                metrics.sortino_ratio = (excess_return / metrics.downside_deviation) * np.sqrt(self.trading_days_per_year * 24 * 60)
            
            # Calmar Ratio
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = annualized_mean_return / metrics.max_drawdown
            
            # Information Ratio (vs zero benchmark for market-neutral)
            if metrics.volatility > 0:
                metrics.information_ratio = mean_return / metrics.volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted ratios: {e}")
    
    async def _calculate_drawdown_metrics(self, metrics: PerformanceMetrics):
        """Calculate drawdown analysis metrics."""
        try:
            if len(self.nav_series) < 2:
                return
            
            nav_array = np.array(list(self.nav_series))
            
            # Calculate running maximum (peak)
            running_max = np.maximum.accumulate(nav_array)
            
            # Calculate drawdown series
            drawdown = (running_max - nav_array) / running_max
            
            # Current drawdown
            metrics.current_drawdown = drawdown[-1]
            
            # Maximum drawdown
            metrics.max_drawdown = np.max(drawdown)
            
            # Maximum drawdown duration
            if metrics.max_drawdown > 0:
                # Find the longest period below peak
                underwater_periods = []
                current_period = 0
                
                for dd in drawdown:
                    if dd > 0:
                        current_period += 1
                    else:
                        if current_period > 0:
                            underwater_periods.append(current_period)
                            current_period = 0
                
                if current_period > 0:
                    underwater_periods.append(current_period)
                
                if underwater_periods:
                    max_duration_periods = max(underwater_periods)
                    metrics.max_drawdown_duration = max_duration_periods / (24 * 60)  # Convert to days
            
            # Recovery factor
            if metrics.max_drawdown > 0:
                metrics.recovery_factor = metrics.total_return / metrics.max_drawdown
            
            # Pain index (average drawdown)
            metrics.pain_index = np.mean(drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
    
    async def _calculate_distribution_metrics(self, metrics: PerformanceMetrics):
        """Calculate return distribution metrics."""
        try:
            if len(self.returns_series) < self.min_periods:
                return
            
            returns_array = np.array(list(self.returns_series))
            
            # Skewness and kurtosis
            metrics.skewness = self._calculate_skewness(returns_array)
            metrics.kurtosis = self._calculate_kurtosis(returns_array)
            
            # Value at Risk
            metrics.var_95 = np.percentile(returns_array, 5)
            metrics.var_99 = np.percentile(returns_array, 1)
            
            # Conditional VaR (Expected Shortfall)
            var_95_threshold = metrics.var_95
            tail_returns = returns_array[returns_array <= var_95_threshold]
            if len(tail_returns) > 0:
                metrics.cvar_95 = np.mean(tail_returns)
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution metrics: {e}")
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        try:
            if len(returns) < 3:
                return 0.0
            
            mean = np.mean(returns)
            std = np.std(returns)
            
            if std == 0:
                return 0.0
            
            skew = np.mean(((returns - mean) / std) ** 3)
            return skew
            
        except Exception as e:
            self.logger.error(f"Error calculating skewness: {e}")
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        try:
            if len(returns) < 4:
                return 0.0
            
            mean = np.mean(returns)
            std = np.std(returns)
            
            if std == 0:
                return 0.0
            
            kurt = np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis
            return kurt
            
        except Exception as e:
            self.logger.error(f"Error calculating kurtosis: {e}")
            return 0.0
    
    async def _calculate_trading_metrics(self, metrics: PerformanceMetrics, portfolio_state: Dict[str, Any]):
        """Calculate trading-specific metrics."""
        try:
            if not self.trades_history:
                return
            
            # Filter recent trades
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_trades = [
                trade for trade in self.trades_history 
                if trade.get('timestamp', datetime.min) >= recent_cutoff
            ]
            
            if not recent_trades:
                return
            
            # Basic trade counts
            metrics.total_trades = len(recent_trades)
            
            # Analyze trade results
            winning_trades = []
            losing_trades = []
            trade_pnls = []
            
            for trade in recent_trades:
                pnl = trade.get('pnl', 0.0)
                trade_pnls.append(pnl)
                
                if pnl > 0:
                    winning_trades.append(pnl)
                elif pnl < 0:
                    losing_trades.append(pnl)
            
            metrics.profitable_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            
            # Win rate
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.profitable_trades / metrics.total_trades
            
            # Average win/loss
            if winning_trades:
                metrics.avg_win = np.mean(winning_trades)
                metrics.largest_win = max(winning_trades)
            
            if losing_trades:
                metrics.avg_loss = np.mean(losing_trades)
                metrics.largest_loss = min(losing_trades)  # Most negative
            
            # Profit factor
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0
            
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
            elif total_wins > 0:
                metrics.profit_factor = float('inf')
            
            # Expectancy
            if metrics.total_trades > 0:
                metrics.expectancy = sum(trade_pnls) / metrics.total_trades
            
            # Turnover ratio (simplified)
            total_volume = sum(abs(trade.get('quantity', 0) * trade.get('price', 0)) for trade in recent_trades)
            current_nav = portfolio_state.get('net_asset_value', 1.0)
            if current_nav > 0:
                metrics.turnover_ratio = total_volume / current_nav
            
        except Exception as e:
            self.logger.error(f"Error calculating trading metrics: {e}")
    
    async def calculate_risk_decomposition(self, strategies: Dict[str, Any]) -> RiskDecomposition:
        """Calculate portfolio risk decomposition."""
        try:
            decomposition = RiskDecomposition()
            
            if not strategies or len(self.returns_series) < self.min_periods:
                return decomposition
            
            # Calculate total portfolio volatility
            portfolio_returns = np.array(list(self.returns_series))
            decomposition.total_portfolio_risk = np.std(portfolio_returns)
            
            # Calculate strategy risk contributions
            total_capital = sum(s.capital_allocation for s in strategies.values())
            
            for strategy_id, strategy in strategies.items():
                if hasattr(strategy, 'pnl_history') and len(strategy.pnl_history) > 1:
                    # Calculate strategy returns
                    strategy_returns = np.diff(strategy.pnl_history) / np.array(strategy.pnl_history[:-1])
                    strategy_vol = np.std(strategy_returns[-len(portfolio_returns):])
                    
                    # Weight by capital allocation
                    weight = strategy.capital_allocation / total_capital if total_capital > 0 else 0
                    
                    # Risk contribution (simplified)
                    risk_contribution = weight * strategy_vol
                    decomposition.strategy_risk_contributions[strategy_id] = risk_contribution
                    
                    # Marginal risk contribution
                    if len(strategy_returns) >= len(portfolio_returns):
                        aligned_strategy_returns = strategy_returns[-len(portfolio_returns):]
                        correlation = np.corrcoef(portfolio_returns, aligned_strategy_returns)[0, 1]
                        marginal_contribution = weight * strategy_vol * correlation
                        decomposition.marginal_risk_contributions[strategy_id] = marginal_contribution
            
            decomposition.last_updated = datetime.now()
            return decomposition
            
        except Exception as e:
            self.logger.error(f"Error calculating risk decomposition: {e}")
            return RiskDecomposition()
    
    async def calculate_benchmark_comparison(self, benchmark_returns: List[float], 
                                           benchmark_name: str = "BTC") -> BenchmarkComparison:
        """Calculate performance vs benchmark."""
        try:
            comparison = BenchmarkComparison(benchmark_name=benchmark_name)
            
            if (len(self.returns_series) < self.min_periods or 
                len(benchmark_returns) < self.min_periods):
                return comparison
            
            # Align return series
            min_length = min(len(self.returns_series), len(benchmark_returns))
            portfolio_returns = np.array(list(self.returns_series)[-min_length:])
            bench_returns = np.array(benchmark_returns[-min_length:])
            
            # Excess return
            excess_returns = portfolio_returns - bench_returns
            comparison.excess_return = np.mean(excess_returns)
            
            # Tracking error
            comparison.tracking_error = np.std(excess_returns)
            
            # Information ratio
            if comparison.tracking_error > 0:
                comparison.information_ratio = comparison.excess_return / comparison.tracking_error
            
            # Beta and alpha
            if np.std(bench_returns) > 0:
                covariance = np.cov(portfolio_returns, bench_returns)[0, 1]
                benchmark_variance = np.var(bench_returns)
                comparison.beta = covariance / benchmark_variance
                
                portfolio_mean = np.mean(portfolio_returns)
                benchmark_mean = np.mean(bench_returns)
                comparison.alpha = portfolio_mean - (comparison.beta * benchmark_mean)
            
            # Correlation
            comparison.correlation = np.corrcoef(portfolio_returns, bench_returns)[0, 1]
            
            # Capture ratios
            up_periods = bench_returns > 0
            down_periods = bench_returns < 0
            
            if np.sum(up_periods) > 0:
                portfolio_up = np.mean(portfolio_returns[up_periods])
                benchmark_up = np.mean(bench_returns[up_periods])
                if benchmark_up != 0:
                    comparison.upside_capture = portfolio_up / benchmark_up
            
            if np.sum(down_periods) > 0:
                portfolio_down = np.mean(portfolio_returns[down_periods])
                benchmark_down = np.mean(bench_returns[down_periods])
                if benchmark_down != 0:
                    comparison.downside_capture = portfolio_down / benchmark_down
            
            # Overall capture ratio
            if comparison.downside_capture != 0:
                comparison.capture_ratio = comparison.upside_capture / abs(comparison.downside_capture)
            
            comparison.last_updated = datetime.now()
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark comparison: {e}")
            return BenchmarkComparison()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        return {
            "performance": {
                "total_return": latest.total_return,
                "annualized_return": latest.annualized_return,
                "sharpe_ratio": latest.sharpe_ratio,
                "sortino_ratio": latest.sortino_ratio,
                "calmar_ratio": latest.calmar_ratio
            },
            "risk": {
                "volatility": latest.annualized_volatility,
                "max_drawdown": latest.max_drawdown,
                "current_drawdown": latest.current_drawdown,
                "var_95": latest.var_95,
                "var_99": latest.var_99
            },
            "trading": {
                "total_trades": latest.total_trades,
                "win_rate": latest.win_rate,
                "profit_factor": latest.profit_factor,
                "expectancy": latest.expectancy
            },
            "distribution": {
                "skewness": latest.skewness,
                "kurtosis": latest.kurtosis
            },
            "data_quality": {
                "periods": len(self.returns_series),
                "sufficient_data": len(self.returns_series) >= self.min_periods
            }
        }
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        self.returns_series.clear()
        self.nav_series.clear()
        self.trades_history.clear()
        self.peak_nav = 0.0
        self.logger.info("Portfolio metrics history cleared")