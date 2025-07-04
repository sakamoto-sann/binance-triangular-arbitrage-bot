"""
Advanced Trading System - Portfolio Manager
Central orchestrator for managing multiple trading strategies with advanced risk controls and analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

# Import all advanced components
from ..strategies.base_strategy import BaseStrategy, StrategyStatus, StrategyType
from ..risk.dynamic_risk_manager import DynamicRiskManager, RiskLevel
from ..analytics.performance_attribution import PerformanceAttributor
from ..analytics.portfolio_metrics import PortfolioMetricsCalculator
from ..analytics.risk_metrics import RiskMetricsCalculator
from ..data.market_data_aggregator import MarketDataAggregator
from ..data.funding_rate_analyzer import FundingRateAnalyzer
from ..data.volatility_surface_tracker import VolatilitySurfaceTracker
from ..data.liquidity_flow_analyzer import LiquidityFlowAnalyzer
from ..core.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class PortfolioMode(Enum):
    """Portfolio operation modes."""
    ACTIVE = "ACTIVE"
    CONSERVATIVE = "CONSERVATIVE" 
    EMERGENCY = "EMERGENCY"
    MAINTENANCE = "MAINTENANCE"
    SHUTDOWN = "SHUTDOWN"

@dataclass
class PortfolioConfig:
    """Portfolio-level configuration."""
    total_capital: float
    max_strategies: int = 10
    rebalance_frequency_minutes: int = 15
    risk_check_frequency_seconds: int = 30
    emergency_stop_drawdown: float = 0.15  # 15%
    max_correlation_threshold: float = 0.8
    max_leverage: float = 3.0
    enable_auto_rebalancing: bool = True
    enable_dynamic_allocation: bool = False

@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""
    total_capital: float = 0.0
    net_asset_value: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    portfolio_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    active_strategies: int = 0
    total_positions: int = 0
    capital_utilization: float = 0.0
    correlation_risk: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class PortfolioManager:
    """
    Central portfolio manager orchestrating multiple trading strategies.
    
    This class is responsible for:
    - Managing strategy lifecycle (start, stop, pause, resume)
    - Capital allocation and rebalancing
    - Portfolio-level risk management
    - Performance monitoring and reporting
    - Emergency procedures and circuit breakers
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize the portfolio manager.
        
        Args:
            config: Portfolio configuration parameters
        """
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, float] = {}
        
        # Core components
        self.market_data = MarketDataAggregator()
        self.risk_manager = DynamicRiskManager(config.total_capital)
        self.performance_attributor = PerformanceAttributor()
        self.metrics_calculator = PortfolioMetricsCalculator()
        self.risk_metrics = RiskMetricsCalculator()
        self.system_monitor = SystemMonitor()
        
        # Advanced data analysis components  
        self.funding_analyzer: Optional[FundingRateAnalyzer] = None
        self.volatility_tracker: Optional[VolatilitySurfaceTracker] = None
        self.liquidity_analyzer: Optional[LiquidityFlowAnalyzer] = None
        
        # Portfolio state
        self.is_active = False
        self.is_emergency_stopped = False
        self.current_mode = PortfolioMode.MAINTENANCE
        self.last_rebalance_time = None
        self.last_risk_check_time = None
        
        # Performance tracking
        self.portfolio_metrics = PortfolioMetrics(total_capital=config.total_capital)
        self.nav_history: List[float] = []
        self.returns_history: List[float] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PortfolioManager initialized with ${config.total_capital:,.0f} capital")
    
    async def add_strategy(self, strategy: BaseStrategy, allocation_pct: float) -> bool:
        """
        Add a strategy to the portfolio.
        
        Args:
            strategy: Strategy instance to add
            allocation_pct: Percentage of portfolio capital to allocate (0.0-1.0)
            
        Returns:
            bool: True if strategy added successfully
        """
        try:
            if strategy.strategy_id in self.strategies:
                self.logger.warning(f"Strategy {strategy.strategy_id} already exists")
                return False
            
            if len(self.strategies) >= self.config.max_strategies:
                self.logger.error(f"Maximum strategies ({self.config.max_strategies}) reached")
                return False
            
            if allocation_pct <= 0 or allocation_pct > 1:
                self.logger.error(f"Invalid allocation percentage: {allocation_pct}")
                return False
            
            # Check if total allocation would exceed 100%
            total_allocation = sum(self.strategy_allocations.values()) + allocation_pct
            if total_allocation > 1.0:
                self.logger.error(f"Total allocation would exceed 100%: {total_allocation:.1%}")
                return False
            
            # Calculate capital allocation
            strategy.capital_allocation = self.config.total_capital * allocation_pct
            
            # Add strategy
            self.strategies[strategy.strategy_id] = strategy
            self.strategy_allocations[strategy.strategy_id] = allocation_pct
            
            self.logger.info(f"Strategy {strategy.strategy_id} added with {allocation_pct:.1%} allocation (${strategy.capital_allocation:,.0f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy.strategy_id}: {e}")
            return False
    
    async def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the portfolio.
        
        Args:
            strategy_id: ID of strategy to remove
            
        Returns:
            bool: True if strategy removed successfully
        """
        try:
            if strategy_id not in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} not found")
                return False
            
            strategy = self.strategies[strategy_id]
            
            # Stop strategy first
            await strategy.stop()
            
            # Emergency stop if needed
            if strategy.status == StrategyStatus.ACTIVE:
                await strategy.emergency_stop()
            
            # Remove from portfolio
            del self.strategies[strategy_id]
            del self.strategy_allocations[strategy_id]
            
            self.logger.info(f"Strategy {strategy_id} removed from portfolio")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing strategy {strategy_id}: {e}")
            return False
    
    async def start_portfolio(self) -> bool:
        """Start portfolio execution with all advanced components."""
        try:
            self.logger.info("Starting advanced portfolio execution...")
            
            # Start market data aggregation first
            if not await self.market_data.start():
                self.logger.error("Failed to start market data aggregator")
                return False
            
            # Initialize advanced data analysis components
            self.funding_analyzer = FundingRateAnalyzer(self.market_data)
            self.volatility_tracker = VolatilitySurfaceTracker(self.market_data) 
            self.liquidity_analyzer = LiquidityFlowAnalyzer(self.market_data)
            
            # Start all data analysis components
            analysis_start_tasks = [
                self.funding_analyzer.start(),
                self.volatility_tracker.start(),
                self.liquidity_analyzer.start(),
                self.system_monitor.start()
            ]
            
            analysis_results = await asyncio.gather(*analysis_start_tasks, return_exceptions=True)
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to start analysis component {i}: {result}")
            
            # Initialize all strategies
            failed_strategies = []
            for strategy_id, strategy in self.strategies.items():
                if not await strategy.start():
                    failed_strategies.append(strategy_id)
                    self.logger.error(f"Failed to start strategy: {strategy_id}")
            
            # Remove failed strategies
            for strategy_id in failed_strategies:
                await self.remove_strategy(strategy_id)
            
            if not self.strategies:
                self.logger.error("No strategies successfully started")
                return False
            
            # Start comprehensive monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._portfolio_monitoring_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._rebalancing_loop()),
                asyncio.create_task(self._performance_tracking_loop()),
                asyncio.create_task(self._advanced_analytics_loop()),
                asyncio.create_task(self._mode_management_loop())
            ]
            
            self.is_active = True
            self.current_mode = PortfolioMode.ACTIVE
            self.logger.info(f"Advanced portfolio started with {len(self.strategies)} active strategies")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting portfolio: {e}")
            return False
    
    async def stop_portfolio(self) -> bool:
        """Stop portfolio execution and all components."""
        try:
            self.logger.info("Stopping advanced portfolio execution...")
            
            self.current_mode = PortfolioMode.SHUTDOWN
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Stop all strategies
            for strategy in self.strategies.values():
                await strategy.stop()
            
            # Stop all analysis components
            stop_tasks = []
            if self.funding_analyzer:
                stop_tasks.append(self.funding_analyzer.stop())
            if self.volatility_tracker:
                stop_tasks.append(self.volatility_tracker.stop())
            if self.liquidity_analyzer:
                stop_tasks.append(self.liquidity_analyzer.stop())
            if self.system_monitor:
                stop_tasks.append(self.system_monitor.stop())
            
            # Stop market data last
            stop_tasks.append(self.market_data.stop())
            
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            self.is_active = False
            self.logger.info("Advanced portfolio stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping portfolio: {e}")
            return False
    
    async def emergency_stop_portfolio(self) -> bool:
        """Emergency stop - immediately halt all trading activity."""
        try:
            self.logger.critical("EMERGENCY STOP TRIGGERED")
            
            self.is_emergency_stopped = True
            
            # Cancel all monitoring tasks immediately
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Emergency stop all strategies in parallel
            emergency_tasks = [
                strategy.emergency_stop() for strategy in self.strategies.values()
            ]
            await asyncio.gather(*emergency_tasks, return_exceptions=True)
            
            # Stop market data
            await self.market_data.stop()
            
            self.is_active = False
            self.logger.critical("Emergency stop completed")
            return True
            
        except Exception as e:
            self.logger.critical(f"Error during emergency stop: {e}")
            return False
    
    async def _portfolio_monitoring_loop(self):
        """Main portfolio monitoring loop."""
        while self.is_active:
            try:
                # Update all strategy metrics
                update_tasks = [
                    strategy.update_metrics() for strategy in self.strategies.values()
                ]
                await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Calculate portfolio metrics
                await self._calculate_portfolio_metrics()
                
                # Execute strategy cycles
                market_data = await self.market_data.get_current_data()
                execution_tasks = [
                    strategy.execute_cycle(market_data) 
                    for strategy in self.strategies.values()
                    if strategy.status == StrategyStatus.ACTIVE
                ]
                
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Handle execution results
                for i, (strategy_id, result) in enumerate(zip(self.strategies.keys(), results)):
                    strategy = self.strategies[strategy_id]
                    if isinstance(result, Exception):
                        await strategy.handle_error(result)
                    elif result:
                        strategy.reset_error_count()
                
                # Sleep before next cycle
                await asyncio.sleep(1)  # 1 second execution cycle
                
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring and management loop."""
        while self.is_active:
            try:
                # Check portfolio-level risks
                risk_status = await self.risk_manager.assess_portfolio_risk(
                    self.strategies, self.portfolio_metrics
                )
                
                # Check for emergency stop conditions
                if (self.portfolio_metrics.current_drawdown > self.config.emergency_stop_drawdown or
                    risk_status.get('critical_risk', False)):
                    self.logger.critical("Emergency stop conditions detected")
                    await self.emergency_stop_portfolio()
                    break
                
                # Check individual strategy risk limits
                for strategy in self.strategies.values():
                    if not await strategy.check_risk_limits():
                        self.logger.warning(f"Risk limit breach for strategy: {strategy.strategy_id}")
                        await strategy.pause()
                
                # Update correlation risk
                await self._update_correlation_risk()
                
                self.last_risk_check_time = datetime.now()
                await asyncio.sleep(self.config.risk_check_frequency_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _rebalancing_loop(self):
        """Portfolio rebalancing loop."""
        while self.is_active:
            try:
                if (self.config.enable_auto_rebalancing and 
                    (self.last_rebalance_time is None or 
                     datetime.now() - self.last_rebalance_time > timedelta(minutes=self.config.rebalance_frequency_minutes))):
                    
                    await self._rebalance_portfolio()
                    self.last_rebalance_time = datetime.now()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self):
        """Performance tracking and attribution loop."""
        while self.is_active:
            try:
                # Update performance attribution
                attribution = await self.performance_attributor.calculate_attribution(self.strategies)
                
                # Store NAV history
                current_nav = self.portfolio_metrics.net_asset_value
                self.nav_history.append(current_nav)
                
                # Calculate returns if we have history
                if len(self.nav_history) > 1:
                    return_pct = (current_nav - self.nav_history[-2]) / self.nav_history[-2]
                    self.returns_history.append(return_pct)
                
                # Trim history to last 10,000 points for memory management
                if len(self.nav_history) > 10000:
                    self.nav_history = self.nav_history[-5000:]
                    self.returns_history = self.returns_history[-5000:]
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio metrics."""
        try:
            # Aggregate strategy metrics
            total_pnl = sum(s.metrics.total_pnl for s in self.strategies.values())
            total_unrealized = sum(s.metrics.unrealized_pnl for s in self.strategies.values())
            total_realized = sum(s.metrics.realized_pnl for s in self.strategies.values())
            
            # Calculate NAV
            nav = self.config.total_capital + total_pnl
            
            # Update basic metrics
            self.portfolio_metrics.total_pnl = total_pnl
            self.portfolio_metrics.unrealized_pnl = total_unrealized
            self.portfolio_metrics.realized_pnl = total_realized
            self.portfolio_metrics.net_asset_value = nav
            self.portfolio_metrics.portfolio_return = total_pnl / self.config.total_capital
            
            # Count active strategies and positions
            self.portfolio_metrics.active_strategies = sum(
                1 for s in self.strategies.values() if s.status == StrategyStatus.ACTIVE
            )
            self.portfolio_metrics.total_positions = sum(
                s.metrics.active_positions for s in self.strategies.values()
            )
            
            # Calculate capital utilization
            total_exposure = sum(
                sum(abs(pos.net_exposure) for pos in s.positions.values())
                for s in self.strategies.values()
            )
            self.portfolio_metrics.capital_utilization = min(total_exposure / self.config.total_capital, 1.0)
            
            # Calculate advanced metrics if we have enough history
            if len(self.returns_history) > 30:
                await self._calculate_advanced_metrics()
            
            self.portfolio_metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
    
    async def _calculate_advanced_metrics(self):
        """Calculate advanced performance metrics."""
        try:
            if not self.returns_history:
                return
            
            returns_array = np.array(self.returns_history)
            
            # Volatility (annualized)
            self.portfolio_metrics.volatility = np.std(returns_array) * np.sqrt(365 * 24)  # Hourly data
            
            # Sharpe ratio
            if self.portfolio_metrics.volatility > 0:
                risk_free_rate = 0.02 / (365 * 24)  # 2% annual, converted to hourly
                excess_return = np.mean(returns_array) - risk_free_rate
                self.portfolio_metrics.sharpe_ratio = excess_return / np.std(returns_array) * np.sqrt(365 * 24)
            
            # Sortino ratio
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    risk_free_rate = 0.02 / (365 * 24)
                    self.portfolio_metrics.sortino_ratio = (np.mean(returns_array) - risk_free_rate) / downside_std * np.sqrt(365 * 24)
            
            # Drawdown calculations
            nav_array = np.array(self.nav_history)
            peak = np.maximum.accumulate(nav_array)
            drawdown = (peak - nav_array) / peak
            
            self.portfolio_metrics.max_drawdown = np.max(drawdown)
            self.portfolio_metrics.current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
            
            # Calmar ratio
            if self.portfolio_metrics.max_drawdown > 0:
                annual_return = self.portfolio_metrics.portfolio_return * (365 * 24) / len(self.returns_history)
                self.portfolio_metrics.calmar_ratio = annual_return / self.portfolio_metrics.max_drawdown
            
            # VaR calculations
            if len(returns_array) >= 100:
                self.portfolio_metrics.var_95 = np.percentile(returns_array, 5)
                self.portfolio_metrics.var_99 = np.percentile(returns_array, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
    
    async def _update_correlation_risk(self):
        """Update correlation risk assessment."""
        try:
            if len(self.strategies) < 2:
                self.portfolio_metrics.correlation_risk = 0.0
                return
            
            # Get strategy returns
            strategy_returns = {}
            for strategy_id, strategy in self.strategies.items():
                if len(strategy.pnl_history) > 1:
                    returns = np.diff(strategy.pnl_history) / np.array(strategy.pnl_history[:-1])
                    strategy_returns[strategy_id] = returns[-100:]  # Last 100 returns
            
            if len(strategy_returns) < 2:
                return
            
            # Calculate correlation matrix
            strategy_ids = list(strategy_returns.keys())
            returns_matrix = np.array([strategy_returns[sid] for sid in strategy_ids])
            
            # Ensure all return series have the same length
            min_length = min(len(returns) for returns in returns_matrix)
            returns_matrix = np.array([returns[-min_length:] for returns in returns_matrix])
            
            if min_length > 10:  # Need at least 10 data points
                self.correlation_matrix = np.corrcoef(returns_matrix)
                
                # Calculate average correlation (excluding diagonal)
                mask = ~np.eye(self.correlation_matrix.shape[0], dtype=bool)
                avg_correlation = np.mean(np.abs(self.correlation_matrix[mask]))
                self.portfolio_metrics.correlation_risk = avg_correlation
                
                # Check for high correlation warning
                if avg_correlation > self.config.max_correlation_threshold:
                    self.logger.warning(f"High correlation detected: {avg_correlation:.1%}")
            
        except Exception as e:
            self.logger.error(f"Error updating correlation risk: {e}")
    
    async def _rebalance_portfolio(self):
        """Rebalance portfolio allocations if needed."""
        try:
            if not self.config.enable_dynamic_allocation:
                return
            
            self.logger.info("Performing portfolio rebalancing...")
            
            # Calculate target allocations based on performance
            current_nav = self.portfolio_metrics.net_asset_value
            
            for strategy_id, strategy in self.strategies.items():
                current_allocation = strategy.capital_allocation / current_nav
                target_allocation = self.strategy_allocations[strategy_id]
                
                allocation_drift = abs(current_allocation - target_allocation)
                
                # Rebalance if drift > 5%
                if allocation_drift > 0.05:
                    new_capital = current_nav * target_allocation
                    old_capital = strategy.capital_allocation
                    
                    strategy.capital_allocation = new_capital
                    
                    self.logger.info(f"Rebalanced {strategy_id}: ${old_capital:,.0f} -> ${new_capital:,.0f}")
            
        except Exception as e:
            self.logger.error(f"Error during rebalancing: {e}")
    
    async def _advanced_analytics_loop(self):
        """Advanced analytics and data processing loop."""
        while self.is_active:
            try:
                # Comprehensive risk metrics calculation
                if self.risk_metrics and len(self.returns_history) > 30:
                    returns_data = self.returns_history[-100:]  # Last 100 returns
                    portfolio_value = self.portfolio_metrics.net_asset_value
                    
                    var_results = await self.risk_metrics.calculate_portfolio_var(
                        returns_data, portfolio_value
                    )
                    
                    # Update portfolio metrics with VaR
                    if var_results:
                        for confidence_level, var_result in var_results.items():
                            if confidence_level == 0.95:
                                self.portfolio_metrics.var_95 = var_result.historical_var / portfolio_value
                            elif confidence_level == 0.99:
                                self.portfolio_metrics.var_99 = var_result.historical_var / portfolio_value
                
                # Run stress tests periodically
                stress_test_results = await self.risk_metrics.run_stress_tests(
                    self.strategies, self.portfolio_metrics.net_asset_value
                )
                
                # Correlation analysis
                correlation_analysis = await self.risk_metrics.analyze_correlations(self.strategies)
                if correlation_analysis:
                    self.portfolio_metrics.correlation_risk = correlation_analysis.average_correlation
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in advanced analytics loop: {e}")
                await asyncio.sleep(1800)
    
    async def _mode_management_loop(self):
        """Portfolio mode management and dynamic adjustments."""
        while self.is_active:
            try:
                # Get system health status
                system_status = self.system_monitor.get_system_status() if self.system_monitor else {}
                system_health = system_status.get('health', {}).get('health_score', 1.0)
                
                # Determine appropriate mode based on conditions
                new_mode = self._determine_portfolio_mode(system_health)
                
                if new_mode != self.current_mode:
                    await self._change_portfolio_mode(new_mode)
                
                # Mode-specific adjustments
                await self._apply_mode_adjustments()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Error in mode management loop: {e}")
                await asyncio.sleep(120)
    
    def _determine_portfolio_mode(self, system_health: float) -> PortfolioMode:
        """Determine appropriate portfolio mode based on current conditions."""
        try:
            # Emergency conditions
            if (self.is_emergency_stopped or
                self.portfolio_metrics.current_drawdown > self.config.emergency_stop_drawdown or
                system_health < 0.3):
                return PortfolioMode.EMERGENCY
            
            # Conservative conditions
            elif (self.portfolio_metrics.current_drawdown > self.config.emergency_stop_drawdown * 0.7 or
                  self.portfolio_metrics.correlation_risk > self.config.max_correlation_threshold or
                  system_health < 0.6):
                return PortfolioMode.CONSERVATIVE
            
            # Normal active conditions
            elif (self.portfolio_metrics.current_drawdown < self.config.emergency_stop_drawdown * 0.3 and
                  system_health > 0.8):
                return PortfolioMode.ACTIVE
            
            # Default to current mode if no clear signal
            return self.current_mode
            
        except Exception as e:
            self.logger.error(f"Error determining portfolio mode: {e}")
            return self.current_mode
    
    async def _change_portfolio_mode(self, new_mode: PortfolioMode):
        """Change portfolio operating mode."""
        try:
            old_mode = self.current_mode
            self.current_mode = new_mode
            
            self.logger.info(f"Portfolio mode changed: {old_mode.value} -> {new_mode.value}")
            
            # Notify all strategies of mode change
            for strategy in self.strategies.values():
                if hasattr(strategy, 'on_portfolio_mode_change'):
                    await strategy.on_portfolio_mode_change(new_mode)
            
        except Exception as e:
            self.logger.error(f"Error changing portfolio mode: {e}")
    
    async def _apply_mode_adjustments(self):
        """Apply mode-specific portfolio adjustments."""
        try:
            if self.current_mode == PortfolioMode.EMERGENCY:
                # Emergency mode: Reduce all exposures dramatically
                for strategy in self.strategies.values():
                    if hasattr(strategy, 'set_risk_multiplier'):
                        await strategy.set_risk_multiplier(0.3)  # 30% of normal risk
            
            elif self.current_mode == PortfolioMode.CONSERVATIVE:
                # Conservative mode: Moderate risk reduction
                for strategy in self.strategies.values():
                    if hasattr(strategy, 'set_risk_multiplier'):
                        await strategy.set_risk_multiplier(0.7)  # 70% of normal risk
            
            elif self.current_mode == PortfolioMode.ACTIVE:
                # Active mode: Normal risk levels
                for strategy in self.strategies.values():
                    if hasattr(strategy, 'set_risk_multiplier'):
                        await strategy.set_risk_multiplier(1.0)  # 100% normal risk
            
        except Exception as e:
            self.logger.error(f"Error applying mode adjustments: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status with all advanced metrics."""
        try:
            # Get advanced component statuses
            system_status = self.system_monitor.get_system_status() if self.system_monitor else {}
            risk_dashboard = self.risk_manager.get_risk_dashboard() if self.risk_manager else {}
            
            return {
                "portfolio_summary": {
                    "is_active": self.is_active,
                    "is_emergency_stopped": self.is_emergency_stopped,
                    "current_mode": self.current_mode.value,
                    "total_capital": self.config.total_capital,
                    "net_asset_value": self.portfolio_metrics.net_asset_value,
                    "total_return": self.portfolio_metrics.portfolio_return,
                    "active_strategies": self.portfolio_metrics.active_strategies,
                    "total_strategies": len(self.strategies)
                },
                "performance_metrics": {
                    "total_pnl": self.portfolio_metrics.total_pnl,
                    "sharpe_ratio": self.portfolio_metrics.sharpe_ratio,
                    "sortino_ratio": self.portfolio_metrics.sortino_ratio,
                    "calmar_ratio": self.portfolio_metrics.calmar_ratio,
                    "max_drawdown": self.portfolio_metrics.max_drawdown,
                    "current_drawdown": self.portfolio_metrics.current_drawdown,
                    "volatility": self.portfolio_metrics.volatility,
                    "correlation_risk": self.portfolio_metrics.correlation_risk
                },
                "risk_metrics": {
                    "var_95": self.portfolio_metrics.var_95,
                    "var_99": self.portfolio_metrics.var_99,
                    "capital_utilization": self.portfolio_metrics.capital_utilization,
                    "total_positions": self.portfolio_metrics.total_positions
                },
                "advanced_analytics": {
                    "funding_analyzer_status": self.funding_analyzer.get_analysis_status() if self.funding_analyzer else {},
                    "volatility_tracker_status": self.volatility_tracker.get_tracker_status() if self.volatility_tracker else {},
                    "liquidity_analyzer_status": self.liquidity_analyzer.get_analyzer_status() if self.liquidity_analyzer else {},
                    "risk_summary": self.risk_metrics.get_risk_summary() if self.risk_metrics else {}
                },
                "system_health": system_status,
                "risk_dashboard": risk_dashboard,
                "strategies": {
                    strategy_id: strategy.get_status_summary()
                    for strategy_id, strategy in self.strategies.items()
                },
                "last_updated": self.portfolio_metrics.last_updated.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive portfolio report with all analytics."""
        try:
            report = {}
            
            # Performance attribution
            if self.performance_attributor:
                attribution_report = await self.performance_attributor.get_attribution_report()
                report["performance_attribution"] = attribution_report
            
            # Portfolio metrics
            if self.metrics_calculator:
                portfolio_state_dict = {
                    'net_asset_value': self.portfolio_metrics.net_asset_value,
                    'realized_pnl': self.portfolio_metrics.realized_pnl,
                    'unrealized_pnl': self.portfolio_metrics.unrealized_pnl,
                    'recent_trades': []  # Would collect from strategies
                }
                metrics = await self.metrics_calculator.calculate_portfolio_metrics(portfolio_state_dict)
                report["detailed_metrics"] = self.metrics_calculator.get_metrics_summary()
            
            # Risk analysis
            if self.risk_metrics:
                risk_summary = self.risk_metrics.get_risk_summary()
                report["risk_analysis"] = risk_summary
            
            # Data analysis insights
            data_insights = {}
            if self.funding_analyzer:
                opportunities = await self.funding_analyzer.get_arbitrage_opportunities()
                data_insights["funding_opportunities"] = len(opportunities)
            
            if self.volatility_tracker:
                for symbol in ["BTCUSDT", "ETHUSDT"]:
                    regime = await self.volatility_tracker.get_volatility_regime(symbol)
                    if regime:
                        data_insights[f"{symbol}_vol_regime"] = regime.regime
            
            report["data_insights"] = data_insights
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            # Stop portfolio if active
            if self.is_active:
                await self.stop_portfolio()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("PortfolioManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")