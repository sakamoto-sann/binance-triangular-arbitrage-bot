"""
Advanced Trading System - Performance Attribution
Real-time P&L breakdown and attribution analysis for portfolio strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StrategyAttribution:
    """Performance attribution for a single strategy."""
    strategy_id: str
    strategy_type: str
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # P&L source breakdown
    grid_pnl: float = 0.0
    funding_pnl: float = 0.0
    basis_pnl: float = 0.0
    spread_pnl: float = 0.0
    vol_timing_pnl: float = 0.0
    carry_pnl: float = 0.0
    
    # Cost breakdown
    transaction_costs: float = 0.0
    funding_costs: float = 0.0
    slippage_costs: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    information_ratio: float = 0.0
    contribution_to_portfolio_return: float = 0.0
    contribution_to_portfolio_risk: float = 0.0
    
    # Activity metrics
    total_trades: int = 0
    turnover: float = 0.0
    active_time_pct: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioAttribution:
    """Portfolio-level performance attribution."""
    total_portfolio_pnl: float = 0.0
    strategy_attributions: Dict[str, StrategyAttribution] = field(default_factory=dict)
    
    # Portfolio-level breakdowns
    total_alpha: float = 0.0  # Strategy selection
    total_beta: float = 0.0   # Market exposure
    interaction_effects: float = 0.0  # Strategy interactions
    
    # Diversification metrics
    diversification_ratio: float = 0.0
    effective_strategies: float = 0.0  # Equivalent number of independent strategies
    concentration_index: float = 0.0  # Herfindahl index for strategy concentration
    
    # Attribution accuracy
    unexplained_pnl: float = 0.0
    attribution_quality: float = 1.0  # How well attribution explains total P&L
    
    last_updated: datetime = field(default_factory=datetime.now)

class PerformanceAttributor:
    """
    Real-time performance attribution engine.
    
    Breaks down portfolio P&L by strategy and source, providing detailed
    insights into what's driving performance and risk.
    """
    
    def __init__(self):
        """Initialize the performance attributor."""
        self.attribution_history: List[PortfolioAttribution] = []
        self.pnl_components: Dict[str, List[float]] = defaultdict(list)
        self.benchmark_returns: List[float] = []
        
        # Attribution settings
        self.lookback_periods = 100  # Number of periods for rolling calculations
        self.min_periods_for_metrics = 30  # Minimum periods needed for ratios
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("PerformanceAttributor initialized")
    
    async def calculate_attribution(self, strategies: Dict[str, Any]) -> PortfolioAttribution:
        """
        Calculate comprehensive performance attribution.
        
        Args:
            strategies: Dictionary of active strategies
            
        Returns:
            PortfolioAttribution with detailed breakdown
        """
        try:
            attribution = PortfolioAttribution()
            
            # Calculate strategy-level attributions
            total_portfolio_pnl = 0.0
            for strategy_id, strategy in strategies.items():
                strategy_attr = await self._calculate_strategy_attribution(strategy)
                attribution.strategy_attributions[strategy_id] = strategy_attr
                total_portfolio_pnl += strategy_attr.total_pnl
            
            attribution.total_portfolio_pnl = total_portfolio_pnl
            
            # Calculate portfolio-level metrics
            await self._calculate_portfolio_level_metrics(attribution, strategies)
            
            # Calculate diversification metrics
            await self._calculate_diversification_metrics(attribution, strategies)
            
            # Store in history
            self.attribution_history.append(attribution)
            if len(self.attribution_history) > self.lookback_periods:
                self.attribution_history = self.attribution_history[-self.lookback_periods:]
            
            attribution.last_updated = datetime.now()
            
            self.logger.debug(f"Attribution calculated: ${total_portfolio_pnl:,.2f} total P&L")
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating attribution: {e}")
            return PortfolioAttribution()
    
    async def _calculate_strategy_attribution(self, strategy) -> StrategyAttribution:
        """Calculate attribution for a single strategy."""
        try:
            attr = StrategyAttribution(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type.value,
                total_pnl=strategy.metrics.total_pnl,
                realized_pnl=strategy.metrics.realized_pnl,
                unrealized_pnl=strategy.metrics.unrealized_pnl,
                sharpe_ratio=strategy.metrics.sharpe_ratio,
                total_trades=strategy.metrics.total_trades
            )
            
            # Get strategy-specific metrics
            specific_metrics = await strategy.get_strategy_specific_metrics()
            
            # Map strategy-specific P&L components
            attr.grid_pnl = specific_metrics.get('grid_pnl', 0.0)
            attr.funding_pnl = specific_metrics.get('funding_pnl', 0.0)
            attr.basis_pnl = specific_metrics.get('basis_pnl', 0.0)
            attr.spread_pnl = specific_metrics.get('spread_pnl', 0.0)
            attr.vol_timing_pnl = specific_metrics.get('vol_timing_pnl', 0.0)
            attr.carry_pnl = specific_metrics.get('carry_pnl', 0.0)
            
            # Map cost components
            attr.transaction_costs = specific_metrics.get('transaction_costs', 0.0)
            attr.funding_costs = specific_metrics.get('funding_costs', 0.0)
            attr.slippage_costs = specific_metrics.get('slippage_costs', 0.0)
            
            # Calculate activity metrics
            attr.turnover = specific_metrics.get('turnover', 0.0)
            attr.active_time_pct = self._calculate_active_time_pct(strategy)
            
            # Calculate risk-adjusted metrics
            if len(strategy.pnl_history) > self.min_periods_for_metrics:
                attr.information_ratio = self._calculate_information_ratio(strategy.pnl_history)
            
            return attr
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy attribution for {strategy.strategy_id}: {e}")
            return StrategyAttribution(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type.value
            )
    
    def _calculate_active_time_pct(self, strategy) -> float:
        """Calculate percentage of time strategy was actively trading."""
        try:
            # This would be based on strategy's trading history
            # For now, return a simple estimate based on position count
            if hasattr(strategy, 'position_history') and strategy.position_history:
                active_periods = sum(1 for pos in strategy.position_history[-100:] 
                                   if any(abs(p.get('net_exposure', 0)) > 100 for p in pos.values()))
                return active_periods / min(len(strategy.position_history), 100)
            return 0.5  # Default estimate
        except:
            return 0.5
    
    def _calculate_information_ratio(self, pnl_history: List[float]) -> float:
        """Calculate information ratio vs. benchmark."""
        try:
            if len(pnl_history) < 2:
                return 0.0
            
            # Convert P&L to returns
            returns = []
            for i in range(1, len(pnl_history)):
                if pnl_history[i-1] != 0:
                    ret = (pnl_history[i] - pnl_history[i-1]) / abs(pnl_history[i-1])
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            
            # Use zero benchmark (risk-free rate) for market-neutral strategies
            benchmark_return = 0.0
            excess_returns = returns_array - benchmark_return
            
            if np.std(excess_returns) > 0:
                return np.mean(excess_returns) / np.std(excess_returns)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    async def _calculate_portfolio_level_metrics(self, attribution: PortfolioAttribution, strategies: Dict[str, Any]):
        """Calculate portfolio-level attribution metrics."""
        try:
            if not attribution.strategy_attributions:
                return
            
            # Calculate total alpha (strategy selection effect)
            total_strategy_pnl = sum(attr.total_pnl for attr in attribution.strategy_attributions.values())
            attribution.total_alpha = total_strategy_pnl
            
            # Beta would be market exposure - for market-neutral strategies this should be ~0
            attribution.total_beta = 0.0  # Market-neutral by design
            
            # Calculate interaction effects (correlation benefits/costs)
            if len(attribution.strategy_attributions) > 1:
                attribution.interaction_effects = self._calculate_interaction_effects(strategies)
            
            # Attribution quality check
            attribution.unexplained_pnl = attribution.total_portfolio_pnl - total_strategy_pnl
            
            if attribution.total_portfolio_pnl != 0:
                attribution.attribution_quality = 1.0 - abs(attribution.unexplained_pnl) / abs(attribution.total_portfolio_pnl)
            else:
                attribution.attribution_quality = 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio-level metrics: {e}")
    
    def _calculate_interaction_effects(self, strategies: Dict[str, Any]) -> float:
        """Calculate P&L from strategy interactions (diversification effects)."""
        try:
            # This is a simplified calculation
            # In practice, would need more sophisticated correlation analysis
            
            if len(strategies) < 2:
                return 0.0
            
            # Estimate based on correlation reduction benefits
            total_individual_variance = 0.0
            for strategy in strategies.values():
                if len(strategy.pnl_history) > 1:
                    returns = np.diff(strategy.pnl_history)
                    total_individual_variance += np.var(returns)
            
            # Assume some diversification benefit
            diversification_benefit = total_individual_variance * 0.1  # 10% variance reduction
            
            # Convert to approximate P&L benefit
            return diversification_benefit * 0.01  # Rough approximation
            
        except Exception as e:
            self.logger.error(f"Error calculating interaction effects: {e}")
            return 0.0
    
    async def _calculate_diversification_metrics(self, attribution: PortfolioAttribution, strategies: Dict[str, Any]):
        """Calculate diversification and concentration metrics."""
        try:
            if not attribution.strategy_attributions:
                return
            
            # Calculate concentration index (Herfindahl)
            total_pnl = max(abs(attribution.total_portfolio_pnl), 1.0)  # Avoid division by zero
            weights_squared = sum(
                (attr.total_pnl / total_pnl) ** 2 
                for attr in attribution.strategy_attributions.values()
            )
            attribution.concentration_index = weights_squared
            
            # Effective number of strategies
            if weights_squared > 0:
                attribution.effective_strategies = 1.0 / weights_squared
            else:
                attribution.effective_strategies = len(attribution.strategy_attributions)
            
            # Diversification ratio (would need return correlations for proper calculation)
            if len(attribution.strategy_attributions) > 1:
                attribution.diversification_ratio = min(attribution.effective_strategies / len(attribution.strategy_attributions), 1.0)
            else:
                attribution.diversification_ratio = 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification metrics: {e}")
    
    async def get_attribution_report(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive attribution report.
        
        Args:
            period_days: Number of days to include in report
            
        Returns:
            Dict with detailed attribution analysis
        """
        try:
            if not self.attribution_history:
                return {"error": "No attribution history available"}
            
            # Get recent attribution data
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_attributions = [
                attr for attr in self.attribution_history 
                if attr.last_updated >= cutoff_date
            ]
            
            if not recent_attributions:
                recent_attributions = self.attribution_history[-1:]
            
            # Calculate period aggregates
            report = {
                "period_summary": {
                    "start_date": recent_attributions[0].last_updated.isoformat(),
                    "end_date": recent_attributions[-1].last_updated.isoformat(),
                    "total_portfolio_pnl": recent_attributions[-1].total_portfolio_pnl,
                    "attribution_quality": np.mean([attr.attribution_quality for attr in recent_attributions])
                },
                "strategy_breakdown": {},
                "pnl_sources": {},
                "cost_analysis": {},
                "diversification_analysis": {
                    "effective_strategies": recent_attributions[-1].effective_strategies,
                    "concentration_index": recent_attributions[-1].concentration_index,
                    "diversification_ratio": recent_attributions[-1].diversification_ratio
                },
                "performance_drivers": await self._identify_performance_drivers(recent_attributions)
            }
            
            # Strategy breakdown
            latest_attribution = recent_attributions[-1]
            for strategy_id, strategy_attr in latest_attribution.strategy_attributions.items():
                report["strategy_breakdown"][strategy_id] = {
                    "total_pnl": strategy_attr.total_pnl,
                    "pnl_contribution_pct": (strategy_attr.total_pnl / max(abs(latest_attribution.total_portfolio_pnl), 1)) * 100,
                    "sharpe_ratio": strategy_attr.sharpe_ratio,
                    "information_ratio": strategy_attr.information_ratio,
                    "total_trades": strategy_attr.total_trades,
                    "active_time_pct": strategy_attr.active_time_pct
                }
            
            # P&L sources aggregate
            total_grid = sum(attr.grid_pnl for attr in latest_attribution.strategy_attributions.values())
            total_funding = sum(attr.funding_pnl for attr in latest_attribution.strategy_attributions.values())
            total_basis = sum(attr.basis_pnl for attr in latest_attribution.strategy_attributions.values())
            total_spread = sum(attr.spread_pnl for attr in latest_attribution.strategy_attributions.values())
            
            report["pnl_sources"] = {
                "grid_trading": total_grid,
                "funding_income": total_funding,
                "basis_trading": total_basis,
                "spread_capture": total_spread
            }
            
            # Cost analysis
            total_txn_costs = sum(attr.transaction_costs for attr in latest_attribution.strategy_attributions.values())
            total_funding_costs = sum(attr.funding_costs for attr in latest_attribution.strategy_attributions.values())
            total_slippage = sum(attr.slippage_costs for attr in latest_attribution.strategy_attributions.values())
            
            report["cost_analysis"] = {
                "transaction_costs": total_txn_costs,
                "funding_costs": total_funding_costs,
                "slippage_costs": total_slippage,
                "total_costs": total_txn_costs + total_funding_costs + total_slippage
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating attribution report: {e}")
            return {"error": f"Failed to generate report: {e}"}
    
    async def _identify_performance_drivers(self, attributions: List[PortfolioAttribution]) -> Dict[str, Any]:
        """Identify key performance drivers over the period."""
        try:
            if not attributions:
                return {}
            
            # Find best and worst performing strategies
            latest = attributions[-1]
            strategy_performance = {
                strategy_id: attr.total_pnl 
                for strategy_id, attr in latest.strategy_attributions.items()
            }
            
            if strategy_performance:
                best_strategy = max(strategy_performance, key=strategy_performance.get)
                worst_strategy = min(strategy_performance, key=strategy_performance.get)
            else:
                best_strategy = worst_strategy = None
            
            # Identify dominant P&L sources
            pnl_sources = {}
            for attr in latest.strategy_attributions.values():
                pnl_sources['grid'] = pnl_sources.get('grid', 0) + attr.grid_pnl
                pnl_sources['funding'] = pnl_sources.get('funding', 0) + attr.funding_pnl
                pnl_sources['basis'] = pnl_sources.get('basis', 0) + attr.basis_pnl
                pnl_sources['spread'] = pnl_sources.get('spread', 0) + attr.spread_pnl
            
            dominant_source = max(pnl_sources, key=lambda k: abs(pnl_sources[k])) if pnl_sources else None
            
            return {
                "best_performing_strategy": best_strategy,
                "worst_performing_strategy": worst_strategy,
                "dominant_pnl_source": dominant_source,
                "diversification_score": latest.diversification_ratio,
                "attribution_confidence": latest.attribution_quality
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying performance drivers: {e}")
            return {}
    
    async def get_real_time_attribution(self) -> Optional[PortfolioAttribution]:
        """Get the most recent attribution data."""
        if self.attribution_history:
            return self.attribution_history[-1]
        return None
    
    def clear_history(self):
        """Clear attribution history (useful for testing or memory management)."""
        self.attribution_history.clear()
        self.pnl_components.clear()
        self.logger.info("Attribution history cleared")