"""
Advanced Trading System - Risk Metrics
VaR/CVaR calculations and stress testing for portfolio risk management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class VarResults:
    """Value at Risk calculation results."""
    confidence_level: float
    
    # VaR measures
    historical_var: float = 0.0
    parametric_var: float = 0.0
    monte_carlo_var: float = 0.0
    
    # Conditional VaR (Expected Shortfall)
    conditional_var: float = 0.0
    
    # Model validation
    model_accuracy: float = 0.0  # Backtesting accuracy
    kupiec_test_pvalue: float = 0.0  # Kupiec test p-value
    
    # Time horizons
    one_day_var: float = 0.0
    one_week_var: float = 0.0
    one_month_var: float = 0.0
    
    last_calculated: datetime = field(default_factory=datetime.now)

@dataclass
class StressTestResult:
    """Stress test scenario result."""
    scenario_name: str
    scenario_description: str
    
    # Scenario parameters
    market_shock: float = 0.0  # Market return shock
    volatility_shock: float = 1.0  # Volatility multiplier
    correlation_shock: float = 0.0  # Correlation increase
    
    # Results
    portfolio_loss: float = 0.0
    worst_strategy_loss: float = 0.0
    time_to_recovery_days: float = 0.0
    
    # Component impacts
    strategy_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Risk measures under stress
    stressed_var_95: float = 0.0
    stressed_var_99: float = 0.0
    stressed_volatility: float = 0.0
    
    probability: float = 0.0  # Scenario probability
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CorrelationAnalysis:
    """Correlation analysis results."""
    # Correlation matrices
    current_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    historical_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stressed_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlation statistics
    average_correlation: float = 0.0
    maximum_correlation: float = 0.0
    correlation_stability: float = 0.0  # How stable correlations are
    
    # Regime analysis
    correlation_regime: str = "normal"  # "low", "normal", "high", "crisis"
    regime_probability: float = 0.0
    
    # Eigenvalue analysis
    principal_components: List[float] = field(default_factory=list)
    explained_variance_ratio: List[float] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)

class RiskMetricsCalculator:
    """
    Advanced risk metrics calculation engine.
    
    Provides comprehensive VaR calculations, stress testing, and correlation
    analysis for portfolio risk management.
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        """
        Initialize risk metrics calculator.
        
        Args:
            confidence_levels: VaR confidence levels (default: [0.95, 0.99])
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
        # Data storage
        self.var_results: Dict[float, VarResults] = {}
        self.stress_test_results: List[StressTestResult] = []
        self.correlation_analysis: Optional[CorrelationAnalysis] = None
        
        # Historical data
        self.returns_history: deque = deque(maxlen=2000)
        self.volatility_history: deque = deque(maxlen=500)
        self.correlation_history: deque = deque(maxlen=100)
        
        # Stress test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        # Settings
        self.min_observations = 100
        self.monte_carlo_simulations = 10000
        self.var_backtesting_window = 250
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("RiskMetricsCalculator initialized")
    
    def _initialize_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize stress test scenarios."""
        return [
            {
                "name": "covid_crash_2020",
                "description": "COVID-19 market crash (March 2020)",
                "market_shock": -0.40,
                "volatility_shock": 2.5,
                "correlation_shock": 0.3,
                "probability": 0.02
            },
            {
                "name": "luna_collapse_2022",
                "description": "Terra Luna ecosystem collapse",
                "market_shock": -0.30,
                "volatility_shock": 2.0,
                "correlation_shock": 0.25,
                "probability": 0.05
            },
            {
                "name": "crypto_winter_2018",
                "description": "Crypto bear market 2018",
                "market_shock": -0.80,
                "volatility_shock": 1.8,
                "correlation_shock": 0.4,
                "probability": 0.10
            },
            {
                "name": "flash_crash",
                "description": "Sudden liquidity crisis",
                "market_shock": -0.20,
                "volatility_shock": 3.0,
                "correlation_shock": 0.1,
                "probability": 0.15
            },
            {
                "name": "regulation_shock",
                "description": "Major regulatory crackdown",
                "market_shock": -0.50,
                "volatility_shock": 1.5,
                "correlation_shock": 0.2,
                "probability": 0.08
            },
            {
                "name": "exchange_hack",
                "description": "Major exchange security breach",
                "market_shock": -0.15,
                "volatility_shock": 1.8,
                "correlation_shock": 0.15,
                "probability": 0.10
            },
            {
                "name": "defi_exploit",
                "description": "Large DeFi protocol exploit",
                "market_shock": -0.25,
                "volatility_shock": 2.2,
                "correlation_shock": 0.2,
                "probability": 0.12
            }
        ]
    
    async def calculate_portfolio_var(self, returns_data: List[float], 
                                    portfolio_value: float) -> Dict[float, VarResults]:
        """
        Calculate Value at Risk using multiple methodologies.
        
        Args:
            returns_data: Historical returns data
            portfolio_value: Current portfolio value
            
        Returns:
            Dict mapping confidence levels to VaR results
        """
        try:
            self.returns_history.extend(returns_data)
            
            if len(self.returns_history) < self.min_observations:
                self.logger.warning(f"Insufficient data for VaR calculation: {len(self.returns_history)} < {self.min_observations}")
                return {}
            
            results = {}
            returns_array = np.array(list(self.returns_history))
            
            for confidence_level in self.confidence_levels:
                var_result = VarResults(confidence_level=confidence_level)
                
                # Historical VaR
                var_result.historical_var = await self._calculate_historical_var(
                    returns_array, confidence_level, portfolio_value
                )
                
                # Parametric VaR
                var_result.parametric_var = await self._calculate_parametric_var(
                    returns_array, confidence_level, portfolio_value
                )
                
                # Monte Carlo VaR
                var_result.monte_carlo_var = await self._calculate_monte_carlo_var(
                    returns_array, confidence_level, portfolio_value
                )
                
                # Conditional VaR
                var_result.conditional_var = await self._calculate_conditional_var(
                    returns_array, confidence_level, portfolio_value
                )
                
                # Multi-horizon VaR
                await self._calculate_multi_horizon_var(var_result, returns_array, confidence_level, portfolio_value)
                
                # Backtesting
                await self._backtest_var_model(var_result, returns_array, portfolio_value)
                
                results[confidence_level] = var_result
            
            self.var_results = results
            
            self.logger.debug(f"VaR calculated for {len(results)} confidence levels")
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return {}
    
    async def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float, 
                                      portfolio_value: float) -> float:
        """Calculate historical simulation VaR."""
        try:
            # Sort returns and find percentile
            sorted_returns = np.sort(returns)
            percentile_index = int((1 - confidence_level) * len(sorted_returns))
            
            if percentile_index >= len(sorted_returns):
                percentile_index = len(sorted_returns) - 1
            
            var_return = sorted_returns[percentile_index]
            var_amount = abs(var_return * portfolio_value)
            
            return var_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating historical VaR: {e}")
            return 0.0
    
    async def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float, 
                                      portfolio_value: float) -> float:
        """Calculate parametric (normal distribution) VaR."""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # VaR calculation
            var_return = mean_return + z_score * std_return
            var_amount = abs(var_return * portfolio_value)
            
            return var_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating parametric VaR: {e}")
            return 0.0
    
    async def _calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float, 
                                       portfolio_value: float) -> float:
        """Calculate Monte Carlo simulation VaR."""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate Monte Carlo scenarios
            simulated_returns = np.random.normal(
                mean_return, std_return, self.monte_carlo_simulations
            )
            
            # Calculate VaR from simulated returns
            sorted_simulated = np.sort(simulated_returns)
            percentile_index = int((1 - confidence_level) * len(sorted_simulated))
            
            if percentile_index >= len(sorted_simulated):
                percentile_index = len(sorted_simulated) - 1
            
            var_return = sorted_simulated[percentile_index]
            var_amount = abs(var_return * portfolio_value)
            
            return var_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0
    
    async def _calculate_conditional_var(self, returns: np.ndarray, confidence_level: float, 
                                       portfolio_value: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        try:
            # Sort returns
            sorted_returns = np.sort(returns)
            percentile_index = int((1 - confidence_level) * len(sorted_returns))
            
            # Calculate expected shortfall (average of tail losses)
            tail_returns = sorted_returns[:percentile_index + 1]
            if len(tail_returns) > 0:
                expected_shortfall = np.mean(tail_returns)
                cvar_amount = abs(expected_shortfall * portfolio_value)
                return cvar_amount
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating conditional VaR: {e}")
            return 0.0
    
    async def _calculate_multi_horizon_var(self, var_result: VarResults, returns: np.ndarray, 
                                         confidence_level: float, portfolio_value: float):
        """Calculate VaR for multiple time horizons."""
        try:
            daily_var = var_result.historical_var / portfolio_value  # As percentage
            
            # Scale VaR for different horizons (square root of time rule)
            var_result.one_day_var = daily_var * portfolio_value
            var_result.one_week_var = daily_var * np.sqrt(7) * portfolio_value
            var_result.one_month_var = daily_var * np.sqrt(30) * portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-horizon VaR: {e}")
    
    async def _backtest_var_model(self, var_result: VarResults, returns: np.ndarray, 
                                portfolio_value: float):
        """Backtest VaR model accuracy."""
        try:
            if len(returns) < self.var_backtesting_window:
                return
            
            # Use rolling window for backtesting
            violations = 0
            total_tests = 0
            
            window_size = min(250, len(returns) // 2)  # Use half data for estimation
            
            for i in range(window_size, len(returns)):
                # Estimate VaR using historical data up to point i
                estimation_window = returns[i-window_size:i]
                var_threshold = np.percentile(estimation_window, (1 - var_result.confidence_level) * 100)
                
                # Check if next return violates VaR
                actual_return = returns[i]
                if actual_return < var_threshold:
                    violations += 1
                
                total_tests += 1
            
            # Calculate accuracy metrics
            if total_tests > 0:
                violation_rate = violations / total_tests
                expected_rate = 1 - var_result.confidence_level
                var_result.model_accuracy = 1 - abs(violation_rate - expected_rate) / expected_rate
                
                # Kupiec test for unconditional coverage
                if violations > 0 and total_tests > 0:
                    likelihood_ratio = -2 * (
                        violations * np.log(expected_rate) + 
                        (total_tests - violations) * np.log(1 - expected_rate) -
                        violations * np.log(violation_rate) - 
                        (total_tests - violations) * np.log(1 - violation_rate)
                    )
                    var_result.kupiec_test_pvalue = 1 - stats.chi2.cdf(likelihood_ratio, 1)
            
        except Exception as e:
            self.logger.error(f"Error in VaR backtesting: {e}")
    
    async def run_stress_tests(self, strategies: Dict[str, Any], 
                             portfolio_value: float) -> List[StressTestResult]:
        """
        Run comprehensive stress testing scenarios.
        
        Args:
            strategies: Dictionary of active strategies
            portfolio_value: Current portfolio value
            
        Returns:
            List of stress test results
        """
        try:
            results = []
            
            for scenario in self.stress_scenarios:
                result = await self._run_single_stress_test(
                    scenario, strategies, portfolio_value
                )
                if result:
                    results.append(result)
            
            # Add custom scenarios
            custom_scenarios = await self._generate_custom_scenarios(strategies)
            for scenario in custom_scenarios:
                result = await self._run_single_stress_test(
                    scenario, strategies, portfolio_value
                )
                if result:
                    results.append(result)
            
            self.stress_test_results = results
            
            self.logger.info(f"Completed {len(results)} stress test scenarios")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress tests: {e}")
            return []
    
    async def _run_single_stress_test(self, scenario: Dict[str, Any], 
                                    strategies: Dict[str, Any], 
                                    portfolio_value: float) -> Optional[StressTestResult]:
        """Run a single stress test scenario."""
        try:
            result = StressTestResult(
                scenario_name=scenario["name"],
                scenario_description=scenario["description"],
                market_shock=scenario["market_shock"],
                volatility_shock=scenario["volatility_shock"],
                correlation_shock=scenario["correlation_shock"],
                probability=scenario["probability"]
            )
            
            total_loss = 0.0
            worst_strategy_loss = 0.0
            
            # Apply scenario to each strategy
            for strategy_id, strategy in strategies.items():
                strategy_loss = await self._calculate_strategy_stress_impact(
                    strategy, scenario
                )
                
                result.strategy_impacts[strategy_id] = strategy_loss
                total_loss += strategy_loss
                worst_strategy_loss = min(worst_strategy_loss, strategy_loss)
            
            result.portfolio_loss = total_loss
            result.worst_strategy_loss = worst_strategy_loss
            
            # Estimate recovery time
            result.time_to_recovery_days = await self._estimate_recovery_time(
                total_loss, portfolio_value, scenario
            )
            
            # Calculate stressed risk measures
            await self._calculate_stressed_risk_measures(result, scenario)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running stress test {scenario.get('name', 'unknown')}: {e}")
            return None
    
    async def _calculate_strategy_stress_impact(self, strategy, scenario: Dict[str, Any]) -> float:
        """Calculate stress impact on individual strategy."""
        try:
            # Get strategy characteristics
            strategy_type = getattr(strategy, 'strategy_type', 'unknown')
            market_exposure = getattr(strategy, 'market_exposure', 0.0)
            volatility_exposure = getattr(strategy, 'volatility_exposure', 1.0)
            
            # Base impact from market shock
            market_impact = scenario["market_shock"] * market_exposure
            
            # Volatility impact
            vol_impact = (scenario["volatility_shock"] - 1.0) * volatility_exposure * 0.1
            
            # Strategy-specific adjustments
            if strategy_type.lower() in ['grid', 'market_making']:
                # Grid strategies benefit from volatility but hurt from directional moves
                market_impact *= 0.5  # Reduced directional impact
                vol_impact *= -0.3    # Benefit from volatility (negative = gain)
            elif strategy_type.lower() in ['arbitrage', 'basis']:
                # Arbitrage strategies relatively protected but affected by correlation changes
                market_impact *= 0.2
                correlation_impact = scenario["correlation_shock"] * 0.05
                market_impact += correlation_impact
            
            total_impact = market_impact + vol_impact
            
            # Convert to absolute loss amount
            strategy_value = getattr(strategy, 'capital_allocation', 0.0)
            return total_impact * strategy_value
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy stress impact: {e}")
            return 0.0
    
    async def _estimate_recovery_time(self, loss_amount: float, portfolio_value: float, 
                                    scenario: Dict[str, Any]) -> float:
        """Estimate time to recover from stress scenario."""
        try:
            if portfolio_value <= 0 or loss_amount >= 0:
                return 0.0
            
            loss_percentage = abs(loss_amount) / portfolio_value
            
            # Base recovery time based on loss severity
            if loss_percentage < 0.05:
                base_recovery_days = 30
            elif loss_percentage < 0.10:
                base_recovery_days = 60
            elif loss_percentage < 0.20:
                base_recovery_days = 120
            elif loss_percentage < 0.40:
                base_recovery_days = 365
            else:
                base_recovery_days = 730  # 2 years for severe losses
            
            # Adjust based on scenario characteristics
            volatility_factor = scenario["volatility_shock"]
            correlation_factor = 1 + scenario["correlation_shock"]
            
            adjusted_recovery = base_recovery_days * volatility_factor * correlation_factor
            
            return min(adjusted_recovery, 1095)  # Cap at 3 years
            
        except Exception as e:
            self.logger.error(f"Error estimating recovery time: {e}")
            return 365.0  # Default 1 year
    
    async def _calculate_stressed_risk_measures(self, result: StressTestResult, 
                                              scenario: Dict[str, Any]):
        """Calculate risk measures under stress conditions."""
        try:
            if len(self.returns_history) < self.min_observations:
                return
            
            returns_array = np.array(list(self.returns_history))
            
            # Apply stress to historical returns
            stressed_returns = returns_array * (1 + scenario["market_shock"])
            stressed_returns *= scenario["volatility_shock"]
            
            # Calculate stressed VaR
            result.stressed_var_95 = abs(np.percentile(stressed_returns, 5))
            result.stressed_var_99 = abs(np.percentile(stressed_returns, 1))
            result.stressed_volatility = np.std(stressed_returns)
            
        except Exception as e:
            self.logger.error(f"Error calculating stressed risk measures: {e}")
    
    async def _generate_custom_scenarios(self, strategies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate custom stress scenarios based on current portfolio."""
        try:
            custom_scenarios = []
            
            # Scenario based on current volatility regime
            if len(self.returns_history) > 50:
                recent_vol = np.std(list(self.returns_history)[-30:])
                if recent_vol > 0.05:  # High volatility regime
                    custom_scenarios.append({
                        "name": "volatility_spike",
                        "description": "Current high volatility continues",
                        "market_shock": -0.10,
                        "volatility_shock": 2.0,
                        "correlation_shock": 0.2,
                        "probability": 0.20
                    })
            
            # Strategy-specific scenarios
            strategy_types = [getattr(s, 'strategy_type', 'unknown') for s in strategies.values()]
            
            if 'grid' in [st.lower() for st in strategy_types]:
                custom_scenarios.append({
                    "name": "low_volatility_regime",
                    "description": "Sustained low volatility hurts grid strategies",
                    "market_shock": 0.0,
                    "volatility_shock": 0.3,
                    "correlation_shock": -0.1,
                    "probability": 0.15
                })
            
            return custom_scenarios
            
        except Exception as e:
            self.logger.error(f"Error generating custom scenarios: {e}")
            return []
    
    async def analyze_correlations(self, strategies: Dict[str, Any]) -> CorrelationAnalysis:
        """
        Analyze correlation structure and regime changes.
        
        Args:
            strategies: Dictionary of active strategies
            
        Returns:
            CorrelationAnalysis with correlation metrics
        """
        try:
            analysis = CorrelationAnalysis()
            
            if len(strategies) < 2:
                return analysis
            
            # Collect strategy returns
            strategy_returns = {}
            min_length = float('inf')
            
            for strategy_id, strategy in strategies.items():
                if hasattr(strategy, 'pnl_history') and len(strategy.pnl_history) > 1:
                    returns = np.diff(strategy.pnl_history) / np.array(strategy.pnl_history[:-1])
                    strategy_returns[strategy_id] = returns[-200:]  # Last 200 observations
                    min_length = min(min_length, len(strategy_returns[strategy_id]))
            
            if len(strategy_returns) < 2 or min_length < 30:
                return analysis
            
            # Align return series
            aligned_returns = {}
            for strategy_id, returns in strategy_returns.items():
                aligned_returns[strategy_id] = returns[-min_length:]
            
            # Calculate current correlations
            strategy_ids = list(aligned_returns.keys())
            returns_matrix = np.array([aligned_returns[sid] for sid in strategy_ids])
            current_corr_matrix = np.corrcoef(returns_matrix)
            
            # Store correlations
            for i, sid1 in enumerate(strategy_ids):
                analysis.current_correlations[sid1] = {}
                for j, sid2 in enumerate(strategy_ids):
                    analysis.current_correlations[sid1][sid2] = current_corr_matrix[i, j]
            
            # Calculate correlation statistics
            off_diagonal = current_corr_matrix[~np.eye(current_corr_matrix.shape[0], dtype=bool)]
            analysis.average_correlation = np.mean(np.abs(off_diagonal))
            analysis.maximum_correlation = np.max(np.abs(off_diagonal))
            
            # Historical correlations (rolling window)
            if min_length > 60:
                historical_corrs = []
                window_size = 30
                
                for start in range(0, min_length - window_size, 10):
                    window_returns = {}
                    for sid in strategy_ids:
                        window_returns[sid] = aligned_returns[sid][start:start + window_size]
                    
                    window_matrix = np.array([window_returns[sid] for sid in strategy_ids])
                    window_corr = np.corrcoef(window_matrix)
                    historical_corrs.append(window_corr)
                
                if historical_corrs:
                    # Calculate correlation stability
                    corr_changes = []
                    for i in range(1, len(historical_corrs)):
                        diff = np.abs(historical_corrs[i] - historical_corrs[i-1])
                        corr_changes.append(np.mean(diff[~np.eye(diff.shape[0], dtype=bool)]))
                    
                    if corr_changes:
                        analysis.correlation_stability = 1.0 - np.mean(corr_changes)
            
            # Determine correlation regime
            if analysis.average_correlation < 0.3:
                analysis.correlation_regime = "low"
                analysis.regime_probability = 0.8
            elif analysis.average_correlation < 0.6:
                analysis.correlation_regime = "normal"
                analysis.regime_probability = 0.9
            elif analysis.average_correlation < 0.8:
                analysis.correlation_regime = "high"
                analysis.regime_probability = 0.7
            else:
                analysis.correlation_regime = "crisis"
                analysis.regime_probability = 0.6
            
            # Principal component analysis
            try:
                eigenvalues, eigenvectors = np.linalg.eig(current_corr_matrix)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                
                analysis.principal_components = eigenvalues.tolist()
                total_variance = np.sum(eigenvalues)
                if total_variance > 0:
                    analysis.explained_variance_ratio = (eigenvalues / total_variance).tolist()
            except Exception as pca_error:
                self.logger.warning(f"PCA analysis failed: {pca_error}")
            
            # Stressed correlations (increase all correlations)
            stressed_corr_matrix = current_corr_matrix.copy()
            for i in range(len(strategy_ids)):
                for j in range(len(strategy_ids)):
                    if i != j:
                        # Increase correlation towards 1 under stress
                        current_corr = stressed_corr_matrix[i, j]
                        stressed_corr_matrix[i, j] = current_corr + (1 - abs(current_corr)) * 0.3
            
            for i, sid1 in enumerate(strategy_ids):
                analysis.stressed_correlations[sid1] = {}
                for j, sid2 in enumerate(strategy_ids):
                    analysis.stressed_correlations[sid1][sid2] = stressed_corr_matrix[i, j]
            
            analysis.last_updated = datetime.now()
            self.correlation_analysis = analysis
            
            self.logger.debug(f"Correlation analysis completed: avg={analysis.average_correlation:.2f}, regime={analysis.correlation_regime}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return CorrelationAnalysis()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        summary = {
            "var_metrics": {},
            "stress_test_summary": {},
            "correlation_summary": {},
            "data_quality": {
                "returns_observations": len(self.returns_history),
                "sufficient_data": len(self.returns_history) >= self.min_observations
            }
        }
        
        # VaR metrics summary
        if self.var_results:
            for confidence_level, var_result in self.var_results.items():
                summary["var_metrics"][f"var_{int(confidence_level*100)}"] = {
                    "historical_var": var_result.historical_var,
                    "parametric_var": var_result.parametric_var,
                    "conditional_var": var_result.conditional_var,
                    "model_accuracy": var_result.model_accuracy
                }
        
        # Stress test summary
        if self.stress_test_results:
            worst_case = min(self.stress_test_results, key=lambda x: x.portfolio_loss)
            most_likely = max(self.stress_test_results, key=lambda x: x.probability)
            
            summary["stress_test_summary"] = {
                "scenarios_tested": len(self.stress_test_results),
                "worst_case_loss": worst_case.portfolio_loss,
                "worst_case_scenario": worst_case.scenario_name,
                "most_likely_loss": most_likely.portfolio_loss,
                "most_likely_scenario": most_likely.scenario_name,
                "average_recovery_time": np.mean([r.time_to_recovery_days for r in self.stress_test_results])
            }
        
        # Correlation summary
        if self.correlation_analysis:
            summary["correlation_summary"] = {
                "average_correlation": self.correlation_analysis.average_correlation,
                "maximum_correlation": self.correlation_analysis.maximum_correlation,
                "correlation_regime": self.correlation_analysis.correlation_regime,
                "correlation_stability": self.correlation_analysis.correlation_stability
            }
        
        return summary
    
    def clear_history(self):
        """Clear risk metrics history."""
        self.var_results.clear()
        self.stress_test_results.clear()
        self.correlation_analysis = None
        self.returns_history.clear()
        self.volatility_history.clear()
        self.correlation_history.clear()
        self.logger.info("Risk metrics history cleared")