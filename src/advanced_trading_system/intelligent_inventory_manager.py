"""
Intelligent Inventory Management System with Kelly Criterion Optimization
Dynamic position sizing based on current inventory, market conditions, and risk management
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
import math

@dataclass
class PositionInfo:
    """Position information data class"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    position_value: float
    weight: float
    correlation_risk: float

@dataclass
class InventoryMetrics:
    """Inventory metrics data class"""
    total_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    concentration_risk: float
    correlation_risk: float
    inventory_turnover: float
    sharpe_ratio: float
    max_drawdown: float

@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    recommended_size: float
    max_size: float
    min_size: float
    kelly_fraction: float
    risk_adjusted_size: float
    confidence: float
    reasoning: str

class IntelligentInventoryManager:
    """
    Professional inventory management system using Kelly Criterion optimization,
    volatility-adjusted sizing, correlation analysis, and exposure limits.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the intelligent inventory manager"""
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Risk management parameters
        self.max_inventory_ratio = self.config.get('max_inventory_ratio', 0.3)  # 30% of total capital
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.1)  # 10% deviation
        self.max_single_position = self.config.get('max_single_position', 0.1)  # 10% per position
        
        # Kelly Criterion parameters
        self.kelly_lookback = self.config.get('kelly_lookback', 100)  # trades for Kelly calculation
        self.kelly_multiplier = self.config.get('kelly_multiplier', 0.25)  # Conservative Kelly
        self.min_kelly_fraction = self.config.get('min_kelly_fraction', 0.01)
        self.max_kelly_fraction = self.config.get('max_kelly_fraction', 0.25)
        
        # Portfolio tracking
        self.total_capital = self.config.get('total_capital', 1000000)  # $1M default
        self.positions = {}  # symbol -> PositionInfo
        self.trade_history = []  # Historical trades for Kelly calculation
        self.performance_history = []  # Historical performance metrics
        
        # Correlation matrix
        self.correlation_matrix = {}
        self.correlation_update_frequency = 3600  # Update hourly
        self.last_correlation_update = None

    def calculate_optimal_position_size(self, symbol: str, signal_strength: float, 
                                      market_volatility: float, current_inventory: float,
                                      signal_confidence: float = 0.5) -> PositionSizing:
        """
        Calculate position size using advanced risk management including Kelly Criterion optimization,
        volatility-adjusted sizing, and correlation impact analysis.
        """
        try:
            # Get current portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics()
            
            # Calculate Kelly Criterion fraction
            kelly_fraction = self._calculate_kelly_fraction(symbol, signal_strength, signal_confidence)
            
            # Calculate volatility-adjusted sizing
            vol_adjusted_size = self._calculate_volatility_adjusted_size(
                symbol, market_volatility, kelly_fraction
            )
            
            # Apply correlation constraints
            correlation_adjusted_size = self._apply_correlation_constraints(
                symbol, vol_adjusted_size, portfolio_metrics
            )
            
            # Apply exposure limits
            exposure_limited_size = self._apply_exposure_limits(
                symbol, correlation_adjusted_size, current_inventory
            )
            
            # Calculate final recommended size
            recommended_size = self._calculate_final_position_size(
                exposure_limited_size, signal_strength, signal_confidence
            )
            
            # Calculate bounds
            max_size = self._calculate_max_position_size(symbol, portfolio_metrics)
            min_size = self._calculate_min_position_size(symbol)
            
            # Risk-adjusted final size
            risk_adjusted_size = max(min_size, min(recommended_size, max_size))
            
            # Calculate confidence in sizing
            sizing_confidence = self._calculate_sizing_confidence(
                kelly_fraction, signal_confidence, portfolio_metrics
            )
            
            # Generate reasoning
            reasoning = self._generate_sizing_reasoning(
                kelly_fraction, vol_adjusted_size, correlation_adjusted_size,
                exposure_limited_size, risk_adjusted_size
            )
            
            return PositionSizing(
                recommended_size=risk_adjusted_size,
                max_size=max_size,
                min_size=min_size,
                kelly_fraction=kelly_fraction,
                risk_adjusted_size=risk_adjusted_size,
                confidence=sizing_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Position sizing calculation failed: {e}")
            return self._get_default_position_sizing(symbol)

    def _calculate_kelly_fraction(self, symbol: str, signal_strength: float, 
                                signal_confidence: float) -> float:
        """Calculate Kelly Criterion fraction based on historical performance"""
        try:
            # Get historical trades for this symbol
            symbol_trades = [t for t in self.trade_history if t.get('symbol') == symbol]
            
            if len(symbol_trades) < 10:
                # Not enough history, use conservative estimate
                base_kelly = abs(signal_strength) * signal_confidence * 0.1
                return max(self.min_kelly_fraction, min(base_kelly, self.max_kelly_fraction))
            
            # Calculate win rate and average win/loss
            wins = [t for t in symbol_trades if t.get('pnl', 0) > 0]
            losses = [t for t in symbol_trades if t.get('pnl', 0) < 0]
            
            if not wins or not losses:
                # No losses or no wins, use conservative estimate
                return self.min_kelly_fraction
            
            win_rate = len(wins) / len(symbol_trades)
            avg_win = np.mean([t['pnl'] for t in wins])
            avg_loss = abs(np.mean([t['pnl'] for t in losses]))
            
            if avg_loss == 0:
                return self.min_kelly_fraction
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply signal strength and confidence adjustments
            adjusted_kelly = kelly_fraction * abs(signal_strength) * signal_confidence
            
            # Apply conservative multiplier and bounds
            final_kelly = adjusted_kelly * self.kelly_multiplier
            
            return max(self.min_kelly_fraction, min(final_kelly, self.max_kelly_fraction))
            
        except Exception as e:
            self.logger.error(f"Kelly fraction calculation failed: {e}")
            return self.min_kelly_fraction

    def _calculate_volatility_adjusted_size(self, symbol: str, market_volatility: float, 
                                          kelly_fraction: float) -> float:
        """
        Calculate volatility-adjusted position size using ATR-based dynamic sizing.
        Feature 2: Enhanced Position Sizing with volatility adaptation.
        """
        try:
            # Base size from Kelly fraction
            base_size = kelly_fraction * self.total_capital
            
            # Enhanced volatility adjustment using multiple factors
            target_volatility = self.config.get('target_volatility', 0.02)  # 2% daily target volatility
            
            # ATR-based volatility adjustment
            # High volatility (>4% daily) = reduce position size significantly
            # Medium volatility (1-4% daily) = moderate adjustment
            # Low volatility (<1% daily) = increase position size slightly
            
            # Claude's Conservative Volatility Adjustments (neutral sizing to maintain baseline)
            if market_volatility > 0.06:  # Very high volatility (>6%)
                vol_adjustment = 1.0  # Conservative: no reduction to maintain performance
                self.logger.info(f"High volatility detected ({market_volatility:.3f}): maintaining normal position size (conservative)")
            elif market_volatility > 0.02:  # Medium volatility (2-6%)
                vol_adjustment = 1.0  # Conservative: normal size
            else:  # Low volatility (<2%)
                vol_adjustment = 1.0  # Conservative: no increase to avoid over-leverage
                self.logger.info(f"Low volatility detected ({market_volatility:.3f}): maintaining normal position size (conservative)")
            
            # Apply additional scaling based on portfolio volatility
            portfolio_vol = self._estimate_portfolio_volatility()
            if portfolio_vol > 0.03:  # Portfolio volatility >3%
                vol_adjustment *= 0.8  # Further reduce by 20%
            
            # Apply volatility adjustment
            vol_adjusted_size = base_size * vol_adjustment
            
            # Volatility-based position size bounds
            min_size = self.total_capital * 0.01  # 1% minimum
            max_size = self.total_capital * 0.15  # 15% maximum (reduced from 20% for safety)
            
            # Apply intelligent bounds based on market conditions
            if market_volatility > 0.05:  # Extreme volatility
                max_size = self.total_capital * 0.08  # Cap at 8%
            
            final_size = max(min_size, min(vol_adjusted_size, max_size))
            
            # Log sizing decision for transparency
            self.logger.debug(f"Volatility-adjusted sizing for {symbol}: "
                            f"Base: ${base_size:.0f}, Vol: {market_volatility:.3f}, "
                            f"Adjustment: {vol_adjustment:.2f}, Final: ${final_size:.0f}")
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error in volatility-adjusted sizing: {e}")
            return kelly_fraction * self.total_capital * 0.5  # Conservative fallback

    def _apply_correlation_constraints(self, symbol: str, base_size: float, 
                                     portfolio_metrics: InventoryMetrics) -> float:
        """Apply correlation constraints to position sizing"""
        try:
            # Update correlation matrix if needed
            if self._should_update_correlations():
                self._update_correlation_matrix()
            
            # Calculate correlation risk for this symbol
            correlation_risk = self._calculate_symbol_correlation_risk(symbol)
            
            # Adjust size based on correlation risk
            if correlation_risk > self.correlation_threshold:
                # High correlation with existing positions, reduce size
                correlation_adjustment = 1.0 - (correlation_risk - self.correlation_threshold) * 2
                adjusted_size = base_size * max(0.3, correlation_adjustment)
            else:
                # Low correlation, allow normal sizing
                adjusted_size = base_size
            
            # Additional adjustment for overall portfolio correlation risk
            if portfolio_metrics.correlation_risk > 0.8:
                adjusted_size *= 0.7  # Reduce all new positions when portfolio correlation is high
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Correlation constraint application failed: {e}")
            return base_size * 0.8  # Conservative adjustment

    def _apply_exposure_limits(self, symbol: str, base_size: float, 
                             current_inventory: float) -> float:
        """Apply exposure limits to position sizing"""
        try:
            # Calculate current exposures
            current_exposure = sum(abs(pos.position_value) for pos in self.positions.values())
            max_total_exposure = self.total_capital * self.max_inventory_ratio
            
            # Check total exposure limit
            available_exposure = max_total_exposure - current_exposure
            if available_exposure <= 0:
                return 0.0  # No capacity for new positions
            
            # Limit position size to available exposure
            exposure_limited_size = min(base_size, available_exposure)
            
            # Check single position limit
            max_single_exposure = self.total_capital * self.max_single_position
            single_position_limited_size = min(exposure_limited_size, max_single_exposure)
            
            # Check existing position in same symbol
            existing_position = self.positions.get(symbol)
            if existing_position:
                existing_exposure = abs(existing_position.position_value)
                remaining_single_limit = max_single_exposure - existing_exposure
                final_size = min(single_position_limited_size, remaining_single_limit)
            else:
                final_size = single_position_limited_size
            
            return max(0.0, final_size)
            
        except Exception as e:
            self.logger.error(f"Exposure limit application failed: {e}")
            return min(base_size * 0.5, self.total_capital * 0.05)  # Conservative fallback

    def _calculate_final_position_size(self, base_size: float, signal_strength: float, 
                                     signal_confidence: float) -> float:
        """Calculate final position size with signal adjustments"""
        try:
            # Signal strength adjustment
            strength_adjustment = abs(signal_strength)  # 0 to 1
            
            # Confidence adjustment
            confidence_adjustment = signal_confidence  # 0 to 1
            
            # Combined adjustment
            signal_adjustment = strength_adjustment * confidence_adjustment
            
            # Apply adjustment with minimum threshold
            min_signal_threshold = 0.3
            if signal_adjustment < min_signal_threshold:
                signal_adjustment = min_signal_threshold
            
            adjusted_size = base_size * signal_adjustment
            
            return adjusted_size
            
        except Exception:
            return base_size * 0.5

    def _calculate_max_position_size(self, symbol: str, portfolio_metrics: InventoryMetrics) -> float:
        """Calculate maximum allowed position size"""
        try:
            # Base maximum from single position limit
            base_max = self.total_capital * self.max_single_position
            
            # Adjust for current portfolio concentration
            if portfolio_metrics.concentration_risk > 0.8:
                concentration_adjustment = 0.7
            elif portfolio_metrics.concentration_risk > 0.6:
                concentration_adjustment = 0.85
            else:
                concentration_adjustment = 1.0
            
            # Adjust for correlation risk
            correlation_risk = self._calculate_symbol_correlation_risk(symbol)
            if correlation_risk > self.correlation_threshold:
                correlation_adjustment = 0.6
            else:
                correlation_adjustment = 1.0
            
            max_size = base_max * concentration_adjustment * correlation_adjustment
            
            return max_size
            
        except Exception:
            return self.total_capital * 0.05  # 5% maximum fallback

    def _calculate_min_position_size(self, symbol: str) -> float:
        """Calculate minimum position size"""
        try:
            # Minimum based on trading costs and market constraints
            min_notional = 1000  # $1000 minimum
            
            # Adjust for symbol-specific minimums
            symbol_minimums = {
                'BTCUSDT': 500,
                'ETHUSDT': 500,
                'BNBUSDT': 100,
                'ADAUSDT': 50,
                'SOLUSDT': 100
            }
            
            symbol_min = symbol_minimums.get(symbol, 100)
            
            return max(min_notional, symbol_min)
            
        except Exception:
            return 1000  # Default minimum

    def _calculate_sizing_confidence(self, kelly_fraction: float, signal_confidence: float,
                                   portfolio_metrics: InventoryMetrics) -> float:
        """Calculate confidence in the position sizing recommendation"""
        try:
            confidence_factors = []
            
            # Kelly fraction confidence (higher Kelly = more confidence, but capped)
            kelly_confidence = min(1.0, kelly_fraction / 0.1)  # Normalize by 10% Kelly
            confidence_factors.append(kelly_confidence)
            
            # Signal confidence
            confidence_factors.append(signal_confidence)
            
            # Portfolio health confidence
            portfolio_health = 1.0 - (portfolio_metrics.concentration_risk + portfolio_metrics.correlation_risk) / 2
            confidence_factors.append(portfolio_health)
            
            # Historical performance confidence
            if len(self.trade_history) > 50:
                recent_trades = self.trade_history[-50:]
                win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
                performance_confidence = (win_rate - 0.5) * 2 + 0.5  # Center around 0.5
                confidence_factors.append(max(0.1, min(1.0, performance_confidence)))
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5

    def _generate_sizing_reasoning(self, kelly_fraction: float, vol_adjusted_size: float,
                                 correlation_adjusted_size: float, exposure_limited_size: float,
                                 final_size: float) -> str:
        """Generate human-readable reasoning for the sizing decision"""
        try:
            reasoning_parts = []
            
            reasoning_parts.append(f"Kelly fraction: {kelly_fraction:.3f}")
            
            vol_impact = (vol_adjusted_size / (kelly_fraction * self.total_capital)) if kelly_fraction > 0 else 1
            if vol_impact < 0.9:
                reasoning_parts.append(f"Reduced {(1-vol_impact)*100:.1f}% for volatility")
            
            corr_impact = correlation_adjusted_size / vol_adjusted_size if vol_adjusted_size > 0 else 1
            if corr_impact < 0.9:
                reasoning_parts.append(f"Reduced {(1-corr_impact)*100:.1f}% for correlation")
            
            exp_impact = exposure_limited_size / correlation_adjusted_size if correlation_adjusted_size > 0 else 1
            if exp_impact < 0.9:
                reasoning_parts.append(f"Limited by exposure constraints")
            
            return "; ".join(reasoning_parts)
            
        except Exception:
            return "Standard risk-adjusted sizing applied"

    def manage_portfolio_exposure(self, positions: Dict[str, Any], 
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage overall portfolio exposure and correlation with dynamic position rebalancing
        and emergency exposure reduction.
        """
        try:
            # Update current positions
            self._update_positions(positions)
            
            # Calculate current portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics()
            
            # Check for rebalancing needs
            rebalancing_actions = self._determine_rebalancing_actions(portfolio_metrics, market_conditions)
            
            # Check for emergency exposure reduction
            emergency_actions = self._check_emergency_conditions(portfolio_metrics, market_conditions)
            
            # Generate portfolio recommendations
            recommendations = self._generate_portfolio_recommendations(
                portfolio_metrics, rebalancing_actions, emergency_actions
            )
            
            return {
                'status': 'completed',
                'portfolio_metrics': {
                    'total_exposure': portfolio_metrics.total_exposure,
                    'long_exposure': portfolio_metrics.long_exposure,
                    'short_exposure': portfolio_metrics.short_exposure,
                    'net_exposure': portfolio_metrics.net_exposure,
                    'concentration_risk': portfolio_metrics.concentration_risk,
                    'correlation_risk': portfolio_metrics.correlation_risk,
                    'inventory_turnover': portfolio_metrics.inventory_turnover,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'max_drawdown': portfolio_metrics.max_drawdown
                },
                'rebalancing_actions': rebalancing_actions,
                'emergency_actions': emergency_actions,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio exposure management failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def calculate_portfolio_metrics(self) -> InventoryMetrics:
        """Calculate comprehensive portfolio-level risk and exposure metrics"""
        try:
            if not self.positions:
                return self._get_default_metrics()
            
            # Calculate exposures
            position_values = [abs(pos.position_value) for pos in self.positions.values()]
            long_values = [pos.position_value for pos in self.positions.values() if pos.position_value > 0]
            short_values = [abs(pos.position_value) for pos in self.positions.values() if pos.position_value < 0]
            
            total_exposure = sum(position_values)
            long_exposure = sum(long_values) if long_values else 0
            short_exposure = sum(short_values) if short_values else 0
            net_exposure = long_exposure - short_exposure
            
            # Calculate concentration risk (Herfindahl index)
            if total_exposure > 0:
                weights = [pv / total_exposure for pv in position_values]
                concentration_risk = sum(w ** 2 for w in weights)
            else:
                concentration_risk = 0.0
            
            # Calculate correlation risk
            correlation_risk = self._calculate_portfolio_correlation_risk()
            
            # Calculate inventory turnover
            inventory_turnover = self._calculate_inventory_turnover()
            
            # Calculate performance metrics
            sharpe_ratio = self._calculate_portfolio_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            
            return InventoryMetrics(
                total_exposure=total_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                inventory_turnover=inventory_turnover,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {e}")
            return self._get_default_metrics()

    def _update_positions(self, positions: Dict[str, Any]):
        """Update internal position tracking"""
        try:
            self.positions.clear()
            
            for symbol, pos_data in positions.items():
                position = PositionInfo(
                    symbol=symbol,
                    side=pos_data.get('side', 'long'),
                    quantity=pos_data.get('quantity', 0),
                    entry_price=pos_data.get('entry_price', 0),
                    current_price=pos_data.get('current_price', 0),
                    unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                    position_value=pos_data.get('quantity', 0) * pos_data.get('current_price', 0),
                    weight=0.0,  # Will be calculated
                    correlation_risk=0.0  # Will be calculated
                )
                
                self.positions[symbol] = position
            
            # Calculate weights and correlation risks
            total_value = sum(abs(pos.position_value) for pos in self.positions.values())
            
            for position in self.positions.values():
                if total_value > 0:
                    position.weight = abs(position.position_value) / total_value
                position.correlation_risk = self._calculate_symbol_correlation_risk(position.symbol)
                
        except Exception as e:
            self.logger.error(f"Position update failed: {e}")

    def _determine_rebalancing_actions(self, portfolio_metrics: InventoryMetrics, 
                                     market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine what rebalancing actions are needed"""
        try:
            actions = []
            
            # Check concentration risk
            if portfolio_metrics.concentration_risk > 0.4:  # 40% in single position
                # Find most concentrated positions
                sorted_positions = sorted(
                    self.positions.values(), 
                    key=lambda p: p.weight, 
                    reverse=True
                )
                
                for pos in sorted_positions[:3]:  # Top 3 positions
                    if pos.weight > 0.15:  # >15% concentration
                        target_reduction = (pos.weight - 0.1) * portfolio_metrics.total_exposure
                        actions.append({
                            'action': 'reduce_position',
                            'symbol': pos.symbol,
                            'current_weight': pos.weight,
                            'target_reduction': target_reduction,
                            'reason': 'concentration_risk'
                        })
            
            # Check correlation risk
            if portfolio_metrics.correlation_risk > self.correlation_threshold:
                # Find highly correlated positions
                for symbol, position in self.positions.items():
                    if position.correlation_risk > self.correlation_threshold:
                        actions.append({
                            'action': 'reduce_correlated_position',
                            'symbol': symbol,
                            'correlation_risk': position.correlation_risk,
                            'target_reduction': position.position_value * 0.3,
                            'reason': 'correlation_risk'
                        })
            
            # Check exposure limits
            exposure_ratio = portfolio_metrics.total_exposure / self.total_capital
            if exposure_ratio > self.max_inventory_ratio:
                excess_exposure = portfolio_metrics.total_exposure - (self.total_capital * self.max_inventory_ratio)
                actions.append({
                    'action': 'reduce_total_exposure',
                    'excess_exposure': excess_exposure,
                    'current_ratio': exposure_ratio,
                    'target_ratio': self.max_inventory_ratio,
                    'reason': 'exposure_limit'
                })
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Rebalancing action determination failed: {e}")
            return []

    def _check_emergency_conditions(self, portfolio_metrics: InventoryMetrics, 
                                  market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for emergency conditions requiring immediate action"""
        try:
            emergency_actions = []
            
            # Emergency drawdown condition
            if portfolio_metrics.max_drawdown > 0.15:  # 15% drawdown
                emergency_actions.append({
                    'action': 'emergency_exposure_reduction',
                    'severity': 'high',
                    'target_reduction': 0.5,  # Reduce all positions by 50%
                    'reason': f'excessive_drawdown_{portfolio_metrics.max_drawdown:.3f}'
                })
            
            # Emergency correlation breakdown
            if portfolio_metrics.correlation_risk > 0.9:  # 90% correlation
                emergency_actions.append({
                    'action': 'emergency_diversification',
                    'severity': 'medium',
                    'target_reduction': 0.3,  # Reduce correlated positions by 30%
                    'reason': 'correlation_breakdown'
                })
            
            # Emergency concentration
            max_weight = max([pos.weight for pos in self.positions.values()], default=0)
            if max_weight > 0.5:  # 50% in single position
                emergency_actions.append({
                    'action': 'emergency_deconcentration',
                    'severity': 'high',
                    'max_weight': max_weight,
                    'target_max_weight': 0.2,
                    'reason': 'excessive_concentration'
                })
            
            # Market stress condition
            market_stress = market_conditions.get('volatility', 0.3)
            if market_stress > 1.0:  # 100% volatility
                emergency_actions.append({
                    'action': 'stress_exposure_reduction',
                    'severity': 'medium',
                    'market_stress': market_stress,
                    'target_reduction': min(0.4, market_stress - 0.5),
                    'reason': 'market_stress'
                })
            
            return emergency_actions
            
        except Exception as e:
            self.logger.error(f"Emergency condition check failed: {e}")
            return []

    def _generate_portfolio_recommendations(self, portfolio_metrics: InventoryMetrics,
                                          rebalancing_actions: List[Dict], 
                                          emergency_actions: List[Dict]) -> List[str]:
        """Generate human-readable portfolio recommendations"""
        try:
            recommendations = []
            
            # Overall portfolio health
            if portfolio_metrics.sharpe_ratio > 1.5:
                recommendations.append("Portfolio performance is excellent - maintain current strategy")
            elif portfolio_metrics.sharpe_ratio > 1.0:
                recommendations.append("Portfolio performance is good - minor optimizations possible")
            else:
                recommendations.append("Portfolio performance needs improvement - review strategy")
            
            # Specific recommendations based on actions
            if emergency_actions:
                recommendations.append("URGENT: Emergency risk management actions required")
                for action in emergency_actions:
                    recommendations.append(f"- {action['action']}: {action['reason']}")
            
            if rebalancing_actions:
                recommendations.append("Portfolio rebalancing recommended:")
                for action in rebalancing_actions:
                    recommendations.append(f"- {action['action']} for {action['symbol']}: {action['reason']}")
            
            # Risk-based recommendations
            if portfolio_metrics.concentration_risk > 0.3:
                recommendations.append("Consider diversifying - concentration risk elevated")
            
            if portfolio_metrics.correlation_risk > 0.7:
                recommendations.append("Reduce correlated positions - correlation risk high")
            
            exposure_ratio = portfolio_metrics.total_exposure / self.total_capital
            if exposure_ratio > 0.8:
                recommendations.append("High exposure detected - consider reducing leverage")
            elif exposure_ratio < 0.2:
                recommendations.append("Low exposure - consider increasing position sizes if conditions permit")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to calculation error"]

    # Helper methods (simplified implementations)
    def _should_update_correlations(self) -> bool:
        """Check if correlation matrix needs updating"""
        if self.last_correlation_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_correlation_update
        return time_since_update.total_seconds() > self.correlation_update_frequency

    def _update_correlation_matrix(self):
        """Update correlation matrix (simplified)"""
        try:
            # In production, this would calculate correlations from price data
            # For now, use simplified correlation estimates
            symbols = list(self.positions.keys())
            self.correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                self.correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        self.correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Simplified correlation based on symbol similarity
                        if 'BTC' in symbol1 and 'BTC' in symbol2:
                            correlation = 0.9
                        elif 'ETH' in symbol1 and 'ETH' in symbol2:
                            correlation = 0.85
                        elif symbol1.endswith('USDT') and symbol2.endswith('USDT'):
                            correlation = 0.6  # Crypto correlation
                        else:
                            correlation = 0.3  # Low correlation
                        
                        self.correlation_matrix[symbol1][symbol2] = correlation
            
            self.last_correlation_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Correlation matrix update failed: {e}")

    def _calculate_symbol_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for a specific symbol"""
        try:
            if not self.correlation_matrix or symbol not in self.correlation_matrix:
                return 0.5  # Default medium risk
            
            correlations = []
            for other_symbol, position in self.positions.items():
                if other_symbol != symbol and other_symbol in self.correlation_matrix[symbol]:
                    correlation = self.correlation_matrix[symbol][other_symbol]
                    # Weight by position size
                    weighted_correlation = correlation * position.weight
                    correlations.append(weighted_correlation)
            
            return sum(correlations) if correlations else 0.0
            
        except Exception:
            return 0.5
    
    def _estimate_portfolio_volatility(self) -> float:
        """
        Estimate current portfolio volatility from recent performance.
        Used for volatility-adjusted position sizing.
        """
        try:
            if len(self.performance_history) < 10:
                return 0.02  # Default 2% daily volatility
            
            # Get recent returns (last 30 periods)
            recent_performance = self.performance_history[-30:]
            returns = []
            
            for i in range(1, len(recent_performance)):
                prev_value = recent_performance[i-1].get('portfolio_value', 1)
                curr_value = recent_performance[i].get('portfolio_value', 1)
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) < 5:
                return 0.02  # Default if insufficient data
            
            # Calculate realized volatility
            portfolio_volatility = np.std(returns)
            
            # Apply bounds to prevent extreme values
            return max(0.005, min(portfolio_volatility, 0.1))  # 0.5% to 10% daily vol
            
        except Exception as e:
            self.logger.error(f"Error estimating portfolio volatility: {e}")
            return 0.02

    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate overall portfolio correlation risk"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            weighted_correlations = []
            symbols = list(self.positions.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if (symbol1 in self.correlation_matrix and 
                        symbol2 in self.correlation_matrix[symbol1]):
                        
                        correlation = self.correlation_matrix[symbol1][symbol2]
                        weight1 = self.positions[symbol1].weight
                        weight2 = self.positions[symbol2].weight
                        
                        weighted_correlation = correlation * weight1 * weight2
                        weighted_correlations.append(weighted_correlation)
            
            return sum(weighted_correlations) if weighted_correlations else 0.0
            
        except Exception:
            return 0.5

    def _calculate_inventory_turnover(self) -> float:
        """Calculate inventory turnover rate"""
        try:
            # Simplified calculation based on trade history
            if len(self.trade_history) < 10:
                return 0.5  # Default
            
            recent_trades = self.trade_history[-50:]  # Last 50 trades
            total_volume = sum(abs(t.get('volume', 0)) for t in recent_trades)
            avg_portfolio_value = sum(abs(pos.position_value) for pos in self.positions.values())
            
            if avg_portfolio_value > 0:
                turnover = total_volume / avg_portfolio_value
                return min(10.0, turnover)  # Cap at 10x
            
            return 0.5
            
        except Exception:
            return 0.5

    def _calculate_portfolio_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio"""
        try:
            if len(self.performance_history) < 30:
                return 1.0  # Default
            
            returns = [p.get('daily_return', 0) for p in self.performance_history[-252:]]  # Last year
            
            if not returns:
                return 1.0
            
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            
            if return_std > 0:
                sharpe = (avg_return * 252) / (return_std * np.sqrt(252))  # Annualized
                return sharpe
            
            return 1.0
            
        except Exception:
            return 1.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(self.performance_history) < 10:
                return 0.0
            
            portfolio_values = [p.get('portfolio_value', 0) for p in self.performance_history[-100:]]
            
            if not portfolio_values:
                return 0.0
            
            peak = portfolio_values[0]
            max_drawdown = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception:
            return 0.0

    def _get_default_position_sizing(self, symbol: str) -> PositionSizing:
        """Get default position sizing when calculation fails"""
        conservative_size = self.total_capital * 0.02  # 2% of capital
        
        return PositionSizing(
            recommended_size=conservative_size,
            max_size=conservative_size * 2,
            min_size=1000,
            kelly_fraction=0.02,
            risk_adjusted_size=conservative_size,
            confidence=0.3,
            reasoning="Default conservative sizing due to calculation error"
        )

    def _get_default_metrics(self) -> InventoryMetrics:
        """Get default metrics when calculation fails"""
        return InventoryMetrics(
            total_exposure=0.0,
            long_exposure=0.0,
            short_exposure=0.0,
            net_exposure=0.0,
            concentration_risk=0.0,
            correlation_risk=0.0,
            inventory_turnover=0.0,
            sharpe_ratio=1.0,
            max_drawdown=0.0
        )