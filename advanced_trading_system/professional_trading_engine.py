"""
Professional Trading Engine - Main Integration Module
Coordinates all advanced trading system components for institutional-grade crypto trading
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Import all system components
from .volatility_adaptive_grid import VolatilityAdaptiveGrid
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer
from .advanced_delta_hedger import AdvancedDeltaHedger
from .funding_rate_arbitrage import FundingRateArbitrage
from .order_flow_analyzer import OrderFlowAnalyzer
from .intelligent_inventory_manager import IntelligentInventoryManager

@dataclass
class TradingDecision:
    """Trading decision output"""
    action: str  # 'buy', 'sell', 'hold', 'hedge', 'arbitrage'
    symbol: str
    quantity: float
    price: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_profit: float
    time_horizon: int
    strategy_type: str

class ProfessionalTradingEngine:
    """
    Main trading engine that coordinates all advanced components for professional
    institutional-grade crypto trading with BitVol/LXVX volatility indicators.
    """
    
    def __init__(self, binance_client=None, config: Dict = None):
        """Initialize the professional trading engine"""
        self.binance_client = binance_client
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Professional Trading Engine v1.0")
        
        # Initialize all components
        self._initialize_components()
        
        # Trading state
        self.active_positions = {}
        self.trading_enabled = True
        self.emergency_mode = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0
        }

    def _initialize_components(self):
        """Initialize all trading system components"""
        try:
            # Core volatility and grid management
            self.volatility_grid = VolatilityAdaptiveGrid(
                binance_client=self.binance_client,
                config=self.config.get('volatility_grid', {})
            )
            
            # Multi-timeframe analysis with volatility surface
            self.timeframe_analyzer = MultiTimeframeAnalyzer(
                binance_client=self.binance_client,
                config=self.config.get('timeframe_analyzer', {})
            )
            
            # Advanced delta hedging with gamma management
            self.delta_hedger = AdvancedDeltaHedger(
                binance_client=self.binance_client,
                config=self.config.get('delta_hedger', {})
            )
            
            # Cross-exchange funding rate arbitrage
            self.funding_arbitrage = FundingRateArbitrage(
                config=self.config.get('funding_arbitrage', {})
            )
            
            # Order flow and market microstructure analysis
            self.order_flow_analyzer = OrderFlowAnalyzer(
                binance_client=self.binance_client,
                config=self.config.get('order_flow', {})
            )
            
            # Intelligent inventory management with Kelly Criterion
            self.inventory_manager = IntelligentInventoryManager(
                config=self.config.get('inventory_manager', {})
            )
            
            self.logger.info("All trading components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive market analysis combining all components.
        Returns institutional-grade market intelligence.
        """
        try:
            self.logger.info(f"Analyzing market conditions for {symbol}")
            
            # Run all analyses in parallel for efficiency
            tasks = [
                self.volatility_grid.get_volatility_surface_analysis(symbol),
                self.timeframe_analyzer.get_comprehensive_analysis(symbol),
                self.order_flow_analyzer.get_comprehensive_analysis(symbol),
                self.funding_arbitrage.scan_funding_opportunities()
            ]
            
            # Execute all analyses concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            volatility_analysis = results[0] if not isinstance(results[0], Exception) else {}
            timeframe_analysis = results[1] if not isinstance(results[1], Exception) else {}
            order_flow_analysis = results[2] if not isinstance(results[2], Exception) else {}
            funding_opportunities = results[3] if not isinstance(results[3], Exception) else []
            
            # Combine all analyses
            market_conditions = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'volatility_analysis': volatility_analysis,
                'timeframe_analysis': timeframe_analysis,
                'order_flow_analysis': order_flow_analysis,
                'funding_opportunities': [
                    {
                        'symbol': opp.symbol,
                        'rate_difference': opp.rate_difference,
                        'annualized_return': opp.annualized_return,
                        'confidence': opp.confidence
                    }
                    for opp in funding_opportunities if opp.symbol == symbol
                ],
                'market_regime': self._determine_market_regime(
                    volatility_analysis, timeframe_analysis, order_flow_analysis
                ),
                'risk_assessment': self._assess_market_risk(
                    volatility_analysis, timeframe_analysis, order_flow_analysis
                )
            }
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Market analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def generate_trading_decision(self, symbol: str, 
                                      market_conditions: Optional[Dict] = None) -> TradingDecision:
        """
        Generate professional trading decision based on comprehensive analysis.
        Integrates all components for institutional-grade decision making.
        """
        try:
            # Get market conditions if not provided
            if not market_conditions:
                market_conditions = await self.analyze_market_conditions(symbol)
            
            # Extract key metrics
            volatility_metrics = market_conditions.get('volatility_analysis', {}).get('volatility_metrics', {})
            composite_signal = market_conditions.get('timeframe_analysis', {}).get('composite_signal', {})
            order_flow_metrics = market_conditions.get('order_flow_analysis', {}).get('order_flow_metrics', {})
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return self._get_default_decision(symbol, "Unable to get current price")
            
            # Calculate position sizing
            signal_strength = composite_signal.get('signal', 0.0)
            signal_confidence = composite_signal.get('confidence', 0.5)
            market_volatility = volatility_metrics.get('composite_vol', 0.3)
            
            position_sizing = self.inventory_manager.calculate_optimal_position_size(
                symbol=symbol,
                signal_strength=signal_strength,
                market_volatility=market_volatility,
                current_inventory=self._get_current_inventory(symbol),
                signal_confidence=signal_confidence
            )
            
            # Determine action based on integrated analysis
            action = self._determine_action(
                signal_strength, signal_confidence, order_flow_metrics, volatility_metrics
            )
            
            # Calculate optimal execution strategy
            execution_strategy = self.order_flow_analyzer.optimize_order_placement(
                symbol=symbol,
                side='buy' if signal_strength > 0 else 'sell',
                quantity=position_sizing.recommended_size / current_price,
                market_conditions={
                    'flow_metrics': order_flow_metrics,
                    'urgency': 'medium'
                }
            )
            
            # Calculate risk score
            risk_score = self._calculate_integrated_risk_score(
                volatility_metrics, order_flow_metrics, position_sizing
            )
            
            # Estimate expected profit
            expected_profit = self._estimate_expected_profit(
                signal_strength, signal_confidence, position_sizing.recommended_size,
                volatility_metrics.get('composite_vol', 0.3)
            )
            
            # Generate reasoning
            reasoning = self._generate_decision_reasoning(
                action, signal_strength, signal_confidence, risk_score,
                volatility_metrics, order_flow_metrics
            )
            
            return TradingDecision(
                action=action,
                symbol=symbol,
                quantity=position_sizing.recommended_size / current_price,
                price=execution_strategy.recommended_price or current_price,
                confidence=signal_confidence * position_sizing.confidence,
                reasoning=reasoning,
                risk_score=risk_score,
                expected_profit=expected_profit,
                time_horizon=execution_strategy.time_horizon,
                strategy_type=execution_strategy.strategy_type
            )
            
        except Exception as e:
            self.logger.error(f"Trading decision generation failed: {e}")
            return self._get_default_decision(symbol, f"Decision generation error: {str(e)}")

    async def execute_trading_strategy(self, symbol: str) -> Dict[str, Any]:
        """
        Execute complete trading strategy including grid management, hedging, and arbitrage.
        This is the main execution loop for the professional trading system.
        """
        try:
            if not self.trading_enabled:
                return {'status': 'disabled', 'message': 'Trading is disabled'}
            
            self.logger.info(f"Executing trading strategy for {symbol}")
            
            # 1. Analyze market conditions
            market_conditions = await self.analyze_market_conditions(symbol)
            
            # 2. Check for emergency conditions
            emergency_status = await self._check_emergency_conditions(market_conditions)
            if emergency_status['emergency']:
                return await self._handle_emergency(symbol, emergency_status)
            
            # 3. Generate trading decision
            trading_decision = await self.generate_trading_decision(symbol, market_conditions)
            
            # 4. Execute primary strategy
            primary_execution = await self._execute_primary_strategy(trading_decision, market_conditions)
            
            # 5. Manage delta hedging
            hedge_management = await self._manage_delta_hedging(symbol, market_conditions)
            
            # 6. Check funding rate arbitrage opportunities
            funding_arbitrage = await self._check_funding_arbitrage(symbol, market_conditions)
            
            # 7. Update inventory management
            inventory_update = await self._update_inventory_management(symbol, market_conditions)
            
            # 8. Update performance metrics
            self._update_performance_metrics(primary_execution, hedge_management, funding_arbitrage)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'trading_decision': {
                    'action': trading_decision.action,
                    'quantity': trading_decision.quantity,
                    'price': trading_decision.price,
                    'confidence': trading_decision.confidence,
                    'reasoning': trading_decision.reasoning
                },
                'execution_results': {
                    'primary_strategy': primary_execution,
                    'hedge_management': hedge_management,
                    'funding_arbitrage': funding_arbitrage,
                    'inventory_update': inventory_update
                },
                'market_conditions': market_conditions.get('market_regime', 'unknown'),
                'risk_score': trading_decision.risk_score,
                'expected_profit': trading_decision.expected_profit
            }
            
        except Exception as e:
            self.logger.error(f"Trading strategy execution failed: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _execute_primary_strategy(self, decision: TradingDecision, 
                                      market_conditions: Dict) -> Dict[str, Any]:
        """Execute the primary trading strategy"""
        try:
            if decision.action == 'hold':
                return {'action': 'hold', 'reason': 'No trading signal'}
            
            # Calculate grid parameters for volatility-adaptive grid
            grid_params = await self.volatility_grid.calculate_dynamic_spacing(
                decision.symbol, decision.price
            )
            
            # Simulate order execution (in production, this would place real orders)
            execution_result = {
                'action': decision.action,
                'symbol': decision.symbol,
                'quantity': decision.quantity,
                'price': decision.price,
                'grid_spacing': grid_params.spacing,
                'grid_levels': grid_params.max_levels,
                'volatility_regime': grid_params.regime,
                'execution_strategy': decision.strategy_type,
                'status': 'simulated',  # Would be 'executed' in production
                'timestamp': datetime.now().isoformat()
            }
            
            # Update position tracking
            self._update_position_tracking(decision)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Primary strategy execution failed: {e}")
            return {'action': 'failed', 'error': str(e)}

    async def _manage_delta_hedging(self, symbol: str, market_conditions: Dict) -> Dict[str, Any]:
        """Manage delta hedging for existing positions"""
        try:
            current_position = self.active_positions.get(symbol)
            if not current_position:
                return {'status': 'no_position', 'action': 'none'}
            
            # Calculate portfolio delta
            portfolio_delta = self._calculate_portfolio_delta()
            
            # Manage dynamic hedging
            hedge_result = await self.delta_hedger.manage_dynamic_hedging(
                symbol=symbol,
                portfolio_delta=portfolio_delta,
                market_conditions=market_conditions
            )
            
            return hedge_result
            
        except Exception as e:
            self.logger.error(f"Delta hedging management failed: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _check_funding_arbitrage(self, symbol: str, market_conditions: Dict) -> Dict[str, Any]:
        """Check and execute funding rate arbitrage opportunities"""
        try:
            # Get funding opportunities for this symbol
            funding_opportunities = market_conditions.get('funding_opportunities', [])
            
            if not funding_opportunities:
                return {'status': 'no_opportunities', 'action': 'none'}
            
            # Find best opportunity
            best_opportunity = max(funding_opportunities, key=lambda x: x['annualized_return'])
            
            # Check if opportunity meets threshold
            if (best_opportunity['annualized_return'] > 0.05 and  # 5% minimum return
                best_opportunity['confidence'] > 0.6):  # 60% confidence
                
                # Would execute arbitrage in production
                return {
                    'status': 'opportunity_found',
                    'action': 'execute_arbitrage',
                    'opportunity': best_opportunity,
                    'execution': 'simulated'  # Would be real execution in production
                }
            
            return {'status': 'opportunity_below_threshold', 'action': 'none'}
            
        except Exception as e:
            self.logger.error(f"Funding arbitrage check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _update_inventory_management(self, symbol: str, market_conditions: Dict) -> Dict[str, Any]:
        """Update inventory management and portfolio exposure"""
        try:
            # Get current positions for portfolio management
            current_positions = self._get_all_positions()
            
            # Manage portfolio exposure
            portfolio_management = self.inventory_manager.manage_portfolio_exposure(
                positions=current_positions,
                market_conditions=market_conditions
            )
            
            return portfolio_management
            
        except Exception as e:
            self.logger.error(f"Inventory management update failed: {e}")
            return {'status': 'error', 'error': str(e)}

    async def _check_emergency_conditions(self, market_conditions: Dict) -> Dict[str, Any]:
        """Check for emergency market conditions"""
        try:
            emergency_indicators = []
            
            # Check volatility spike
            volatility_metrics = market_conditions.get('volatility_analysis', {}).get('volatility_metrics', {})
            composite_vol = volatility_metrics.get('composite_vol', 0.3)
            
            if composite_vol > 1.0:  # 100% volatility
                emergency_indicators.append('volatility_spike')
            
            # Check correlation breakdown
            if volatility_metrics.get('confidence', 1.0) < 0.3:
                emergency_indicators.append('low_confidence')
            
            # Check order flow stress
            order_flow_metrics = market_conditions.get('order_flow_analysis', {}).get('order_flow_metrics', {})
            if order_flow_metrics.get('execution_difficulty', 0.5) > 0.8:
                emergency_indicators.append('execution_stress')
            
            # Check funding rate extremes
            funding_opportunities = market_conditions.get('funding_opportunities', [])
            extreme_funding = any(abs(opp.get('rate_difference', 0)) > 0.02 for opp in funding_opportunities)
            if extreme_funding:
                emergency_indicators.append('extreme_funding')
            
            return {
                'emergency': len(emergency_indicators) > 0,
                'indicators': emergency_indicators,
                'severity': len(emergency_indicators) / 4.0  # Normalize to 0-1
            }
            
        except Exception as e:
            self.logger.error(f"Emergency condition check failed: {e}")
            return {'emergency': False, 'error': str(e)}

    async def _handle_emergency(self, symbol: str, emergency_status: Dict) -> Dict[str, Any]:
        """Handle emergency market conditions"""
        try:
            self.logger.warning(f"Emergency conditions detected for {symbol}: {emergency_status['indicators']}")
            
            emergency_actions = []
            
            # Reduce position sizes
            if 'volatility_spike' in emergency_status['indicators']:
                emergency_actions.append('reduce_positions')
            
            # Increase hedging
            if 'execution_stress' in emergency_status['indicators']:
                emergency_actions.append('increase_hedging')
            
            # Pause new positions
            if emergency_status['severity'] > 0.75:
                emergency_actions.append('pause_trading')
                self.trading_enabled = False
            
            # Execute emergency protocols
            for action in emergency_actions:
                if action == 'reduce_positions':
                    await self._emergency_position_reduction(symbol, emergency_status['severity'])
                elif action == 'increase_hedging':
                    await self._emergency_hedge_adjustment(symbol, emergency_status['severity'])
            
            return {
                'status': 'emergency_handled',
                'actions_taken': emergency_actions,
                'severity': emergency_status['severity'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Emergency handling failed: {e}")
            return {'status': 'emergency_handling_failed', 'error': str(e)}

    async def _emergency_position_reduction(self, symbol: str, severity: float):
        """Reduce positions during emergency"""
        try:
            position = self.active_positions.get(symbol)
            if position:
                reduction_factor = min(0.8, severity)  # Reduce up to 80%
                new_size = position['quantity'] * (1 - reduction_factor)
                position['quantity'] = new_size
                self.logger.warning(f"Emergency position reduction for {symbol}: {reduction_factor:.1%}")
                
        except Exception as e:
            self.logger.error(f"Emergency position reduction failed: {e}")

    async def _emergency_hedge_adjustment(self, symbol: str, severity: float):
        """Adjust hedging during emergency"""
        try:
            emergency_result = await self.delta_hedger.emergency_hedge_adjustment(
                symbol=symbol,
                emergency_type='market_stress',
                severity=severity
            )
            self.logger.warning(f"Emergency hedge adjustment for {symbol}: {emergency_result}")
            
        except Exception as e:
            self.logger.error(f"Emergency hedge adjustment failed: {e}")

    # Helper methods for market analysis and decision making
    def _determine_market_regime(self, volatility_analysis: Dict, 
                               timeframe_analysis: Dict, order_flow_analysis: Dict) -> str:
        """Determine overall market regime"""
        try:
            # Extract regime indicators
            vol_regime = volatility_analysis.get('grid_parameters', {}).get('regime', 'normal')
            signal_regime = timeframe_analysis.get('composite_signal', {}).get('regime', 'ranging')
            flow_bias = order_flow_analysis.get('order_flow_metrics', {}).get('directional_bias', 'neutral')
            
            # Combine regimes with priority weighting
            if 'extreme' in vol_regime or 'volatile' in signal_regime:
                return 'high_volatility'
            elif 'trending' in signal_regime and 'bullish' in flow_bias:
                return 'bullish_trending'
            elif 'trending' in signal_regime and 'bearish' in flow_bias:
                return 'bearish_trending'
            elif 'ranging' in signal_regime:
                return 'range_bound'
            else:
                return 'normal'
                
        except Exception:
            return 'unknown'

    def _assess_market_risk(self, volatility_analysis: Dict, 
                          timeframe_analysis: Dict, order_flow_analysis: Dict) -> float:
        """Assess overall market risk score (0-1)"""
        try:
            risk_factors = []
            
            # Volatility risk
            composite_vol = volatility_analysis.get('volatility_metrics', {}).get('composite_vol', 0.3)
            vol_risk = min(1.0, composite_vol / 0.8)  # Normalize by 80% vol
            risk_factors.append(vol_risk)
            
            # Signal confidence risk
            signal_confidence = timeframe_analysis.get('composite_signal', {}).get('confidence', 0.5)
            confidence_risk = 1.0 - signal_confidence
            risk_factors.append(confidence_risk)
            
            # Execution difficulty risk
            execution_difficulty = order_flow_analysis.get('order_flow_metrics', {}).get('execution_difficulty', 0.5)
            risk_factors.append(execution_difficulty)
            
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception:
            return 0.5

    def _determine_action(self, signal_strength: float, signal_confidence: float,
                        order_flow_metrics: Dict, volatility_metrics: Dict) -> str:
        """Determine trading action based on integrated analysis"""
        try:
            # Minimum thresholds
            min_signal_strength = 0.3
            min_confidence = 0.5
            
            # Check if signal meets minimum requirements
            if abs(signal_strength) < min_signal_strength or signal_confidence < min_confidence:
                return 'hold'
            
            # Check execution conditions
            execution_difficulty = order_flow_metrics.get('execution_difficulty', 0.5)
            if execution_difficulty > 0.8:  # Very difficult execution
                return 'hold'
            
            # Check volatility conditions
            composite_vol = volatility_metrics.get('composite_vol', 0.3)
            if composite_vol > 1.0:  # Extreme volatility
                return 'hold'
            
            # Determine direction
            if signal_strength > min_signal_strength:
                return 'buy'
            elif signal_strength < -min_signal_strength:
                return 'sell'
            else:
                return 'hold'
                
        except Exception:
            return 'hold'

    def _calculate_integrated_risk_score(self, volatility_metrics: Dict,
                                       order_flow_metrics: Dict, position_sizing: Any) -> float:
        """Calculate integrated risk score"""
        try:
            risk_components = []
            
            # Volatility risk
            vol_risk = min(1.0, volatility_metrics.get('composite_vol', 0.3) / 0.6)
            risk_components.append(vol_risk)
            
            # Execution risk
            exec_risk = order_flow_metrics.get('execution_difficulty', 0.5)
            risk_components.append(exec_risk)
            
            # Position sizing risk
            kelly_fraction = getattr(position_sizing, 'kelly_fraction', 0.1)
            sizing_risk = min(1.0, kelly_fraction / 0.2)  # Normalize by 20% Kelly
            risk_components.append(sizing_risk)
            
            # Liquidity risk
            liquidity_score = order_flow_metrics.get('liquidity_score', 0.5)
            liquidity_risk = 1.0 - liquidity_score
            risk_components.append(liquidity_risk)
            
            return np.mean(risk_components)
            
        except Exception:
            return 0.5

    def _estimate_expected_profit(self, signal_strength: float, signal_confidence: float,
                                position_size: float, volatility: float) -> float:
        """Estimate expected profit for the trade"""
        try:
            # Base expected return from signal
            base_return = abs(signal_strength) * signal_confidence * 0.02  # 2% max expected return
            
            # Adjust for position size
            profit_estimate = base_return * position_size
            
            # Adjust for volatility (higher vol = higher potential profit but also risk)
            volatility_adjustment = min(1.5, 1.0 + (volatility - 0.3) * 2)
            
            adjusted_profit = profit_estimate * volatility_adjustment
            
            return adjusted_profit
            
        except Exception:
            return 0.0

    def _generate_decision_reasoning(self, action: str, signal_strength: float,
                                   signal_confidence: float, risk_score: float,
                                   volatility_metrics: Dict, order_flow_metrics: Dict) -> str:
        """Generate human-readable reasoning for the decision"""
        try:
            reasoning_parts = []
            
            # Action rationale
            if action == 'buy':
                reasoning_parts.append(f"Bullish signal (strength: {signal_strength:.2f})")
            elif action == 'sell':
                reasoning_parts.append(f"Bearish signal (strength: {signal_strength:.2f})")
            else:
                reasoning_parts.append("No clear directional signal")
            
            # Confidence factor
            reasoning_parts.append(f"confidence: {signal_confidence:.2f}")
            
            # Risk assessment
            if risk_score > 0.7:
                reasoning_parts.append("high-risk environment")
            elif risk_score < 0.3:
                reasoning_parts.append("low-risk environment")
            
            # Market conditions
            vol_regime = volatility_metrics.get('regime', 'normal')
            if vol_regime != 'normal':
                reasoning_parts.append(f"volatility regime: {vol_regime}")
            
            execution_difficulty = order_flow_metrics.get('execution_difficulty', 0.5)
            if execution_difficulty > 0.7:
                reasoning_parts.append("challenging execution conditions")
            
            return "; ".join(reasoning_parts)
            
        except Exception:
            return f"Action: {action} based on systematic analysis"

    # Utility methods
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            if self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            return None
        except Exception:
            return None

    def _get_current_inventory(self, symbol: str) -> float:
        """Get current inventory for symbol"""
        position = self.active_positions.get(symbol, {})
        return position.get('quantity', 0.0)

    def _calculate_portfolio_delta(self) -> float:
        """Calculate portfolio delta"""
        total_delta = 0.0
        for position in self.active_positions.values():
            # Simplified delta calculation
            total_delta += position.get('quantity', 0.0)
        return total_delta

    def _update_position_tracking(self, decision: TradingDecision):
        """Update position tracking"""
        try:
            if decision.action in ['buy', 'sell']:
                current_pos = self.active_positions.get(decision.symbol, {'quantity': 0.0})
                
                if decision.action == 'buy':
                    current_pos['quantity'] = current_pos.get('quantity', 0.0) + decision.quantity
                else:
                    current_pos['quantity'] = current_pos.get('quantity', 0.0) - decision.quantity
                
                current_pos['last_price'] = decision.price
                current_pos['last_update'] = datetime.now().isoformat()
                
                self.active_positions[decision.symbol] = current_pos
                
        except Exception as e:
            self.logger.error(f"Position tracking update failed: {e}")

    def _get_all_positions(self) -> Dict[str, Any]:
        """Get all current positions for portfolio management"""
        return self.active_positions.copy()

    def _update_performance_metrics(self, primary_execution: Dict, 
                                  hedge_management: Dict, funding_arbitrage: Dict):
        """Update performance tracking metrics"""
        try:
            if primary_execution.get('status') == 'simulated':
                self.performance_metrics['total_trades'] += 1
                
                # In production, would track actual P&L
                # For now, simulate based on decision quality
                
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")

    def _get_default_decision(self, symbol: str, reason: str) -> TradingDecision:
        """Get default decision when generation fails"""
        return TradingDecision(
            action='hold',
            symbol=symbol,
            quantity=0.0,
            price=0.0,
            confidence=0.0,
            reasoning=reason,
            risk_score=1.0,  # High risk for failed analysis
            expected_profit=0.0,
            time_horizon=0,
            strategy_type='none'
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'system': {
                    'trading_enabled': self.trading_enabled,
                    'emergency_mode': self.emergency_mode,
                    'active_positions': len(self.active_positions),
                    'timestamp': datetime.now().isoformat()
                },
                'performance': self.performance_metrics,
                'components': {
                    'volatility_grid': 'active',
                    'timeframe_analyzer': 'active',
                    'delta_hedger': 'active',
                    'funding_arbitrage': 'active',
                    'order_flow_analyzer': 'active',
                    'inventory_manager': 'active'
                }
            }
            
        except Exception as e:
            self.logger.error(f"System status check failed: {e}")
            return {
                'system': {'status': 'error', 'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }