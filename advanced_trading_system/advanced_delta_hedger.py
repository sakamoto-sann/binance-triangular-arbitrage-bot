"""
Advanced Delta Hedging System with Gamma Management
Implements sophisticated delta-neutral hedging with option-like gamma exposure management
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
import math

@dataclass
class PositionData:
    """Data class for position information"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

@dataclass
class HedgeParameters:
    """Data class for hedge parameters"""
    hedge_ratio: float
    target_delta: float
    rebalance_threshold: float
    hedge_symbol: str
    hedge_quantity: float
    confidence: float
    gamma_adjustment: float
    vega_adjustment: float

@dataclass
class PortfolioGreeks:
    """Data class for portfolio-level Greeks"""
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    var_95: float
    expected_shortfall: float
    max_loss_1day: float
    hedge_effectiveness: float

class AdvancedDeltaHedger:
    """
    Professional delta hedging system with gamma management and dynamic hedge ratios.
    Not just 1:1 hedging, but intelligent hedge ratio adjustment based on market conditions.
    """
    
    def __init__(self, binance_client=None, config: Dict = None):
        """Initialize the advanced delta hedger"""
        self.binance_client = binance_client
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Hedge configuration
        self.base_hedge_ratio = self.config.get('base_hedge_ratio', 1.0)
        self.gamma_adjustment_factor = self.config.get('gamma_adjustment_factor', 0.1)
        self.hedge_frequency = self.config.get('hedge_frequency', 60)  # seconds
        self.max_delta_deviation = self.config.get('max_delta_deviation', 0.05)  # 5%
        
        # Greek calculation parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% risk-free rate
        self.default_volatility = self.config.get('default_volatility', 0.3)  # 30% default vol
        self.default_time_to_expiry = self.config.get('default_time_to_expiry', 30)  # 30 days
        
        # Rebalancing thresholds
        self.rebalance_thresholds = {
            'delta': self.config.get('delta_threshold', 0.05),      # 5% delta deviation
            'gamma': self.config.get('gamma_threshold', 0.1),       # 10% gamma deviation
            'time': self.config.get('time_threshold', 300),         # 5 minutes
            'price_move': self.config.get('price_move_threshold', 0.02)  # 2% price move
        }
        
        # Portfolio tracking
        self.positions = {}  # symbol -> PositionData
        self.hedge_positions = {}  # symbol -> PositionData
        self.last_rebalance = {}  # symbol -> datetime
        self.hedge_performance = {}  # symbol -> performance metrics
        
        # Supported hedge instruments
        self.hedge_instruments = {
            'BTCUSDT': ['BTCUSDT', 'BTCUSD_PERP'],
            'ETHUSDT': ['ETHUSDT', 'ETHUSD_PERP'],
            'BNBUSDT': ['BNBUSDT'],
            'ADAUSDT': ['ADAUSDT'],
            'SOLUSDT': ['SOLUSDT']
        }

    async def calculate_optimal_hedge_ratio(self, symbol: str, spot_position: float, 
                                          current_price: float, volatility: float) -> HedgeParameters:
        """
        Calculate dynamic hedge ratio considering gamma effects and market conditions.
        Returns optimal hedge parameters for the given position.
        """
        try:
            # Get current position data
            position_data = await self._get_position_data(symbol, spot_position, current_price)
            
            # Calculate portfolio Greeks
            portfolio_greeks = self._calculate_portfolio_greeks([position_data])
            
            # Base hedge ratio calculation
            base_ratio = self._calculate_base_hedge_ratio(position_data, volatility)
            
            # Gamma adjustment
            gamma_adjustment = self._calculate_gamma_adjustment(position_data, volatility)
            
            # Vega adjustment for volatility exposure
            vega_adjustment = self._calculate_vega_adjustment(position_data, volatility)
            
            # Market condition adjustment
            market_adjustment = await self._calculate_market_condition_adjustment(symbol, volatility)
            
            # Final hedge ratio
            final_hedge_ratio = (
                base_ratio * 
                (1 + gamma_adjustment) * 
                (1 + vega_adjustment) * 
                market_adjustment
            )
            
            # Determine hedge instrument and quantity
            hedge_symbol = self._select_hedge_instrument(symbol)
            hedge_quantity = abs(spot_position * final_hedge_ratio)
            
            # Calculate target delta
            target_delta = self._calculate_target_delta(position_data, portfolio_greeks)
            
            # Determine rebalance threshold
            rebalance_threshold = self._calculate_dynamic_rebalance_threshold(
                position_data, volatility, market_adjustment
            )
            
            # Calculate confidence score
            confidence = self._calculate_hedge_confidence(
                position_data, volatility, market_adjustment
            )
            
            return HedgeParameters(
                hedge_ratio=final_hedge_ratio,
                target_delta=target_delta,
                rebalance_threshold=rebalance_threshold,
                hedge_symbol=hedge_symbol,
                hedge_quantity=hedge_quantity,
                confidence=confidence,
                gamma_adjustment=gamma_adjustment,
                vega_adjustment=vega_adjustment
            )
            
        except Exception as e:
            self.logger.error(f"Hedge ratio calculation failed: {e}")
            # Return default parameters
            return self._get_default_hedge_parameters(symbol, spot_position)

    def _calculate_base_hedge_ratio(self, position_data: PositionData, volatility: float) -> float:
        """Calculate base hedge ratio using Black-Scholes-like approach"""
        try:
            # For spot positions, we start with 1:1 hedge ratio
            base_ratio = 1.0
            
            # Adjust for position size (larger positions need more careful hedging)
            position_size_factor = min(2.0, math.log(abs(position_data.quantity) / 1000 + 1))
            
            # Adjust for volatility (higher vol needs more hedging)
            volatility_factor = min(1.5, 1.0 + (volatility - 0.3) * 2)
            
            # Adjust for time decay consideration
            time_factor = 1.0  # For spot positions, no time decay
            
            return base_ratio * position_size_factor * volatility_factor * time_factor
            
        except Exception:
            return self.base_hedge_ratio

    def _calculate_gamma_adjustment(self, position_data: PositionData, volatility: float) -> float:
        """Calculate gamma adjustment factor for hedge ratio"""
        try:
            # Simulate gamma exposure for spot positions
            # Gamma represents the rate of change of delta with respect to price
            
            # For large positions, gamma exposure increases
            position_size = abs(position_data.quantity * position_data.current_price)
            
            # Normalized gamma (as percentage of position size)
            if position_size > 0:
                # Approximate gamma based on position size and volatility
                implied_gamma = (volatility ** 2) * position_size / (position_data.current_price ** 2)
                
                # Gamma adjustment factor
                gamma_factor = implied_gamma * self.gamma_adjustment_factor
                
                # Cap the adjustment
                return min(0.5, max(-0.5, gamma_factor))
            else:
                return 0.0
                
        except Exception:
            return 0.0

    def _calculate_vega_adjustment(self, position_data: PositionData, volatility: float) -> float:
        """Calculate vega adjustment for volatility exposure"""
        try:
            # Vega represents sensitivity to volatility changes
            # For spot positions, we approximate vega exposure
            
            position_value = abs(position_data.quantity * position_data.current_price)
            
            # Approximate vega based on position value and current volatility
            # Higher volatility means higher vega exposure
            implied_vega = position_value * volatility * 0.1  # Scaling factor
            
            # Vega adjustment factor (reduce hedge ratio in low vol, increase in high vol)
            if volatility > 0.5:  # High volatility
                vega_adjustment = 0.1  # Increase hedging
            elif volatility < 0.2:  # Low volatility
                vega_adjustment = -0.05  # Slightly reduce hedging
            else:
                vega_adjustment = 0.0  # No adjustment
            
            return vega_adjustment
            
        except Exception:
            return 0.0

    async def _calculate_market_condition_adjustment(self, symbol: str, volatility: float) -> float:
        """Calculate market condition adjustment factor"""
        try:
            adjustments = []
            
            # Volatility regime adjustment
            if volatility > 0.8:  # Extreme volatility
                adjustments.append(1.3)  # Increase hedging
            elif volatility > 0.5:  # High volatility
                adjustments.append(1.1)  # Slightly increase hedging
            elif volatility < 0.15:  # Very low volatility
                adjustments.append(0.9)  # Reduce hedging
            else:
                adjustments.append(1.0)  # No adjustment
            
            # Liquidity adjustment
            liquidity_factor = await self._assess_market_liquidity(symbol)
            if liquidity_factor < 0.7:  # Poor liquidity
                adjustments.append(0.8)  # Reduce hedging (harder to execute)
            else:
                adjustments.append(1.0)
            
            # Correlation adjustment
            correlation_factor = await self._assess_hedge_correlation(symbol)
            if correlation_factor < 0.8:  # Poor correlation
                adjustments.append(1.2)  # Increase hedging to compensate
            else:
                adjustments.append(1.0)
            
            # Funding rate adjustment
            funding_rate = await self._get_funding_rate(symbol)
            if funding_rate is not None:
                if abs(funding_rate) > 0.01:  # High funding rate
                    adjustments.append(0.9)  # Reduce hedging (funding cost)
                else:
                    adjustments.append(1.0)
            else:
                adjustments.append(1.0)
            
            # Return combined adjustment
            return np.mean(adjustments) if adjustments else 1.0
            
        except Exception:
            return 1.0

    async def _assess_market_liquidity(self, symbol: str) -> float:
        """Assess market liquidity for the symbol"""
        try:
            if self.binance_client:
                # Get order book depth
                order_book = self.binance_client.get_order_book(symbol=symbol, limit=20)
                
                # Calculate bid-ask spread
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                spread = (best_ask - best_bid) / best_bid
                
                # Calculate depth
                bid_depth = sum(float(level[1]) for level in order_book['bids'][:10])
                ask_depth = sum(float(level[1]) for level in order_book['asks'][:10])
                total_depth = bid_depth + ask_depth
                
                # Liquidity score based on spread and depth
                spread_score = max(0.1, 1.0 - (spread * 1000))  # Lower spread = better
                depth_score = min(1.0, total_depth / 1000)  # Normalize depth
                
                return (spread_score + depth_score) / 2
            
            return 0.8  # Default assumption
            
        except Exception:
            return 0.8

    async def _assess_hedge_correlation(self, symbol: str) -> float:
        """Assess correlation between spot and hedge instrument"""
        try:
            hedge_symbol = self._select_hedge_instrument(symbol)
            
            if hedge_symbol == symbol:
                return 1.0  # Perfect correlation (same instrument)
            
            # Get price data for both instruments
            spot_data = await self._get_price_data(symbol, '1h', 48)
            hedge_data = await self._get_price_data(hedge_symbol, '1h', 48)
            
            if spot_data is not None and hedge_data is not None:
                # Calculate returns
                spot_returns = np.log(spot_data['close'].values[1:] / spot_data['close'].values[:-1])
                hedge_returns = np.log(hedge_data['close'].values[1:] / hedge_data['close'].values[:-1])
                
                # Align data
                min_len = min(len(spot_returns), len(hedge_returns))
                spot_returns = spot_returns[-min_len:]
                hedge_returns = hedge_returns[-min_len:]
                
                if min_len > 10:
                    # Calculate correlation
                    correlation = np.corrcoef(spot_returns, hedge_returns)[0, 1]
                    return abs(correlation) if not np.isnan(correlation) else 0.8
            
            return 0.8  # Default assumption
            
        except Exception:
            return 0.8

    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for the symbol"""
        try:
            if self.binance_client:
                # Check if it's a futures symbol
                if symbol.endswith('USDT'):
                    funding_info = self.binance_client.futures_funding_rate(symbol=symbol, limit=1)
                    if funding_info:
                        return float(funding_info[0]['fundingRate'])
            
            return None
            
        except Exception:
            return None

    def _select_hedge_instrument(self, symbol: str) -> str:
        """Select the best hedge instrument for the given symbol"""
        try:
            instruments = self.hedge_instruments.get(symbol, [symbol])
            
            # For now, use the first available instrument
            # In a full implementation, this would consider:
            # - Liquidity
            # - Correlation
            # - Funding costs
            # - Execution costs
            
            return instruments[0]
            
        except Exception:
            return symbol

    def _calculate_target_delta(self, position_data: PositionData, 
                              portfolio_greeks: PortfolioGreeks) -> float:
        """Calculate target delta for the position"""
        try:
            # Target is typically delta-neutral (0)
            # But can be adjusted based on market conditions
            
            base_target = 0.0
            
            # Adjust for portfolio-level considerations
            if abs(portfolio_greeks.net_delta) > 0.1:  # Significant portfolio delta
                # Adjust individual position target to help neutralize portfolio
                adjustment = -portfolio_greeks.net_delta * 0.1  # Partial adjustment
                base_target += adjustment
            
            # Clamp target delta
            return max(-0.1, min(0.1, base_target))
            
        except Exception:
            return 0.0

    def _calculate_dynamic_rebalance_threshold(self, position_data: PositionData, 
                                             volatility: float, market_adjustment: float) -> float:
        """Calculate dynamic rebalance threshold based on market conditions"""
        try:
            # Base threshold
            base_threshold = self.rebalance_thresholds['delta']
            
            # Adjust for volatility
            volatility_adjustment = volatility / 0.3  # Normalize to 30% baseline
            
            # Adjust for market conditions
            market_factor = 2.0 - market_adjustment  # Inverse relationship
            
            # Adjust for position size
            position_size = abs(position_data.quantity * position_data.current_price)
            size_factor = min(2.0, math.log(position_size / 10000 + 1))
            
            # Calculate final threshold
            final_threshold = (
                base_threshold * 
                volatility_adjustment * 
                market_factor * 
                size_factor
            )
            
            # Apply bounds
            return max(0.01, min(0.2, final_threshold))
            
        except Exception:
            return self.rebalance_thresholds['delta']

    def _calculate_hedge_confidence(self, position_data: PositionData, 
                                  volatility: float, market_adjustment: float) -> float:
        """Calculate confidence score for the hedge"""
        try:
            confidence_factors = []
            
            # Volatility factor (moderate volatility is better for hedging)
            if 0.2 <= volatility <= 0.6:
                vol_confidence = 1.0
            elif volatility < 0.2:
                vol_confidence = 0.7  # Too low, harder to hedge
            else:
                vol_confidence = max(0.3, 1.0 - (volatility - 0.6) * 2)
            
            confidence_factors.append(vol_confidence)
            
            # Market condition factor
            market_confidence = min(1.0, market_adjustment)
            confidence_factors.append(market_confidence)
            
            # Position size factor (moderate sizes are easier to hedge)
            position_value = abs(position_data.quantity * position_data.current_price)
            if position_value < 1000:
                size_confidence = 0.6  # Too small
            elif position_value > 100000:
                size_confidence = 0.8  # Large but manageable
            else:
                size_confidence = 1.0  # Optimal size
            
            confidence_factors.append(size_confidence)
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.7

    async def manage_dynamic_hedging(self, symbol: str, portfolio_delta: float, 
                                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Continuously manage hedge based on market conditions and portfolio delta.
        This is the main hedging management function.
        """
        try:
            # Get current position
            position = self.positions.get(symbol)
            if not position:
                return {'status': 'no_position', 'action': 'none'}
            
            # Calculate current portfolio Greeks
            portfolio_greeks = self._calculate_portfolio_greeks([position])
            
            # Determine if rebalancing is needed
            rebalance_needed = await self._should_rebalance(symbol, portfolio_delta, market_conditions)
            
            if not rebalance_needed:
                return {
                    'status': 'no_rebalance_needed',
                    'action': 'none',
                    'current_delta': portfolio_greeks.net_delta,
                    'target_delta': 0.0
                }
            
            # Calculate optimal hedge parameters
            volatility = market_conditions.get('volatility', self.default_volatility)
            hedge_params = await self.calculate_optimal_hedge_ratio(
                symbol, position.quantity, position.current_price, volatility
            )
            
            # Execute hedge adjustment
            hedge_result = await self._execute_hedge_adjustment(symbol, hedge_params, portfolio_greeks)
            
            # Update tracking
            self.last_rebalance[symbol] = datetime.now()
            
            # Calculate cost-benefit analysis
            cost_benefit = self._calculate_hedge_cost_benefit(hedge_result, portfolio_greeks)
            
            return {
                'status': 'hedge_executed',
                'action': hedge_result['action'],
                'hedge_params': {
                    'hedge_ratio': hedge_params.hedge_ratio,
                    'hedge_quantity': hedge_params.hedge_quantity,
                    'confidence': hedge_params.confidence
                },
                'portfolio_greeks': {
                    'net_delta_before': portfolio_greeks.net_delta,
                    'net_delta_after': hedge_result.get('net_delta_after', 0),
                    'net_gamma': portfolio_greeks.net_gamma,
                    'var_95': portfolio_greeks.var_95
                },
                'cost_benefit': cost_benefit,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic hedging management failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _should_rebalance(self, symbol: str, portfolio_delta: float, 
                              market_conditions: Dict[str, Any]) -> bool:
        """Determine if rebalancing is needed"""
        try:
            # Delta threshold check
            if abs(portfolio_delta) > self.rebalance_thresholds['delta']:
                self.logger.info(f"Rebalance needed for {symbol}: Delta threshold exceeded")
                return True
            
            # Time threshold check
            last_rebalance = self.last_rebalance.get(symbol)
            if last_rebalance:
                time_since_rebalance = (datetime.now() - last_rebalance).total_seconds()
                if time_since_rebalance > self.rebalance_thresholds['time']:
                    self.logger.info(f"Rebalance needed for {symbol}: Time threshold exceeded")
                    return True
            else:
                # No previous rebalance, should rebalance
                return True
            
            # Price move threshold check
            position = self.positions.get(symbol)
            if position:
                price_change = abs(position.current_price - position.entry_price) / position.entry_price
                if price_change > self.rebalance_thresholds['price_move']:
                    self.logger.info(f"Rebalance needed for {symbol}: Price move threshold exceeded")
                    return True
            
            # Volatility spike check
            current_vol = market_conditions.get('volatility', 0.3)
            if current_vol > 0.8:  # Extreme volatility
                self.logger.info(f"Rebalance needed for {symbol}: Volatility spike detected")
                return True
            
            return False
            
        except Exception:
            return False

    async def _execute_hedge_adjustment(self, symbol: str, hedge_params: HedgeParameters, 
                                      portfolio_greeks: PortfolioGreeks) -> Dict[str, Any]:
        """Execute the hedge adjustment"""
        try:
            # Get current hedge position
            current_hedge = self.hedge_positions.get(hedge_params.hedge_symbol)
            current_hedge_quantity = current_hedge.quantity if current_hedge else 0.0
            
            # Calculate required hedge adjustment
            required_hedge_quantity = hedge_params.hedge_quantity
            
            # Determine if we're long or short the underlying
            position = self.positions.get(symbol)
            if position and position.side == 'long':
                # Long underlying needs short hedge
                required_hedge_quantity = -required_hedge_quantity
            
            # Calculate the adjustment needed
            adjustment_quantity = required_hedge_quantity - current_hedge_quantity
            
            # Execute the hedge adjustment (simulation)
            execution_result = await self._simulate_hedge_execution(
                hedge_params.hedge_symbol, adjustment_quantity, hedge_params.confidence
            )
            
            # Update hedge position
            if execution_result['status'] == 'success':
                self._update_hedge_position(
                    hedge_params.hedge_symbol, 
                    required_hedge_quantity, 
                    execution_result['execution_price']
                )
                
                # Calculate new portfolio delta
                new_portfolio_greeks = self._calculate_portfolio_greeks([position])
                
                return {
                    'action': 'hedge_adjusted',
                    'adjustment_quantity': adjustment_quantity,
                    'execution_price': execution_result['execution_price'],
                    'execution_cost': execution_result['execution_cost'],
                    'net_delta_after': new_portfolio_greeks.net_delta,
                    'status': 'success'
                }
            else:
                return {
                    'action': 'hedge_failed',
                    'error': execution_result.get('error', 'Unknown error'),
                    'status': 'failed'
                }
                
        except Exception as e:
            self.logger.error(f"Hedge execution failed: {e}")
            return {
                'action': 'hedge_failed',
                'error': str(e),
                'status': 'failed'
            }

    async def _simulate_hedge_execution(self, hedge_symbol: str, quantity: float, 
                                      confidence: float) -> Dict[str, Any]:
        """Simulate hedge execution (in production, this would place actual orders)"""
        try:
            if abs(quantity) < 0.001:  # Minimum quantity threshold
                return {
                    'status': 'success',
                    'execution_price': 0.0,
                    'execution_cost': 0.0,
                    'message': 'No execution needed'
                }
            
            # Get current market price
            current_price = await self._get_current_price(hedge_symbol)
            if current_price is None:
                return {
                    'status': 'failed',
                    'error': 'Unable to get current price'
                }
            
            # Simulate execution with slippage
            slippage_factor = self._calculate_slippage(hedge_symbol, quantity, confidence)
            execution_price = current_price * (1 + slippage_factor)
            
            # Calculate execution cost
            execution_cost = abs(quantity * execution_price * 0.001)  # 0.1% execution cost
            
            return {
                'status': 'success',
                'execution_price': execution_price,
                'execution_cost': execution_cost,
                'slippage': slippage_factor
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _calculate_slippage(self, symbol: str, quantity: float, confidence: float) -> float:
        """Calculate expected slippage for the hedge execution"""
        try:
            # Base slippage (0.1% for normal conditions)
            base_slippage = 0.001
            
            # Adjust for quantity (larger quantities have more slippage)
            quantity_factor = min(3.0, abs(quantity) / 1000)  # Scale by position size
            
            # Adjust for confidence (lower confidence = higher slippage)
            confidence_factor = 2.0 - confidence
            
            # Calculate total slippage
            total_slippage = base_slippage * quantity_factor * confidence_factor
            
            # Apply sign based on trade direction
            slippage_sign = 1 if quantity > 0 else -1
            
            return total_slippage * slippage_sign
            
        except Exception:
            return 0.001  # Default slippage

    def _update_hedge_position(self, hedge_symbol: str, quantity: float, price: float):
        """Update hedge position tracking"""
        try:
            self.hedge_positions[hedge_symbol] = PositionData(
                symbol=hedge_symbol,
                side='long' if quantity > 0 else 'short',
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                delta=quantity,  # For spot positions, delta = quantity
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update hedge position: {e}")

    def _calculate_hedge_cost_benefit(self, hedge_result: Dict[str, Any], 
                                    portfolio_greeks: PortfolioGreeks) -> Dict[str, float]:
        """Calculate cost-benefit analysis of the hedge"""
        try:
            # Execution cost
            execution_cost = hedge_result.get('execution_cost', 0.0)
            
            # Risk reduction benefit
            delta_reduction = abs(portfolio_greeks.net_delta) - abs(hedge_result.get('net_delta_after', 0))
            risk_reduction_value = delta_reduction * 1000  # Approximate value of risk reduction
            
            # Net benefit
            net_benefit = risk_reduction_value - execution_cost
            
            # Cost-benefit ratio
            cost_benefit_ratio = risk_reduction_value / execution_cost if execution_cost > 0 else float('inf')
            
            return {
                'execution_cost': execution_cost,
                'risk_reduction_value': risk_reduction_value,
                'net_benefit': net_benefit,
                'cost_benefit_ratio': cost_benefit_ratio,
                'hedge_effectiveness': min(1.0, delta_reduction / abs(portfolio_greeks.net_delta)) if portfolio_greeks.net_delta != 0 else 0.0
            }
            
        except Exception:
            return {
                'execution_cost': 0.0,
                'risk_reduction_value': 0.0,
                'net_benefit': 0.0,
                'cost_benefit_ratio': 1.0,
                'hedge_effectiveness': 0.0
            }

    async def _get_position_data(self, symbol: str, quantity: float, price: float) -> PositionData:
        """Get or create position data"""
        try:
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
                position.timestamp = datetime.now()
                return position
            else:
                # Create new position
                position = PositionData(
                    symbol=symbol,
                    side='long' if quantity > 0 else 'short',
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    delta=quantity,  # For spot positions
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    timestamp=datetime.now()
                )
                self.positions[symbol] = position
                return position
                
        except Exception as e:
            self.logger.error(f"Failed to get position data: {e}")
            # Return default position
            return PositionData(
                symbol=symbol, side='long', quantity=quantity, entry_price=price,
                current_price=price, unrealized_pnl=0.0, delta=quantity,
                gamma=0.0, theta=0.0, vega=0.0, timestamp=datetime.now()
            )

    def _calculate_portfolio_greeks(self, positions: List[PositionData]) -> PortfolioGreeks:
        """Calculate portfolio-level Greeks"""
        try:
            # Sum up all Greeks
            net_delta = sum(pos.delta for pos in positions)
            net_gamma = sum(pos.gamma for pos in positions)
            net_theta = sum(pos.theta for pos in positions)
            net_vega = sum(pos.vega for pos in positions)
            
            # Calculate portfolio VaR (simplified)
            portfolio_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
            daily_volatility = 0.02  # Assume 2% daily volatility
            var_95 = portfolio_value * daily_volatility * 1.645  # 95% VaR
            
            # Expected shortfall (simplified)
            expected_shortfall = var_95 * 1.3  # Approximate ES
            
            # Maximum 1-day loss estimate
            max_loss_1day = portfolio_value * daily_volatility * 3  # 3-sigma move
            
            # Hedge effectiveness (simplified)
            hedge_effectiveness = max(0.0, 1.0 - abs(net_delta) / portfolio_value) if portfolio_value > 0 else 0.0
            
            return PortfolioGreeks(
                net_delta=net_delta,
                net_gamma=net_gamma,
                net_theta=net_theta,
                net_vega=net_vega,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_loss_1day=max_loss_1day,
                hedge_effectiveness=hedge_effectiveness
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio Greeks calculation failed: {e}")
            return PortfolioGreeks(
                net_delta=0.0, net_gamma=0.0, net_theta=0.0, net_vega=0.0,
                var_95=0.0, expected_shortfall=0.0, max_loss_1day=0.0,
                hedge_effectiveness=0.0
            )

    def _get_default_hedge_parameters(self, symbol: str, quantity: float) -> HedgeParameters:
        """Get default hedge parameters when calculation fails"""
        return HedgeParameters(
            hedge_ratio=self.base_hedge_ratio,
            target_delta=0.0,
            rebalance_threshold=self.rebalance_thresholds['delta'],
            hedge_symbol=symbol,
            hedge_quantity=abs(quantity),
            confidence=0.5,
            gamma_adjustment=0.0,
            vega_adjustment=0.0
        )

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            return None
        except Exception:
            return None

    async def _get_price_data(self, symbol: str, interval: str = '1h', limit: int = 48) -> Optional[pd.DataFrame]:
        """Get price data for analysis"""
        try:
            if self.binance_client:
                klines = self.binance_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades_count',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert to proper data types
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    return df
            
            return None
            
        except Exception:
            return None

    async def get_hedge_status(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive hedge status for a symbol"""
        try:
            position = self.positions.get(symbol)
            hedge_position = self.hedge_positions.get(symbol)
            
            if not position:
                return {
                    'status': 'no_position',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate current Greeks
            portfolio_greeks = self._calculate_portfolio_greeks([position])
            
            # Get hedge performance metrics
            hedge_performance = self.hedge_performance.get(symbol, {})
            
            return {
                'status': 'active',
                'symbol': symbol,
                'position': {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'side': position.side
                },
                'hedge_position': {
                    'quantity': hedge_position.quantity if hedge_position else 0.0,
                    'entry_price': hedge_position.entry_price if hedge_position else 0.0,
                    'current_price': hedge_position.current_price if hedge_position else 0.0,
                    'symbol': hedge_position.symbol if hedge_position else None
                } if hedge_position else None,
                'greeks': {
                    'net_delta': portfolio_greeks.net_delta,
                    'net_gamma': portfolio_greeks.net_gamma,
                    'net_theta': portfolio_greeks.net_theta,
                    'net_vega': portfolio_greeks.net_vega
                },
                'risk_metrics': {
                    'var_95': portfolio_greeks.var_95,
                    'expected_shortfall': portfolio_greeks.expected_shortfall,
                    'max_loss_1day': portfolio_greeks.max_loss_1day,
                    'hedge_effectiveness': portfolio_greeks.hedge_effectiveness
                },
                'hedge_performance': hedge_performance,
                'last_rebalance': self.last_rebalance.get(symbol).isoformat() if self.last_rebalance.get(symbol) else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get hedge status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

    async def emergency_hedge_adjustment(self, symbol: str, emergency_type: str, 
                                       severity: float) -> Dict[str, Any]:
        """Execute emergency hedge adjustment during extreme market conditions"""
        try:
            self.logger.warning(f"Emergency hedge adjustment triggered for {symbol}: {emergency_type}")
            
            position = self.positions.get(symbol)
            if not position:
                return {'status': 'no_position', 'action': 'none'}
            
            # Determine emergency action based on type and severity
            if emergency_type == 'flash_crash':
                # Increase hedge ratio to protect against large moves
                emergency_hedge_ratio = min(2.0, 1.0 + severity)
                
            elif emergency_type == 'liquidity_crisis':
                # Reduce hedge ratio due to execution difficulties
                emergency_hedge_ratio = max(0.5, 1.0 - severity * 0.5)
                
            elif emergency_type == 'correlation_breakdown':
                # Increase hedge ratio to compensate for reduced effectiveness
                emergency_hedge_ratio = min(1.5, 1.0 + severity * 0.5)
                
            else:
                # Default emergency response
                emergency_hedge_ratio = 1.2
            
            # Calculate emergency hedge parameters
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                return {'status': 'error', 'error': 'Unable to get current price'}
            
            emergency_params = HedgeParameters(
                hedge_ratio=emergency_hedge_ratio,
                target_delta=0.0,
                rebalance_threshold=0.01,  # Tighter threshold
                hedge_symbol=self._select_hedge_instrument(symbol),
                hedge_quantity=abs(position.quantity * emergency_hedge_ratio),
                confidence=0.3,  # Lower confidence in emergency
                gamma_adjustment=0.0,
                vega_adjustment=0.0
            )
            
            # Execute emergency hedge adjustment
            portfolio_greeks = self._calculate_portfolio_greeks([position])
            emergency_result = await self._execute_hedge_adjustment(symbol, emergency_params, portfolio_greeks)
            
            # Log emergency action
            self.logger.warning(f"Emergency hedge executed for {symbol}: {emergency_result}")
            
            return {
                'status': 'emergency_hedge_executed',
                'emergency_type': emergency_type,
                'severity': severity,
                'emergency_params': {
                    'hedge_ratio': emergency_params.hedge_ratio,
                    'hedge_quantity': emergency_params.hedge_quantity
                },
                'result': emergency_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Emergency hedge adjustment failed: {e}")
            return {
                'status': 'emergency_failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }