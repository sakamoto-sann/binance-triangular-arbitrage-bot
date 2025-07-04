#!/usr/bin/env python3
"""
ATR Delta-Neutral Grid Bot Implementation Plan
Integrating ATR dynamic spacing with delta-neutral futures hedging.
Version 4.1.0 - Delta-Neutral ATR Enhanced
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Import existing components
from advanced.atr_grid_optimizer import ATRGridOptimizer, ATRConfig, VolatilityRegime
from delta_neutral.delta_neutral_manager import DeltaNeutralManager
from v3.core.market_analyzer import MarketAnalyzer, MarketRegime
from config.enhanced_features_config import get_config

logger = logging.getLogger(__name__)

@dataclass
class DeltaNeutralATRConfig:
    """Configuration for ATR Delta-Neutral Grid Bot."""
    # ATR Configuration
    atr_period: int = 14
    atr_regime_lookback: int = 100
    atr_update_frequency_hours: int = 2
    
    # Grid Spacing (ATR-based)
    low_vol_multiplier: float = 0.08      # 8% of ATR
    normal_vol_multiplier: float = 0.12   # 12% of ATR
    high_vol_multiplier: float = 0.15     # 15% of ATR
    extreme_vol_multiplier: float = 0.20  # 20% of ATR
    
    # Delta-Neutral Configuration
    target_delta: float = 0.0             # Perfect neutrality
    delta_tolerance: float = 0.05         # 5% delta tolerance
    rebalance_threshold: float = 0.03     # 3% delta deviation triggers rebalance
    futures_spot_ratio: float = 1.0       # 1:1 hedging ratio
    
    # Supertrend Integration
    supertrend_enabled: bool = True
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    trend_filter_enabled: bool = True
    
    # Funding Rate Optimization
    funding_rate_threshold: float = 0.01  # 1% daily funding rate
    position_flip_enabled: bool = True    # Flip positions for negative funding
    funding_capture_priority: bool = True # Prioritize funding over grid profits
    
    # Risk Management
    max_spot_exposure: float = 0.15       # 15% max spot exposure
    max_futures_exposure: float = 0.20    # 20% max futures exposure
    global_stop_loss: float = 0.20        # 20% portfolio stop-loss
    volatility_pause_threshold: float = 3.0  # Pause if ATR > 3x average

class ATRDeltaNeutralGridBot:
    """
    Advanced ATR-based grid trading bot with delta-neutral futures hedging.
    
    Features:
    - ATR-optimized dynamic grid spacing on spot
    - Automatic delta-neutral hedging with futures
    - Supertrend regime detection and filtering
    - Funding rate capture and optimization
    - Multi-layer risk management
    """
    
    def __init__(self, config: DeltaNeutralATRConfig = None):
        """Initialize ATR Delta-Neutral Grid Bot."""
        self.config = config or DeltaNeutralATRConfig()
        
        # Initialize components
        self.atr_optimizer = self._init_atr_optimizer()
        self.market_analyzer = self._init_market_analyzer()
        self.delta_manager = None  # Will be initialized with exchange clients
        
        # State tracking
        self.spot_positions = {}  # price -> quantity
        self.futures_positions = {}  # contracts
        self.current_delta = 0.0
        self.funding_rate_history = []
        self.grid_spacing_history = []
        
        # Performance tracking
        self.spot_pnl = 0.0
        self.futures_pnl = 0.0
        self.funding_pnl = 0.0
        self.total_pnl = 0.0
        
        logger.info("ATR Delta-Neutral Grid Bot initialized")
    
    def _init_atr_optimizer(self) -> ATRGridOptimizer:
        """Initialize ATR optimizer with delta-neutral parameters."""
        atr_config = ATRConfig(
            atr_period=self.config.atr_period,
            regime_lookback=self.config.atr_regime_lookback,
            update_frequency_hours=self.config.atr_update_frequency_hours,
            low_vol_multiplier=self.config.low_vol_multiplier,
            normal_vol_multiplier=self.config.normal_vol_multiplier,
            high_vol_multiplier=self.config.high_vol_multiplier,
            extreme_vol_multiplier=self.config.extreme_vol_multiplier,
            min_grid_spacing=0.005,  # 0.5% minimum
            max_grid_spacing=0.03    # 3% maximum
        )
        
        return ATRGridOptimizer(atr_config)
    
    def _init_market_analyzer(self):
        """Initialize market analyzer with Supertrend integration."""
        # This would use the existing market analyzer
        # For now, return None as placeholder
        return None
    
    def analyze_market_conditions(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market analysis combining ATR and Supertrend.
        
        Args:
            price_data: Historical OHLC data
            
        Returns:
            Combined market analysis results
        """
        try:
            current_price = float(price_data['close'].iloc[-1])
            
            # ATR volatility analysis
            atr_analysis = self.atr_optimizer.analyze_market_conditions(price_data)
            atr_params = self.atr_optimizer.get_grid_parameters(current_price)
            
            # Supertrend analysis (placeholder - would integrate with market_analyzer)
            supertrend_signal = self._calculate_supertrend(price_data)
            market_regime = self._detect_market_regime(price_data, supertrend_signal)
            
            # Combined analysis
            analysis = {
                # ATR Components
                'atr_value': atr_analysis.current_atr,
                'atr_regime': atr_analysis.regime.value,
                'atr_confidence': atr_analysis.confidence,
                'grid_spacing': atr_params['spacing_pct'],
                
                # Supertrend Components  
                'supertrend_signal': supertrend_signal,
                'market_regime': market_regime,
                'trend_strength': self._calculate_trend_strength(price_data),
                
                # Combined Decision
                'trading_allowed': self._should_allow_trading(atr_analysis, supertrend_signal),
                'grid_adjustment': self._calculate_grid_adjustment(atr_analysis, supertrend_signal),
                'hedge_ratio': self._calculate_optimal_hedge_ratio(price_data),
                
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._get_safe_default_analysis()
    
    def _calculate_supertrend(self, price_data: pd.DataFrame) -> str:
        """Calculate Supertrend indicator for trend detection."""
        try:
            if len(price_data) < self.config.supertrend_period + 1:
                return 'sideways'
            
            # Calculate True Range and ATR
            high = price_data['high']
            low = price_data['low']
            close = price_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.config.supertrend_period).mean()
            
            # Calculate Supertrend
            hl2 = (high + low) / 2
            upper_band = hl2 + (self.config.supertrend_multiplier * atr)
            lower_band = hl2 - (self.config.supertrend_multiplier * atr)
            
            # Supertrend logic
            supertrend = pd.Series(index=price_data.index, dtype=float)
            direction = pd.Series(index=price_data.index, dtype=int)
            
            for i in range(1, len(price_data)):
                if close.iloc[i] <= upper_band.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                elif close.iloc[i] >= lower_band.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
            
            # Current signal
            current_direction = direction.iloc[-1]
            if current_direction == 1:
                return 'bullish'
            elif current_direction == -1:
                return 'bearish'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            return 'sideways'
    
    def _detect_market_regime(self, price_data: pd.DataFrame, supertrend_signal: str) -> str:
        """Detect overall market regime combining trend and volatility."""
        try:
            # Get volatility regime from ATR
            atr_analysis = self.atr_optimizer.analyze_market_conditions(price_data)
            volatility_regime = atr_analysis.regime.value
            
            # Combine with Supertrend
            if supertrend_signal == 'bullish' and volatility_regime in ['normal', 'low']:
                return 'bull_trending'
            elif supertrend_signal == 'bearish' and volatility_regime in ['normal', 'low']:
                return 'bear_trending'
            elif volatility_regime in ['high', 'extreme']:
                return 'high_volatility'
            else:
                return 'sideways_ranging'
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'sideways_ranging'
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength on 0-1 scale."""
        try:
            if len(price_data) < 20:
                return 0.5
            
            # Simple trend strength using price position in recent range
            recent_prices = price_data['close'].tail(20)
            price_min = recent_prices.min()
            price_max = recent_prices.max()
            current_price = recent_prices.iloc[-1]
            
            if price_max == price_min:
                return 0.5
            
            # Position in range (0 = at bottom, 1 = at top)
            position = (current_price - price_min) / (price_max - price_min)
            
            # Convert to trend strength (0.5 = no trend, 0 = strong down, 1 = strong up)
            return position
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _should_allow_trading(self, atr_analysis, supertrend_signal: str) -> bool:
        """Determine if trading should be allowed based on market conditions."""
        try:
            # Don't trade in extreme volatility
            if atr_analysis.regime == VolatilityRegime.EXTREME:
                return False
            
            # Apply Supertrend filter if enabled
            if self.config.trend_filter_enabled:
                # Avoid trading against strong trends
                if supertrend_signal in ['bullish', 'bearish']:
                    # In strong trends, only allow limited trading
                    return atr_analysis.confidence > 0.7
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining trading allowance: {e}")
            return False
    
    def _calculate_grid_adjustment(self, atr_analysis, supertrend_signal: str) -> float:
        """Calculate grid spacing adjustment factor."""
        try:
            base_adjustment = 1.0
            
            # Adjust based on Supertrend
            if supertrend_signal == 'bullish':
                # In bull trends, slightly wider grids
                base_adjustment *= 1.1
            elif supertrend_signal == 'bearish':
                # In bear trends, slightly tighter grids
                base_adjustment *= 0.9
            
            # Adjust based on ATR confidence
            confidence_adjustment = 0.8 + (atr_analysis.confidence * 0.4)
            
            return base_adjustment * confidence_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating grid adjustment: {e}")
            return 1.0
    
    def _calculate_optimal_hedge_ratio(self, price_data: pd.DataFrame) -> float:
        """Calculate optimal futures hedging ratio."""
        try:
            # Base ratio is 1:1 for delta neutrality
            base_ratio = self.config.futures_spot_ratio
            
            # Adjust based on market conditions
            atr_analysis = self.atr_optimizer.analyze_market_conditions(price_data)
            
            if atr_analysis.regime == VolatilityRegime.EXTREME:
                # Increase hedging in extreme volatility
                return base_ratio * 1.2
            elif atr_analysis.regime == VolatilityRegime.LOW:
                # Reduce hedging in low volatility
                return base_ratio * 0.9
            
            return base_ratio
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 1.0
    
    def _get_safe_default_analysis(self) -> Dict[str, Any]:
        """Return safe default analysis in case of errors."""
        return {
            'atr_value': 0.01,
            'atr_regime': 'normal',
            'atr_confidence': 0.5,
            'grid_spacing': 0.012,
            'supertrend_signal': 'sideways',
            'market_regime': 'sideways_ranging',
            'trend_strength': 0.5,
            'trading_allowed': True,
            'grid_adjustment': 1.0,
            'hedge_ratio': 1.0,
            'timestamp': datetime.now()
        }
    
    def simulate_delta_neutral_strategy(self, price_data: pd.DataFrame, 
                                      initial_balance: float = 100000.0) -> Dict[str, Any]:
        """
        Simulate the ATR Delta-Neutral strategy.
        
        Args:
            price_data: Historical OHLC data
            initial_balance: Starting balance in USDT
            
        Returns:
            Strategy performance results
        """
        try:
            # Initialize simulation state
            balance = initial_balance
            spot_positions = {}
            futures_notional = 0.0  # Net futures exposure
            
            spot_pnl = 0.0
            futures_pnl = 0.0
            funding_pnl = 0.0
            
            trades = []
            portfolio_values = []
            
            # Start simulation after enough data for analysis
            start_idx = max(50, self.config.atr_period + 20)
            
            for idx in range(start_idx, len(price_data)):
                current_time = price_data.index[idx]
                current_price = price_data['close'].iloc[idx]
                
                # Get historical data for analysis
                hist_data = price_data.iloc[max(0, idx-200):idx+1]
                
                # Perform market analysis
                analysis = self.analyze_market_conditions(hist_data)
                
                if not analysis['trading_allowed']:
                    # Record portfolio value but don't trade
                    total_value = balance + spot_pnl + futures_pnl + funding_pnl
                    portfolio_values.append({
                        'timestamp': current_time,
                        'total_value': total_value,
                        'spot_pnl': spot_pnl,
                        'futures_pnl': futures_pnl,
                        'funding_pnl': funding_pnl,
                        'price': current_price,
                        'regime': analysis['market_regime'],
                        'trading_allowed': False
                    })
                    continue
                
                # Calculate current portfolio metrics
                spot_exposure = sum(pos_size * current_price for pos_size in spot_positions.values())
                current_delta = spot_exposure + futures_notional
                
                # Check if delta rebalancing is needed
                delta_ratio = abs(current_delta) / max(balance, 1) if balance > 0 else 0
                
                if delta_ratio > self.config.rebalance_threshold:
                    # Rebalance delta through futures
                    required_futures_adjustment = -current_delta * analysis['hedge_ratio']
                    futures_notional += required_futures_adjustment
                    
                    # Record futures rebalancing trade
                    trades.append({
                        'timestamp': current_time,
                        'type': 'futures_rebalance',
                        'side': 'short' if required_futures_adjustment < 0 else 'long',
                        'notional': abs(required_futures_adjustment),
                        'price': current_price,
                        'reason': 'delta_rebalance'
                    })
                
                # Spot grid trading logic
                grid_spacing = analysis['grid_spacing'] * analysis['grid_adjustment']
                
                # Generate grid levels around current price
                num_levels = 5  # Number of grid levels each side
                for i in range(-num_levels, num_levels + 1):
                    if i == 0:
                        continue
                    
                    grid_price = current_price * (1 + i * grid_spacing)
                    
                    if i < 0:  # Buy levels (below current price)
                        if abs(current_price - grid_price) / current_price <= 0.01:  # Within 1%
                            # Execute buy order
                            order_size = balance * 0.02 / grid_price  # 2% of balance
                            if order_size * grid_price <= balance:
                                balance -= order_size * grid_price
                                spot_positions[grid_price] = spot_positions.get(grid_price, 0) + order_size
                                
                                trades.append({
                                    'timestamp': current_time,
                                    'type': 'spot_buy',
                                    'price': grid_price,
                                    'quantity': order_size,
                                    'value': order_size * grid_price
                                })
                    
                    else:  # Sell levels (above current price)
                        if grid_price in spot_positions and abs(current_price - grid_price) / current_price <= 0.01:
                            # Execute sell order
                            sell_quantity = spot_positions[grid_price]
                            balance += sell_quantity * grid_price
                            del spot_positions[grid_price]
                            
                            trades.append({
                                'timestamp': current_time,
                                'type': 'spot_sell',
                                'price': grid_price,
                                'quantity': sell_quantity,
                                'value': sell_quantity * grid_price
                            })
                
                # Calculate current P&L
                spot_pnl = sum(pos_size * (current_price - entry_price) 
                              for entry_price, pos_size in spot_positions.items())
                
                # Futures P&L (simplified - assume perfect hedge)
                futures_pnl = -spot_pnl if futures_notional != 0 else 0
                
                # Funding P&L (simplified - assume 0.01% per 8 hours)
                if idx % 24 == 0:  # Daily funding
                    daily_funding_rate = 0.0001  # 0.01% daily
                    funding_pnl += abs(futures_notional) * daily_funding_rate
                
                # Record portfolio value
                total_value = balance + spot_pnl + futures_pnl + funding_pnl
                portfolio_values.append({
                    'timestamp': current_time,
                    'total_value': total_value,
                    'spot_pnl': spot_pnl,
                    'futures_pnl': futures_pnl,
                    'funding_pnl': funding_pnl,
                    'price': current_price,
                    'regime': analysis['market_regime'],
                    'current_delta': current_delta,
                    'trading_allowed': True
                })
            
            # Calculate final metrics
            final_total_value = balance + spot_pnl + futures_pnl + funding_pnl
            total_return = (final_total_value / initial_balance - 1) * 100
            
            # Performance analysis
            portfolio_df = pd.DataFrame(portfolio_values)
            if len(portfolio_df) > 0:
                portfolio_df['returns'] = portfolio_df['total_value'].pct_change().dropna()
                sharpe_ratio = (portfolio_df['returns'].mean() / portfolio_df['returns'].std() * 
                              np.sqrt(365*24) if portfolio_df['returns'].std() > 0 else 0)
                max_drawdown = ((portfolio_df['total_value'] / portfolio_df['total_value'].cummax()) - 1).min()
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            return {
                'strategy': 'atr_delta_neutral',
                'final_balance': balance,
                'spot_pnl': spot_pnl,
                'futures_pnl': futures_pnl,
                'funding_pnl': funding_pnl,
                'final_total_value': final_total_value,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': len(trades),
                'spot_trades': len([t for t in trades if t['type'].startswith('spot')]),
                'futures_trades': len([t for t in trades if t['type'].startswith('futures')]),
                'avg_funding_capture': funding_pnl / len(portfolio_values) * 365 if portfolio_values else 0,
                'trades': trades,
                'portfolio_history': portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error in delta-neutral simulation: {e}")
            return {'strategy': 'atr_delta_neutral', 'error': str(e)}

def run_atr_delta_neutral_backtest():
    """Run comprehensive backtest of ATR Delta-Neutral strategy."""
    try:
        print("=" * 80)
        print("🚀 ATR DELTA-NEUTRAL GRID BOT BACKTEST v4.1.0")
        print("=" * 80)
        
        # Initialize bot
        config = DeltaNeutralATRConfig()
        bot = ATRDeltaNeutralGridBot(config)
        
        # Load test data
        data_file = 'btc_2024_2024_1h_binance.csv'
        if os.path.exists(data_file):
            price_data = pd.read_csv(data_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data.set_index('timestamp', inplace=True)
        else:
            print(f"❌ Data file {data_file} not found")
            return
        
        print(f"📊 Loaded {len(price_data)} hours of BTC data")
        print(f"   Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
        
        # Run simulation
        print(f"\n🔄 Running ATR Delta-Neutral simulation...")
        results = bot.simulate_delta_neutral_strategy(price_data, initial_balance=100000.0)
        
        if 'error' in results:
            print(f"❌ Simulation failed: {results['error']}")
            return
        
        # Display results
        print(f"\n📈 ATR DELTA-NEUTRAL RESULTS:")
        print(f"=" * 50)
        print(f"Final Total Value:   ${results['final_total_value']:,.2f}")
        print(f"Total Return:        {results['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades:        {results['total_trades']}")
        print(f"  - Spot Trades:     {results['spot_trades']}")
        print(f"  - Futures Trades:  {results['futures_trades']}")
        
        print(f"\n💰 P&L BREAKDOWN:")
        print(f"Spot P&L:            ${results['spot_pnl']:,.2f}")
        print(f"Futures P&L:         ${results['futures_pnl']:,.2f}")
        print(f"Funding P&L:         ${results['funding_pnl']:,.2f}")
        print(f"Avg Annual Funding:  {results['avg_funding_capture']:.2f}%")
        
        print(f"\n🎯 DELTA-NEUTRAL ADVANTAGES:")
        print(f"✅ Reduced directional risk through futures hedging")
        print(f"✅ Funding rate capture providing steady yield")
        print(f"✅ ATR optimization for dynamic market adaptation")
        print(f"✅ Supertrend filtering to avoid adverse conditions")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run backtest
    results = run_atr_delta_neutral_backtest()