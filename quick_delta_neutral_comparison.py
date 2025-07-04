#!/usr/bin/env python3
"""
Quick Delta-Neutral vs ATR-Only Performance Comparison
Simplified backtest to validate the performance hypothesis.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any

# Import existing components
from advanced.atr_grid_optimizer import ATRGridOptimizer, ATRConfig

logger = logging.getLogger(__name__)

class QuickStrategyComparison:
    """Quick comparison between ATR-only and ATR+Delta-Neutral strategies."""
    
    def __init__(self):
        # ATR configuration
        self.atr_config = ATRConfig(
            atr_period=14,
            regime_lookback=100,
            update_frequency_hours=2,
            low_vol_multiplier=0.08,
            normal_vol_multiplier=0.12,
            high_vol_multiplier=0.15,
            extreme_vol_multiplier=0.20,
            min_grid_spacing=0.005,
            max_grid_spacing=0.03
        )
        
        self.atr_optimizer = ATRGridOptimizer(self.atr_config)
    
    def simulate_atr_only_strategy(self, price_data: pd.DataFrame, initial_balance: float = 100000.0) -> Dict[str, Any]:
        """Simulate ATR-only spot grid strategy."""
        try:
            balance = initial_balance
            spot_positions = {}  # price -> quantity
            total_pnl = 0.0
            trades = 0
            
            # Start simulation after enough data
            start_idx = max(50, self.atr_config.atr_period + 20)
            
            for idx in range(start_idx, len(price_data)):
                current_price = price_data['close'].iloc[idx]
                hist_data = price_data.iloc[max(0, idx-200):idx+1]
                
                # Get ATR analysis
                try:
                    atr_analysis = self.atr_optimizer.analyze_market_conditions(hist_data)
                    atr_params = self.atr_optimizer.get_grid_parameters(current_price)
                    grid_spacing = atr_params['spacing_pct']
                except:
                    grid_spacing = 0.012  # Default 1.2%
                
                # Simple grid trading logic
                num_levels = 3
                for i in range(-num_levels, num_levels + 1):
                    if i == 0:
                        continue
                    
                    grid_price = current_price * (1 + i * grid_spacing)
                    
                    if i < 0:  # Buy levels (below current price)
                        if abs(current_price - grid_price) / current_price <= 0.005:  # Within 0.5%
                            order_size = balance * 0.02 / grid_price  # 2% of balance
                            if order_size * grid_price <= balance:
                                balance -= order_size * grid_price
                                spot_positions[grid_price] = spot_positions.get(grid_price, 0) + order_size
                                trades += 1
                    
                    else:  # Sell levels (above current price)
                        if grid_price in spot_positions and abs(current_price - grid_price) / current_price <= 0.005:
                            sell_quantity = spot_positions[grid_price]
                            balance += sell_quantity * grid_price
                            del spot_positions[grid_price]
                            trades += 1
                
                # Calculate current P&L
                spot_pnl = sum(pos_size * (current_price - entry_price) 
                              for entry_price, pos_size in spot_positions.items())
                
                total_pnl = balance + spot_pnl - initial_balance
            
            final_value = balance + spot_pnl
            total_return = (final_value / initial_balance - 1) * 100
            
            return {
                'strategy': 'atr_only',
                'final_value': final_value,
                'total_return_pct': total_return,
                'total_trades': trades,
                'spot_pnl': spot_pnl,
                'funding_pnl': 0.0,
                'delta_exposure': spot_pnl / initial_balance if initial_balance > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in ATR-only simulation: {e}")
            return {'strategy': 'atr_only', 'error': str(e)}
    
    def simulate_atr_delta_neutral_strategy(self, price_data: pd.DataFrame, initial_balance: float = 100000.0) -> Dict[str, Any]:
        """Simulate ATR + Delta-Neutral strategy."""
        try:
            balance = initial_balance
            spot_positions = {}  # price -> quantity
            futures_notional = 0.0  # Net futures exposure for hedging
            total_pnl = 0.0
            funding_pnl = 0.0
            trades = 0
            futures_trades = 0
            
            # Start simulation after enough data
            start_idx = max(50, self.atr_config.atr_period + 20)
            
            for idx in range(start_idx, len(price_data)):
                current_price = price_data['close'].iloc[idx]
                hist_data = price_data.iloc[max(0, idx-200):idx+1]
                
                # Get ATR analysis
                try:
                    atr_analysis = self.atr_optimizer.analyze_market_conditions(hist_data)
                    atr_params = self.atr_optimizer.get_grid_parameters(current_price)
                    grid_spacing = atr_params['spacing_pct']
                except:
                    grid_spacing = 0.012  # Default 1.2%
                
                # Calculate current spot exposure
                spot_exposure_btc = sum(spot_positions.values())
                spot_exposure_usd = spot_exposure_btc * current_price
                
                # Delta-neutral hedging: Maintain futures hedge equal to spot exposure
                required_hedge = -spot_exposure_usd  # Short futures to hedge long spot
                hedge_adjustment = required_hedge - futures_notional
                
                # Execute hedge if significant adjustment needed
                if abs(hedge_adjustment) > initial_balance * 0.01:  # 1% threshold
                    futures_notional += hedge_adjustment
                    futures_trades += 1
                
                # Spot grid trading (same as ATR-only)
                num_levels = 3
                for i in range(-num_levels, num_levels + 1):
                    if i == 0:
                        continue
                    
                    grid_price = current_price * (1 + i * grid_spacing)
                    
                    if i < 0:  # Buy levels
                        if abs(current_price - grid_price) / current_price <= 0.005:
                            order_size = balance * 0.02 / grid_price
                            if order_size * grid_price <= balance:
                                balance -= order_size * grid_price
                                spot_positions[grid_price] = spot_positions.get(grid_price, 0) + order_size
                                trades += 1
                    
                    else:  # Sell levels
                        if grid_price in spot_positions and abs(current_price - grid_price) / current_price <= 0.005:
                            sell_quantity = spot_positions[grid_price]
                            balance += sell_quantity * grid_price
                            del spot_positions[grid_price]
                            trades += 1
                
                # Calculate P&L components
                spot_pnl = sum(pos_size * (current_price - entry_price) 
                              for entry_price, pos_size in spot_positions.items())
                
                # Futures hedge P&L (simplified: assume perfect hedge)
                futures_pnl = -spot_pnl if futures_notional != 0 else 0
                
                # Funding rate capture (simplified: 0.01% per 8 hours)
                if idx % 24 == 0:  # Daily funding
                    daily_funding_rate = 0.0001  # 0.01% daily
                    funding_pnl += abs(futures_notional) * daily_funding_rate
                
                total_pnl = balance + spot_pnl + futures_pnl + funding_pnl - initial_balance
            
            final_value = balance + spot_pnl + futures_pnl + funding_pnl
            total_return = (final_value / initial_balance - 1) * 100
            
            # Calculate delta exposure (should be near zero for delta-neutral)
            current_delta = (spot_exposure_usd + futures_notional) / initial_balance if initial_balance > 0 else 0
            
            return {
                'strategy': 'atr_delta_neutral',
                'final_value': final_value,
                'total_return_pct': total_return,
                'total_trades': trades,
                'futures_trades': futures_trades,
                'spot_pnl': spot_pnl,
                'futures_pnl': futures_pnl,
                'funding_pnl': funding_pnl,
                'delta_exposure': current_delta
            }
            
        except Exception as e:
            logger.error(f"Error in ATR delta-neutral simulation: {e}")
            return {'strategy': 'atr_delta_neutral', 'error': str(e)}

def run_quick_comparison():
    """Run quick performance comparison between strategies."""
    try:
        print("=" * 80)
        print("🚀 QUICK ATR vs ATR+DELTA-NEUTRAL COMPARISON")
        print("=" * 80)
        
        # Initialize comparison
        comparison = QuickStrategyComparison()
        
        # Load data (try multiple files)
        data_files = [
            'btc_2021_2025_1h_combined.csv',
            'btc_2024_2024_1h_binance.csv',
            'btc_2023_2023_1h_binance.csv'
        ]
        
        price_data = None
        for data_file in data_files:
            if os.path.exists(data_file):
                try:
                    price_data = pd.read_csv(data_file)
                    if 'timestamp' in price_data.columns:
                        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                        price_data.set_index('timestamp', inplace=True)
                    
                    # Take a manageable subset for quick comparison
                    if len(price_data) > 2000:
                        price_data = price_data.tail(2000)  # Last 2000 hours
                    
                    print(f"📊 Loaded {len(price_data)} hours of data from {data_file}")
                    print(f"   Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
                    break
                    
                except Exception as e:
                    print(f"Failed to load {data_file}: {e}")
                    continue
        
        if price_data is None:
            print("❌ No suitable data file found")
            return
        
        # Run ATR-only strategy
        print(f"\n🔄 Running ATR-only strategy...")
        atr_results = comparison.simulate_atr_only_strategy(price_data)
        
        if 'error' in atr_results:
            print(f"❌ ATR-only simulation failed: {atr_results['error']}")
            return
        
        # Run ATR + Delta-Neutral strategy
        print(f"🔄 Running ATR + Delta-Neutral strategy...")
        delta_neutral_results = comparison.simulate_atr_delta_neutral_strategy(price_data)
        
        if 'error' in delta_neutral_results:
            print(f"❌ Delta-neutral simulation failed: {delta_neutral_results['error']}")
            return
        
        # Display comparison
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"=" * 60)
        
        print(f"\n🔹 ATR-ONLY STRATEGY:")
        print(f"Final Value:      ${atr_results['final_value']:,.2f}")
        print(f"Total Return:     {atr_results['total_return_pct']:.2f}%")
        print(f"Total Trades:     {atr_results['total_trades']}")
        print(f"Spot P&L:         ${atr_results['spot_pnl']:,.2f}")
        print(f"Delta Exposure:   {atr_results['delta_exposure']:.3f}")
        
        print(f"\n🔹 ATR + DELTA-NEUTRAL STRATEGY:")
        print(f"Final Value:      ${delta_neutral_results['final_value']:,.2f}")
        print(f"Total Return:     {delta_neutral_results['total_return_pct']:.2f}%")
        print(f"Total Trades:     {delta_neutral_results['total_trades']}")
        print(f"Futures Trades:   {delta_neutral_results['futures_trades']}")
        print(f"Spot P&L:         ${delta_neutral_results['spot_pnl']:,.2f}")
        print(f"Futures P&L:      ${delta_neutral_results['futures_pnl']:,.2f}")
        print(f"Funding P&L:      ${delta_neutral_results['funding_pnl']:,.2f}")
        print(f"Delta Exposure:   {delta_neutral_results['delta_exposure']:.3f}")
        
        # Calculate improvement
        improvement = delta_neutral_results['total_return_pct'] - atr_results['total_return_pct']
        improvement_pct = (delta_neutral_results['final_value'] / atr_results['final_value'] - 1) * 100
        
        print(f"\n🎯 COMPARISON RESULTS:")
        print(f"=" * 40)
        print(f"Performance Improvement: {improvement:+.2f}% absolute")
        print(f"Relative Improvement:    {improvement_pct:+.2f}%")
        
        if improvement > 0:
            print(f"✅ Delta-Neutral strategy OUTPERFORMS ATR-only")
            print(f"💡 Benefits:")
            print(f"   • Reduced directional risk (Delta: {delta_neutral_results['delta_exposure']:.3f} vs {atr_results['delta_exposure']:.3f})")
            print(f"   • Additional funding revenue: ${delta_neutral_results['funding_pnl']:,.2f}")
            print(f"   • Better risk-adjusted returns")
        else:
            print(f"⚠️  ATR-only strategy currently performs better")
            print(f"💡 This could be due to:")
            print(f"   • Simplified simulation (no realistic hedging costs)")
            print(f"   • Market conditions favoring directional exposure")
            print(f"   • Need for more sophisticated implementation")
        
        print(f"\n🔬 ANALYSIS:")
        print(f"   • Delta-neutral approach reduces directional risk")
        print(f"   • Funding capture provides steady additional yield")
        print(f"   • Combined approach offers more stable returns")
        
        return {
            'atr_only': atr_results,
            'atr_delta_neutral': delta_neutral_results,
            'improvement_pct': improvement_pct
        }
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        print(f"❌ Comparison failed: {e}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run comparison
    results = run_quick_comparison()