#!/usr/bin/env python3
"""
ATR-Enhanced Grid Trading Bot Backtest
Comprehensive backtesting of ATR-based dynamic grid spacing vs static grid spacing.
Version 4.0.0 - Conservative ATR Enhancement
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json

# Import our enhanced components
from advanced.atr_grid_optimizer import ATRGridOptimizer, ATRConfig, VolatilityRegime
from config.enhanced_features_config import get_config, is_feature_enabled

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ATRGridBacktester:
    """
    Comprehensive backtester for ATR-enhanced grid trading strategy.
    """
    
    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting portfolio balance in USDT
        """
        self.initial_balance = initial_balance
        self.commission_rate = 0.001  # 0.1% commission
        
        # Enhanced features configuration
        self.enhanced_config = get_config()
        self.atr_config_dict = self.enhanced_config.get('atr_grid_optimization', {})
        
        # Initialize ATR optimizer
        atr_config = ATRConfig(
            atr_period=self.atr_config_dict.get('atr_period', 14),
            regime_lookback=self.atr_config_dict.get('regime_lookback', 100),
            low_vol_multiplier=self.atr_config_dict.get('low_vol_multiplier', 0.08),
            normal_vol_multiplier=self.atr_config_dict.get('normal_vol_multiplier', 0.12),
            high_vol_multiplier=self.atr_config_dict.get('high_vol_multiplier', 0.15),
            extreme_vol_multiplier=self.atr_config_dict.get('extreme_vol_multiplier', 0.20),
            min_grid_spacing=self.atr_config_dict.get('min_grid_spacing', 0.001),
            max_grid_spacing=self.atr_config_dict.get('max_grid_spacing', 0.05)
        )
        
        self.atr_optimizer = ATRGridOptimizer(atr_config)
        
        # Grid configuration
        self.grid_levels = 10
        self.position_size = 0.1  # 10% of balance per grid level
        self.rebalance_threshold = 0.02  # 2% price movement triggers rebalance
        
        logger.info("ATR Grid Backtester initialized")
    
    def generate_realistic_crypto_data(self, start_date: datetime, end_date: datetime, 
                                     initial_price: float = 55000.0) -> pd.DataFrame:
        """
        Generate realistic cryptocurrency price data for backtesting.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            initial_price: Starting price
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Calculate number of hours between dates
            total_hours = int((end_date - start_date).total_seconds() / 3600)
            
            # Generate hourly timestamps
            timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
            
            # Price simulation with realistic crypto characteristics
            np.random.seed(42)  # For reproducible results
            
            prices = []
            current_price = initial_price
            
            # Market regime simulation
            regime_length = 24 * 7  # 1 week regimes
            regimes = ['bear', 'sideways', 'bull']
            
            for i, timestamp in enumerate(timestamps):
                # Determine current regime
                regime_index = (i // regime_length) % len(regimes)
                current_regime = regimes[regime_index]
                
                # Regime-specific parameters
                if current_regime == 'bear':
                    trend = -0.0001  # Slight downward trend
                    volatility = np.random.uniform(0.02, 0.08)  # High volatility
                elif current_regime == 'bull':
                    trend = 0.0003  # Upward trend
                    volatility = np.random.uniform(0.015, 0.05)  # Moderate volatility
                else:  # sideways
                    trend = 0.0  # No trend
                    volatility = np.random.uniform(0.01, 0.03)  # Low volatility
                
                # Price movement with volatility clustering
                price_change = np.random.normal(trend, volatility)
                
                # Add some autocorrelation for realism
                if i > 0:
                    prev_change = prices[-1]['close'] / (prices[-1]['open'] if prices[-1]['open'] > 0 else current_price) - 1
                    price_change += 0.1 * prev_change  # 10% momentum
                
                # Update price
                current_price *= (1 + price_change)
                
                # Generate OHLC from current price
                intraday_volatility = volatility * 0.5
                
                high = current_price * (1 + np.random.uniform(0, intraday_volatility))
                low = current_price * (1 - np.random.uniform(0, intraday_volatility))
                open_price = current_price * (1 + np.random.uniform(-intraday_volatility/2, intraday_volatility/2))
                close_price = current_price
                
                # Ensure OHLC consistency
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                prices.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': np.random.uniform(1000, 50000),  # Random volume
                    'regime': current_regime
                })
            
            df = pd.DataFrame(prices)
            df['returns'] = df['close'].pct_change()
            
            logger.info(f"Generated {len(df)} hours of crypto data from {start_date} to {end_date}")
            logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            logger.info(f"Regime distribution: {df['regime'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating crypto data: {e}")
            return pd.DataFrame()
    
    def simulate_static_grid_strategy(self, price_data: pd.DataFrame, 
                                    static_spacing: float = 0.01) -> Dict[str, Any]:
        """
        Simulate traditional static grid trading strategy.
        
        Args:
            price_data: Historical price data
            static_spacing: Fixed grid spacing (1% default)
            
        Returns:
            Strategy performance results
        """
        try:
            balance = self.initial_balance
            positions = {}  # price -> quantity
            trades = []
            portfolio_values = []
            
            # Initialize grid around first price
            initial_price = price_data['close'].iloc[20]  # Use price after some history
            grid_center = initial_price
            last_rebalance_price = initial_price
            
            # Create initial grid
            grid_prices = []
            for i in range(-self.grid_levels//2, self.grid_levels//2 + 1):
                if i != 0:  # Skip center price
                    grid_price = grid_center * (1 + i * static_spacing)
                    grid_prices.append(grid_price)
            
            total_trades = 0
            profitable_trades = 0
            
            for idx, row in price_data.iterrows():
                if idx < 20:  # Skip initial period
                    continue
                
                current_price = row['close']
                current_time = row['timestamp']
                
                # Check for grid fills
                for grid_price in grid_prices[:]:  # Copy list to avoid modification during iteration
                    if grid_price < current_price and grid_price not in positions:
                        # Buy order (price below current)
                        if (current_price - grid_price) / grid_price <= 0.005:  # Within 0.5% tolerance
                            quantity = (balance * self.position_size / self.grid_levels) / grid_price
                            cost = quantity * grid_price * (1 + self.commission_rate)
                            
                            if cost <= balance:
                                balance -= cost
                                positions[grid_price] = quantity
                                
                                trades.append({
                                    'timestamp': current_time,
                                    'side': 'buy',
                                    'price': grid_price,
                                    'quantity': quantity,
                                    'value': quantity * grid_price,
                                    'commission': quantity * grid_price * self.commission_rate
                                })
                                total_trades += 1
                    
                    elif grid_price > current_price and grid_price in positions:
                        # Sell order (price above current, and we have position)
                        if (grid_price - current_price) / current_price <= 0.005:  # Within 0.5% tolerance
                            quantity = positions[grid_price]
                            revenue = quantity * grid_price * (1 - self.commission_rate)
                            
                            balance += revenue
                            del positions[grid_price]
                            
                            trades.append({
                                'timestamp': current_time,
                                'side': 'sell',
                                'price': grid_price,
                                'quantity': quantity,
                                'value': quantity * grid_price,
                                'commission': quantity * grid_price * self.commission_rate
                            })
                            total_trades += 1
                            
                            # Check if profitable (simplified)
                            buy_trades = [t for t in trades if t['side'] == 'buy' and t['price'] < grid_price]
                            if buy_trades:
                                profitable_trades += 1
                
                # Check if rebalancing is needed
                price_deviation = abs(current_price - last_rebalance_price) / last_rebalance_price
                if price_deviation > self.rebalance_threshold:
                    # Rebalance grid
                    grid_center = current_price
                    last_rebalance_price = current_price
                    
                    # Create new grid
                    grid_prices = []
                    for i in range(-self.grid_levels//2, self.grid_levels//2 + 1):
                        if i != 0:
                            grid_price = grid_center * (1 + i * static_spacing)
                            grid_prices.append(grid_price)
                
                # Calculate portfolio value
                position_value = sum(positions[price] * current_price for price in positions)
                total_value = balance + position_value
                portfolio_values.append({
                    'timestamp': current_time,
                    'balance': balance,
                    'position_value': position_value,
                    'total_value': total_value,
                    'price': current_price
                })
            
            # Final portfolio value
            final_price = price_data['close'].iloc[-1]
            final_position_value = sum(positions[price] * final_price for price in positions)
            final_total_value = balance + final_position_value
            
            # Calculate performance metrics
            total_return = (final_total_value / self.initial_balance - 1) * 100
            
            # Calculate returns for Sharpe ratio
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change().dropna()
            
            if len(portfolio_df['returns']) > 0:
                sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(24*365) if portfolio_df['returns'].std() > 0 else 0
                max_drawdown = ((portfolio_df['total_value'] / portfolio_df['total_value'].cummax()) - 1).min()
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            win_rate = profitable_trades / max(1, total_trades) * 100
            
            return {
                'strategy': 'static_grid',
                'final_balance': balance,
                'final_position_value': final_position_value,
                'final_total_value': final_total_value,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate_pct': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'static_spacing_pct': static_spacing * 100,
                'trades': trades,
                'portfolio_history': portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error in static grid simulation: {e}")
            return {'strategy': 'static_grid', 'error': str(e)}
    
    def simulate_atr_grid_strategy(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate ATR-enhanced dynamic grid trading strategy.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Strategy performance results
        """
        try:
            balance = self.initial_balance
            positions = {}  # price -> quantity
            trades = []
            portfolio_values = []
            grid_spacing_history = []
            
            # Track ATR optimization
            atr_updates = 0
            regime_changes = 0
            
            # Start simulation after enough data for ATR
            start_idx = self.atr_optimizer.config.atr_period + 50
            initial_price = price_data['close'].iloc[start_idx]
            grid_center = initial_price
            last_rebalance_price = initial_price
            current_spacing = 0.01  # Default 1%
            
            # Initial grid
            grid_prices = []
            last_regime = None
            
            total_trades = 0
            profitable_trades = 0
            
            for idx, row in price_data.iterrows():
                if idx < start_idx:
                    continue
                
                current_price = row['close']
                current_time = row['timestamp']
                
                # Get historical data for ATR calculation
                hist_data = price_data.iloc[max(0, idx-200):idx+1]  # Last 200 periods
                
                # Analyze market conditions and get optimal spacing
                if len(hist_data) >= self.atr_optimizer.config.atr_period:
                    try:
                        volatility_analysis = self.atr_optimizer.analyze_market_conditions(hist_data)
                        grid_params = self.atr_optimizer.get_grid_parameters(current_price)
                        
                        new_spacing = grid_params['spacing_pct']
                        current_regime = grid_params['regime']
                        
                        # Track regime changes
                        if last_regime and last_regime != current_regime:
                            regime_changes += 1
                        last_regime = current_regime
                        
                        # Update spacing if significantly different or regime changed
                        spacing_change = abs(new_spacing - current_spacing) / current_spacing
                        if spacing_change > 0.1 or regime_changes > 0:  # 10% change threshold
                            current_spacing = new_spacing
                            atr_updates += 1
                            
                            # Record spacing change
                            grid_spacing_history.append({
                                'timestamp': current_time,
                                'spacing_pct': current_spacing * 100,
                                'regime': current_regime,
                                'atr_value': grid_params.get('atr', 0),
                                'confidence': grid_params.get('confidence', 0)
                            })
                    
                    except Exception as e:
                        logger.warning(f"ATR calculation failed, using default spacing: {e}")
                        current_spacing = 0.01
                
                # Check for grid fills (same logic as static but with dynamic spacing)
                for grid_price in grid_prices[:]:
                    if grid_price < current_price and grid_price not in positions:
                        # Buy order
                        if (current_price - grid_price) / grid_price <= 0.005:
                            quantity = (balance * self.position_size / self.grid_levels) / grid_price
                            cost = quantity * grid_price * (1 + self.commission_rate)
                            
                            if cost <= balance:
                                balance -= cost
                                positions[grid_price] = quantity
                                
                                trades.append({
                                    'timestamp': current_time,
                                    'side': 'buy',
                                    'price': grid_price,
                                    'quantity': quantity,
                                    'value': quantity * grid_price,
                                    'commission': quantity * grid_price * self.commission_rate,
                                    'spacing_used': current_spacing
                                })
                                total_trades += 1
                    
                    elif grid_price > current_price and grid_price in positions:
                        # Sell order
                        if (grid_price - current_price) / current_price <= 0.005:
                            quantity = positions[grid_price]
                            revenue = quantity * grid_price * (1 - self.commission_rate)
                            
                            balance += revenue
                            del positions[grid_price]
                            
                            trades.append({
                                'timestamp': current_time,
                                'side': 'sell',
                                'price': grid_price,
                                'quantity': quantity,
                                'value': quantity * grid_price,
                                'commission': quantity * grid_price * self.commission_rate,
                                'spacing_used': current_spacing
                            })
                            total_trades += 1
                            profitable_trades += 1
                
                # Check if rebalancing is needed (more sophisticated for ATR)
                price_deviation = abs(current_price - last_rebalance_price) / last_rebalance_price
                atr_rebalance = self.atr_optimizer.should_update_grid() if hasattr(self.atr_optimizer, 'should_update_grid') else False
                
                if price_deviation > self.rebalance_threshold or atr_rebalance:
                    # Rebalance grid with current ATR spacing
                    grid_center = current_price
                    last_rebalance_price = current_price
                    
                    # Create new grid with dynamic spacing
                    grid_prices = []
                    for i in range(-self.grid_levels//2, self.grid_levels//2 + 1):
                        if i != 0:
                            grid_price = grid_center * (1 + i * current_spacing)
                            grid_prices.append(grid_price)
                
                # Calculate portfolio value
                position_value = sum(positions[price] * current_price for price in positions)
                total_value = balance + position_value
                portfolio_values.append({
                    'timestamp': current_time,
                    'balance': balance,
                    'position_value': position_value,
                    'total_value': total_value,
                    'price': current_price,
                    'spacing_pct': current_spacing * 100,
                    'regime': last_regime or 'unknown'
                })
            
            # Final calculations
            final_price = price_data['close'].iloc[-1]
            final_position_value = sum(positions[price] * final_price for price in positions)
            final_total_value = balance + final_position_value
            
            total_return = (final_total_value / self.initial_balance - 1) * 100
            
            # Calculate performance metrics
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change().dropna()
            
            if len(portfolio_df['returns']) > 0:
                sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(24*365) if portfolio_df['returns'].std() > 0 else 0
                max_drawdown = ((portfolio_df['total_value'] / portfolio_df['total_value'].cummax()) - 1).min()
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            win_rate = profitable_trades / max(1, total_trades) * 100
            
            # ATR-specific metrics
            atr_metrics = self.atr_optimizer.get_performance_metrics()
            
            return {
                'strategy': 'atr_dynamic_grid',
                'final_balance': balance,
                'final_position_value': final_position_value,
                'final_total_value': final_total_value,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate_pct': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'atr_updates': atr_updates,
                'regime_changes': regime_changes,
                'avg_spacing_pct': np.mean([g['spacing_pct'] for g in grid_spacing_history]) if grid_spacing_history else 1.0,
                'spacing_std_pct': np.std([g['spacing_pct'] for g in grid_spacing_history]) if grid_spacing_history else 0.0,
                'atr_metrics': atr_metrics,
                'trades': trades,
                'portfolio_history': portfolio_values,
                'spacing_history': grid_spacing_history
            }
            
        except Exception as e:
            logger.error(f"Error in ATR grid simulation: {e}")
            return {'strategy': 'atr_dynamic_grid', 'error': str(e)}
    
    def compare_strategies(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare static vs ATR-enhanced grid strategies.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Comprehensive comparison results
        """
        try:
            logger.info("Running static grid strategy backtest...")
            static_results = self.simulate_static_grid_strategy(price_data, static_spacing=0.01)
            
            logger.info("Running ATR-enhanced grid strategy backtest...")
            atr_results = self.simulate_atr_grid_strategy(price_data)
            
            # Calculate improvement metrics
            if 'error' not in static_results and 'error' not in atr_results:
                return_improvement = ((atr_results['total_return_pct'] - static_results['total_return_pct']) 
                                    / abs(static_results['total_return_pct']) * 100 if static_results['total_return_pct'] != 0 else 0)
                
                sharpe_improvement = ((atr_results['sharpe_ratio'] - static_results['sharpe_ratio']) 
                                    / abs(static_results['sharpe_ratio']) * 100 if static_results['sharpe_ratio'] != 0 else 0)
                
                drawdown_improvement = ((static_results['max_drawdown_pct'] - atr_results['max_drawdown_pct']) 
                                      / abs(static_results['max_drawdown_pct']) * 100 if static_results['max_drawdown_pct'] != 0 else 0)
                
                trades_improvement = ((atr_results['total_trades'] - static_results['total_trades']) 
                                    / max(1, static_results['total_trades']) * 100)
                
                comparison = {
                    'static_strategy': static_results,
                    'atr_strategy': atr_results,
                    'improvements': {
                        'return_improvement_pct': return_improvement,
                        'sharpe_improvement_pct': sharpe_improvement,
                        'drawdown_improvement_pct': drawdown_improvement,
                        'trades_improvement_pct': trades_improvement
                    },
                    'winner': 'atr_enhanced' if atr_results['total_return_pct'] > static_results['total_return_pct'] else 'static'
                }
                
                logger.info(f"Strategy comparison completed:")
                logger.info(f"  Static return: {static_results['total_return_pct']:.2f}%")
                logger.info(f"  ATR return: {atr_results['total_return_pct']:.2f}%")
                logger.info(f"  Improvement: {return_improvement:.2f}%")
                logger.info(f"  Winner: {comparison['winner']}")
                
                return comparison
            else:
                return {
                    'static_strategy': static_results,
                    'atr_strategy': atr_results,
                    'error': 'One or both strategies failed'
                }
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {'error': str(e)}

def run_comprehensive_backtest():
    """Run comprehensive backtest of ATR-enhanced grid trading."""
    try:
        print("=" * 80)
        print("🚀 ATR-ENHANCED GRID TRADING BACKTEST v4.0.0")
        print("=" * 80)
        
        # Initialize backtester
        backtester = ATRGridBacktester(initial_balance=100000.0)
        
        # Generate test data (3 months of hourly data)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 4, 1)
        
        print(f"📊 Generating crypto data from {start_date.date()} to {end_date.date()}")
        price_data = backtester.generate_realistic_crypto_data(start_date, end_date, initial_price=55000.0)
        
        if price_data.empty:
            print("❌ Failed to generate price data")
            return
        
        print(f"✅ Generated {len(price_data)} hours of data")
        print(f"   Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
        print(f"   Total return (buy & hold): {((price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # Run strategy comparison
        print(f"\n🔄 Running strategy comparison...")
        results = backtester.compare_strategies(price_data)
        
        if 'error' in results:
            print(f"❌ Backtest failed: {results['error']}")
            return
        
        # Display results
        print(f"\n📈 BACKTEST RESULTS:")
        print(f"=" * 50)
        
        static = results['static_strategy']
        atr = results['atr_strategy']
        improvements = results['improvements']
        
        print(f"📊 STATIC GRID STRATEGY:")
        print(f"   Final Value: ${static['final_total_value']:,.2f}")
        print(f"   Total Return: {static['total_return_pct']:.2f}%")
        print(f"   Sharpe Ratio: {static['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {static['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {static['total_trades']}")
        print(f"   Win Rate: {static['win_rate_pct']:.1f}%")
        print(f"   Grid Spacing: {static['static_spacing_pct']:.1f}% (fixed)")
        
        print(f"\n⚡ ATR-ENHANCED GRID STRATEGY:")
        print(f"   Final Value: ${atr['final_total_value']:,.2f}")
        print(f"   Total Return: {atr['total_return_pct']:.2f}%")
        print(f"   Sharpe Ratio: {atr['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {atr['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {atr['total_trades']}")
        print(f"   Win Rate: {atr['win_rate_pct']:.1f}%")
        print(f"   Avg Grid Spacing: {atr['avg_spacing_pct']:.2f}% ± {atr['spacing_std_pct']:.2f}%")
        print(f"   ATR Updates: {atr['atr_updates']}")
        print(f"   Regime Changes: {atr['regime_changes']}")
        
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"   Winner: {'🏆 ATR-Enhanced' if results['winner'] == 'atr_enhanced' else '🏆 Static Grid'}")
        print(f"   Return Improvement: {improvements['return_improvement_pct']:+.2f}%")
        print(f"   Sharpe Improvement: {improvements['sharpe_improvement_pct']:+.2f}%")
        print(f"   Drawdown Improvement: {improvements['drawdown_improvement_pct']:+.2f}%")
        print(f"   Trade Count Change: {improvements['trades_improvement_pct']:+.2f}%")
        
        # ATR-specific insights
        if 'atr_metrics' in atr:
            metrics = atr['atr_metrics']
            print(f"\n🔍 ATR OPTIMIZATION INSIGHTS:")
            print(f"   Current Regime: {metrics.get('atr_current_regime', 'unknown')}")
            print(f"   Average Confidence: {metrics.get('atr_average_confidence', 0):.2f}")
            print(f"   Fallback Rate: {metrics.get('fallback_rate', 0):.1%}")
            
            if 'atr_regime_distribution' in metrics:
                print(f"   Regime Distribution:")
                for regime, pct in metrics['atr_regime_distribution'].items():
                    print(f"     {regime}: {pct:.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"atr_backtest_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'backtest_config': {
                'initial_balance': backtester.initial_balance,
                'data_period': f"{start_date.date()} to {end_date.date()}",
                'data_points': len(price_data),
                'atr_config': backtester.atr_config_dict
            },
            'static_strategy': {
                k: v for k, v in static.items() 
                if k not in ['trades', 'portfolio_history']  # Exclude large data
            },
            'atr_strategy': {
                k: v for k, v in atr.items() 
                if k not in ['trades', 'portfolio_history', 'spacing_history']  # Exclude large data
            },
            'improvements': improvements,
            'winner': results['winner']
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Conclusion
        print(f"\n🎉 BACKTEST COMPLETED!")
        if results['winner'] == 'atr_enhanced':
            print(f"✅ ATR-Enhanced strategy outperformed static grid by {improvements['return_improvement_pct']:.2f}%")
            print(f"🚀 Ready for paper trading validation!")
        else:
            print(f"⚠️  Static strategy performed better - consider parameter tuning")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Review detailed results in {results_file}")
        print(f"   2. Run paper trading validation")  
        print(f"   3. Deploy to production if results are satisfactory")
        print(f"   4. Monitor performance vs baseline")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")
        return None

if __name__ == "__main__":
    results = run_comprehensive_backtest()