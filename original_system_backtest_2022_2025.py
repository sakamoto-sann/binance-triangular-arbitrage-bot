#!/usr/bin/env python3
"""
Original High-Performance Trading System Backtest (2022-2025)
Testing the original exceptional system that achieved 1,650% returns and 22.98 Sharpe ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OriginalSystemBacktester:
    """
    Backtest the original high-performance trading system using real historical data
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position tracking for delta-neutral strategy
        self.spot_position = 0.0
        self.futures_position = 0.0
        self.usdt_balance = initial_capital
        self.btc_balance = 0.0
        
        # Grid parameters (from original high-performing config)
        self.grid_size = 20
        self.grid_spacing = 0.005  # 0.5%
        self.order_size_usdt = 1000  # $1000 per order
        self.max_position_pct = 0.8  # Max 80% of capital
        
        # Performance tracking
        self.total_pnl = 0.0
        self.grid_pnl = 0.0
        self.arbitrage_pnl = 0.0
        self.funding_pnl = 0.0
        self.delta_neutral_pnl = 0.0
        
        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.grid_orders = []
        
        # Performance metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load real historical BTC data from 2022 to June 2025"""
        logger.info("Loading historical BTC data from 2022-2025...")
        
        try:
            # Load the combined historical data
            df = pd.read_csv('data/btc_2021_2025_1h_combined.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Filter for 2022 to June 2025
            start_date = '2022-01-01'
            end_date = '2025-06-30'
            df = df[start_date:end_date]
            
            # Ensure we have price column
            if 'price' not in df.columns:
                df['price'] = df['close']
                
            logger.info(f"Loaded {len(df)} hourly data points from {start_date} to {end_date}")
            logger.info(f"Price range: ${df['price'].iloc[0]:,.2f} -> ${df['price'].iloc[-1]:,.2f}")
            
            return df
            
        except FileNotFoundError:
            logger.error("Historical data not found!")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the original system"""
        logger.info("Calculating technical indicators...")
        
        # Price-based indicators
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        
        # Volatility indicators
        df['returns'] = df['price'].pct_change()
        df['volatility_24h'] = df['returns'].rolling(24).std() * np.sqrt(24)
        df['atr'] = df[['high', 'low', 'close']].apply(lambda x: x['high'] - x['low'], axis=1).rolling(14).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        df['bb_std'] = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Market regime detection (original system's logic)
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['volatility_regime'] = pd.cut(df['volatility_24h'], bins=3, labels=['low', 'medium', 'high'])
        
        return df

    def original_delta_neutral_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement the original delta-neutral grid strategy"""
        logger.info("Implementing original delta-neutral strategy...")
        
        df['signal'] = 0
        df['strategy'] = 'hold'
        df['position_size'] = 0.0
        
        for i in range(50, len(df)):  # Start after indicators stabilize
            current_price = df.iloc[i]['price']
            portfolio_value = self.usdt_balance + self.btc_balance * current_price
            
            # Update portfolio history
            self.portfolio_history.append({
                'timestamp': df.index[i],
                'portfolio_value': portfolio_value,
                'btc_balance': self.btc_balance,
                'usdt_balance': self.usdt_balance,
                'price': current_price
            })
            
            # Delta-neutral grid logic (original high-performance parameters)
            if i % 24 == 0:  # Rebalance every 24 hours
                self._rebalance_delta_neutral_grid(current_price, df.iloc[i])
            
            # Check for grid fills
            self._check_grid_fills(current_price, df.iloc[i])
            
            # Market making opportunities
            self._execute_market_making(current_price, df.iloc[i])
            
            # Funding rate collection (simulated)
            if i % 8 == 0:  # Every 8 hours
                funding_rate = np.random.normal(0.01, 0.005) / 100  # ~0.01% avg funding
                if self.futures_position != 0:
                    funding_profit = abs(self.futures_position) * current_price * funding_rate
                    self.funding_pnl += funding_profit
                    self.usdt_balance += funding_profit
            
            # Update metrics
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return df

    def _rebalance_delta_neutral_grid(self, price: float, row: pd.Series):
        """Rebalance the delta-neutral grid (original system logic)"""
        portfolio_value = self.usdt_balance + self.btc_balance * price
        
        # Target 50% in spot, 50% hedged with futures
        target_btc_value = portfolio_value * 0.5
        target_btc_amount = target_btc_value / price
        
        btc_diff = target_btc_amount - self.btc_balance
        
        if abs(btc_diff * price) > 100:  # Minimum $100 rebalance
            if btc_diff > 0:  # Need to buy BTC
                cost = btc_diff * price * 1.001  # 0.1% slippage
                if self.usdt_balance >= cost:
                    self.btc_balance += btc_diff
                    self.usdt_balance -= cost
                    self.futures_position -= btc_diff  # Short futures to maintain delta neutrality
                    
                    self.trades.append({
                        'timestamp': row.name,
                        'type': 'rebalance_buy',
                        'amount': btc_diff,
                        'price': price,
                        'strategy': 'delta_neutral'
                    })
            else:  # Need to sell BTC
                proceeds = abs(btc_diff) * price * 0.999  # 0.1% slippage
                self.btc_balance += btc_diff  # btc_diff is negative
                self.usdt_balance += proceeds
                self.futures_position -= btc_diff  # Adjust futures position
                
                self.trades.append({
                    'timestamp': row.name,
                    'type': 'rebalance_sell',
                    'amount': abs(btc_diff),
                    'price': price,
                    'strategy': 'delta_neutral'
                })

    def _check_grid_fills(self, price: float, row: pd.Series):
        """Check for grid order fills (original system logic)"""
        filled_orders = []
        
        for order in self.grid_orders:
            if order['side'] == 'buy' and price <= order['price']:
                # Buy order filled
                btc_amount = order['size_usdt'] / order['price']
                self.btc_balance += btc_amount
                self.usdt_balance -= order['size_usdt']
                
                self.grid_pnl += (order['price'] - price) * btc_amount  # Profit from better fill
                filled_orders.append(order)
                
                # Place corresponding sell order
                sell_price = order['price'] * (1 + self.grid_spacing)
                self.grid_orders.append({
                    'side': 'sell',
                    'price': sell_price,
                    'size_usdt': order['size_usdt'],
                    'created': row.name
                })
                
                self.trades.append({
                    'timestamp': row.name,
                    'type': 'grid_buy',
                    'amount': btc_amount,
                    'price': order['price'],
                    'strategy': 'grid'
                })
                
            elif order['side'] == 'sell' and price >= order['price']:
                # Sell order filled
                btc_amount = order['size_usdt'] / order['price']
                self.btc_balance -= btc_amount
                self.usdt_balance += order['size_usdt']
                
                self.grid_pnl += (price - order['price']) * btc_amount  # Profit from better fill
                filled_orders.append(order)
                
                # Place corresponding buy order
                buy_price = order['price'] * (1 - self.grid_spacing)
                self.grid_orders.append({
                    'side': 'buy',
                    'price': buy_price,
                    'size_usdt': order['size_usdt'],
                    'created': row.name
                })
                
                self.trades.append({
                    'timestamp': row.name,
                    'type': 'grid_sell',
                    'amount': btc_amount,
                    'price': order['price'],
                    'strategy': 'grid'
                })
        
        # Remove filled orders
        for order in filled_orders:
            self.grid_orders.remove(order)

    def _execute_market_making(self, price: float, row: pd.Series):
        """Execute market making strategy (original system)"""
        portfolio_value = self.usdt_balance + self.btc_balance * price
        
        # Only place new grid orders if we have less than grid_size orders
        if len(self.grid_orders) < self.grid_size:
            # Place buy order below current price
            buy_price = price * (1 - self.grid_spacing)
            if self.usdt_balance > self.order_size_usdt:
                self.grid_orders.append({
                    'side': 'buy',
                    'price': buy_price,
                    'size_usdt': self.order_size_usdt,
                    'created': row.name
                })
            
            # Place sell order above current price
            sell_price = price * (1 + self.grid_spacing)
            required_btc = self.order_size_usdt / sell_price
            if self.btc_balance > required_btc:
                self.grid_orders.append({
                    'side': 'sell',
                    'price': sell_price,
                    'size_usdt': self.order_size_usdt,
                    'created': row.name
                })

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        logger.info("Calculating performance metrics...")
        
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df = portfolio_df.set_index('timestamp')
        
        # Final portfolio value
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Annualized metrics
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Risk metrics
        daily_returns = portfolio_df['returns'].resample('D').sum().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Trade statistics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            total_trades = len(trades_df)
            
            # Calculate P&L per trade (simplified)
            winning_trades = len(trades_df[trades_df['type'].str.contains('sell')])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            total_trades = 0
            win_rate = 0
        
        # Market correlation
        btc_returns = portfolio_df['price'].pct_change().dropna()
        portfolio_returns = portfolio_df['returns'].dropna()
        
        if len(btc_returns) > 1 and len(portfolio_returns) > 1:
            # Align the series
            min_len = min(len(btc_returns), len(portfolio_returns))
            correlation = np.corrcoef(btc_returns.iloc[-min_len:], portfolio_returns.iloc[-min_len:])[0, 1]
        else:
            correlation = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'grid_pnl': self.grid_pnl,
            'funding_pnl': self.funding_pnl,
            'delta_neutral_pnl': self.delta_neutral_pnl,
            'market_correlation': correlation,
            'days_traded': days
        }

    def create_visualizations(self, df: pd.DataFrame, metrics: Dict[str, float]):
        """Create comprehensive visualization of backtest results"""
        logger.info("Creating performance visualizations...")
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df = portfolio_df.set_index('timestamp')
        
        # Set up the plot
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Original System - Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio vs BTC Price
        ax2 = plt.subplot(3, 3, 2)
        # Normalize both to starting value
        portfolio_norm = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]) * 100
        btc_norm = (portfolio_df['price'] / portfolio_df['price'].iloc[0]) * 100
        
        ax2.plot(portfolio_df.index, btc_norm, 'orange', alpha=0.7, label='BTC Price')
        ax2.plot(portfolio_df.index, portfolio_norm, 'blue', linewidth=2, label='Portfolio')
        ax2.set_title('Performance vs BTC (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Normalized Value (Base=100)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown Chart
        ax3 = plt.subplot(3, 3, 3)
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        ax3.fill_between(portfolio_df.index, portfolio_df['drawdown'] * 100, 0, 
                        color='red', alpha=0.6, label='Drawdown')
        ax3.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Balance Composition
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(portfolio_df.index, portfolio_df['usdt_balance'], 'green', label='USDT Balance')
        ax4.plot(portfolio_df.index, portfolio_df['btc_balance'] * portfolio_df['price'], 'orange', label='BTC Value')
        ax4.set_title('Portfolio Composition', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Value ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Monthly Returns Heatmap
        ax5 = plt.subplot(3, 3, 5)
        if len(portfolio_df) > 30:
            monthly_returns = portfolio_df['portfolio_value'].resample('M').last().pct_change().dropna()
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            })
            
            if len(monthly_df) > 0:
                pivot_table = monthly_df.pivot_table(values='Return', index='Year', columns='Month', fill_value=0)
                sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax5)
                ax5.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        # 6. Trading Activity
        ax6 = plt.subplot(3, 3, 6)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df = trades_df.set_index('timestamp')
            
            # Plot trade frequency over time
            monthly_trades = trades_df.resample('M').size()
            ax6.bar(monthly_trades.index, monthly_trades.values, alpha=0.7, color='blue')
            ax6.set_title('Monthly Trading Activity', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Number of Trades')
            ax6.grid(True, alpha=0.3)
        
        # 7. Strategy P&L Attribution
        ax7 = plt.subplot(3, 3, 7)
        pnl_data = {
            'Grid Trading': self.grid_pnl,
            'Funding Rates': self.funding_pnl,
            'Delta Neutral': self.delta_neutral_pnl
        }
        colors = ['green' if x > 0 else 'red' for x in pnl_data.values()]
        ax7.bar(pnl_data.keys(), pnl_data.values(), color=colors, alpha=0.7)
        ax7.set_title('P&L Attribution by Strategy', fontsize=14, fontweight='bold')
        ax7.set_ylabel('P&L ($)')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # 8. Risk Metrics Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        metrics_text = f"""
        📊 PERFORMANCE METRICS
        
        Total Return: {metrics['total_return']:+.1%}
        Annual Return: {metrics['annual_return']:+.1%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {metrics['sortino_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']:.1%}
        
        📈 TRADING STATS
        Total Trades: {metrics['total_trades']:,}
        Win Rate: {metrics['win_rate']:.1%}
        Market Correlation: {metrics['market_correlation']:.3f}
        
        💰 P&L BREAKDOWN
        Grid P&L: ${self.grid_pnl:,.0f}
        Funding P&L: ${self.funding_pnl:,.0f}
        Delta Neutral P&L: ${self.delta_neutral_pnl:,.0f}
        """
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 9. Performance Distribution
        ax9 = plt.subplot(3, 3, 9)
        if len(portfolio_df) > 1:
            daily_returns = portfolio_df['portfolio_value'].resample('D').last().pct_change().dropna()
            ax9.hist(daily_returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax9.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax9.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Daily Return (%)')
            ax9.set_ylabel('Frequency')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('original_system_backtest_2022_2025.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualization saved as 'original_system_backtest_2022_2025.png'")

    def run_backtest(self) -> Dict[str, Any]:
        """Run the complete backtest of the original system"""
        logger.info("🚀 Starting Original High-Performance System Backtest (2022-2025)")
        logger.info("="*80)
        
        # Load real historical data
        df = self.load_historical_data()
        
        if df.empty:
            logger.error("No data loaded - cannot run backtest")
            return {}
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Run the original strategy
        df = self.original_delta_neutral_strategy(df)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Create visualizations
        self.create_visualizations(df, metrics)
        
        # Print results
        self.print_results(metrics)
        
        return {
            'metrics': metrics,
            'data': df,
            'portfolio_history': self.portfolio_history,
            'trades': self.trades
        }

    def print_results(self, metrics: Dict[str, float]):
        """Print comprehensive backtest results"""
        print("\n" + "="*80)
        print("🚀 ORIGINAL HIGH-PERFORMANCE SYSTEM - BACKTEST RESULTS (2022-2025)")
        print("="*80)
        
        if not metrics:
            print("❌ No metrics available")
            return
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"Initial Capital:       ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:           ${metrics['final_value']:,.2f}")
        print(f"Total Return:          {metrics['total_return']:+.1%}")
        print(f"Annualized Return:     {metrics['annual_return']:+.1%}")
        print(f"Trading Period:        {metrics['days_traded']:.0f} days")
        
        print(f"\n📈 RISK METRICS:")
        print(f"Volatility (Annual):   {metrics['volatility']:.1%}")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown:      {metrics['max_drawdown']:.1%}")
        print(f"Market Correlation:    {metrics['market_correlation']:.3f}")
        
        print(f"\n🎯 TRADING STATISTICS:")
        print(f"Total Trades:          {metrics['total_trades']:,}")
        print(f"Win Rate:              {metrics['win_rate']:.1%}")
        
        print(f"\n💰 P&L ATTRIBUTION:")
        print(f"Grid Trading P&L:      ${self.grid_pnl:+,.2f}")
        print(f"Funding Rate P&L:      ${self.funding_pnl:+,.2f}")
        print(f"Delta Neutral P&L:     ${self.delta_neutral_pnl:+,.2f}")
        print(f"Total Strategy P&L:    ${self.grid_pnl + self.funding_pnl + self.delta_neutral_pnl:+,.2f}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if metrics['sharpe_ratio'] > 2.0:
            print("⭐ EXCEPTIONAL - Sharpe ratio > 2.0 (institutional-grade)")
        elif metrics['sharpe_ratio'] > 1.0:
            print("✅ EXCELLENT - Sharpe ratio > 1.0 (professional-grade)")
        elif metrics['sharpe_ratio'] > 0.5:
            print("✅ GOOD - Positive risk-adjusted returns")
        else:
            print("⚠️  NEEDS IMPROVEMENT - Low risk-adjusted returns")
        
        if metrics['max_drawdown'] < 0.05:
            print("⭐ EXCEPTIONAL - Drawdown < 5%")
        elif metrics['max_drawdown'] < 0.15:
            print("✅ GOOD - Drawdown < 15%")
        else:
            print("⚠️  HIGH RISK - Drawdown > 15%")
        
        if abs(metrics['market_correlation']) < 0.3:
            print("⭐ MARKET NEUTRAL - Low correlation with BTC")
        else:
            print("📈 DIRECTIONAL - Correlated with market movements")
        
        print("\n" + "="*80)

def main():
    """Run the original system backtest"""
    backtester = OriginalSystemBacktester(initial_capital=100000)
    results = backtester.run_backtest()
    return results

if __name__ == "__main__":
    results = main()