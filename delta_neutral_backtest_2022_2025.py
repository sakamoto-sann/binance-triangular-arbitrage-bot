"""
Delta-Neutral Market Making Backtest (2022-2025)
Comprehensive validation of institutional-grade delta-neutral strategy
"""

import asyncio
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

class DeltaNeutralBacktester:
    """
    Comprehensive delta-neutral market making backtester.
    Tests our institutional-grade strategy across multiple market regimes.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize the delta-neutral backtester."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position tracking
        self.spot_position = 0.0        # BTC spot position
        self.futures_position = 0.0     # BTC futures position
        self.net_delta = 0.0            # Combined delta exposure
        
        # Performance tracking
        self.total_pnl = 0.0
        self.grid_pnl = 0.0             # Profit from grid rebalancing
        self.funding_pnl = 0.0          # Profit from funding rates
        self.basis_pnl = 0.0            # Profit from basis trading
        self.spread_pnl = 0.0           # Profit from market making
        
        # Portfolio history
        self.portfolio_history = []
        self.pnl_history = []
        self.delta_history = []
        self.drawdown_history = []
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.grid_fills = 0
        self.hedge_executions = 0
        
        # Strategy parameters (matching our agreed config)
        self.spot_grid_levels = 40      # Dense grid for profit
        self.futures_ladder_levels = 12 # Sparse ladder for hedging
        self.spot_grid_spacing = 0.005  # 0.5% spacing
        self.futures_ladder_spacing = 0.025  # 2.5% spacing
        self.delta_threshold = 0.04     # 4% delta rebalance threshold
        self.position_size_pct = 0.02   # 2% position size per level
        
        # Risk management
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        self.risk_limit_breached = False
        
        logger.info("Delta-Neutral Backtester initialized for 2022-2025 validation")
    
    def generate_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate realistic BTC market data for backtesting.
        Includes spot prices, futures prices, and funding rates.
        """
        try:
            logger.info(f"Generating market data from {start_date} to {end_date}")
            
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            n_periods = len(dates)
            
            # Generate realistic BTC price evolution (2022-2025)
            np.random.seed(42)  # For reproducible results
            
            # Base price evolution with market regimes
            base_prices = []
            current_price = 47000  # Starting price Jan 2022
            
            for i, date in enumerate(dates):
                # Define market regimes
                if date.year == 2022:
                    # Bear market - gradual decline with high volatility
                    trend = -0.0001
                    volatility = 0.025
                elif date.year == 2023:
                    # Sideways/recovery - consolidation phase
                    trend = 0.00005
                    volatility = 0.015
                else:  # 2024-2025
                    # Bull market - strong uptrend
                    trend = 0.0002
                    volatility = 0.02
                
                # Add some cyclical behavior
                cycle = 0.00005 * np.sin(i / (24 * 30))  # Monthly cycles
                
                # Generate price change
                random_shock = np.random.normal(0, volatility)
                price_change = trend + cycle + random_shock
                
                current_price *= (1 + price_change)
                base_prices.append(current_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'spot_price': base_prices
            })
            
            # Generate futures prices with basis
            df['basis_spread'] = np.random.normal(0.002, 0.001, n_periods)  # 0.2% avg basis
            df['futures_price'] = df['spot_price'] * (1 + df['basis_spread'])
            
            # Generate funding rates (8-hour periods)
            funding_base = np.random.normal(0.0001, 0.0005, n_periods // 8 + 1)  # 0.01% avg
            df['funding_rate'] = np.repeat(funding_base, 8)[:n_periods]
            
            # Add volatility measure (ATR)
            df['high'] = df['spot_price'] * (1 + np.random.uniform(0.005, 0.02, n_periods))
            df['low'] = df['spot_price'] * (1 - np.random.uniform(0.005, 0.02, n_periods))
            df['atr'] = (df['high'] - df['low']) / df['spot_price']
            
            logger.info(f"Generated {len(df)} data points from ${df['spot_price'].iloc[0]:,.0f} to ${df['spot_price'].iloc[-1]:,.0f}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            return pd.DataFrame()
    
    def simulate_spot_grid_trading(self, current_price: float, previous_price: float, atr: float) -> Tuple[float, int]:
        """
        Simulate dense spot grid trading for volatility capture.
        
        Returns:
            Tuple of (grid_profit, trades_executed)
        """
        grid_profit = 0.0
        trades = 0
        
        # Calculate price movement
        price_change_pct = (current_price - previous_price) / previous_price
        
        # Estimate grid fills based on price movement and grid density
        if abs(price_change_pct) > self.spot_grid_spacing:
            # Calculate number of grid levels crossed
            levels_crossed = int(abs(price_change_pct) / self.spot_grid_spacing)
            levels_crossed = min(levels_crossed, self.spot_grid_levels // 4)  # Max 25% of grid
            
            # Each grid fill generates profit from spread capture
            profit_per_fill = current_price * self.position_size_pct * self.spot_grid_spacing * 0.5
            grid_profit = levels_crossed * profit_per_fill
            trades = levels_crossed
            
            # Accumulate position (simplified - real implementation would track each level)
            if price_change_pct > 0:
                # Price up - sells were filled, reduce long position
                self.spot_position -= levels_crossed * self.position_size_pct
            else:
                # Price down - buys were filled, increase long position
                self.spot_position += levels_crossed * self.position_size_pct
        
        return grid_profit, trades
    
    def simulate_futures_hedge(self, spot_position: float, futures_price: float) -> Tuple[float, bool]:
        """
        Simulate futures hedge ladder for delta management.
        
        Returns:
            Tuple of (hedge_cost, hedge_executed)
        """
        # Calculate current net delta
        self.net_delta = self.spot_position + self.futures_position
        
        hedge_cost = 0.0
        hedge_executed = False
        
        # Check if delta hedge is needed
        if abs(self.net_delta) > self.delta_threshold:
            # Calculate required hedge
            required_hedge = -self.net_delta
            
            # Execute hedge (futures position adjustment)
            self.futures_position += required_hedge
            
            # Calculate transaction cost
            hedge_cost = abs(required_hedge) * futures_price * 0.0004  # 0.04% transaction cost
            
            hedge_executed = True
            logger.debug(f"Delta hedge executed: {required_hedge:.4f} BTC, cost: ${hedge_cost:.2f}")
        
        return hedge_cost, hedge_executed
    
    def calculate_funding_pnl(self, funding_rate: float, time_step_hours: float = 1) -> float:
        """
        Calculate funding P&L based on position and funding rate.
        
        Args:
            funding_rate: 8-hour funding rate
            time_step_hours: Time step in hours
            
        Returns:
            Funding P&L for this time step
        """
        # Funding is paid/received on futures position
        if abs(self.futures_position) < 0.001:
            return 0.0
        
        # Convert 8-hour rate to hourly
        hourly_funding_rate = funding_rate / 8
        
        # Calculate funding P&L (receive if short futures with positive funding)
        funding_pnl = -self.futures_position * hourly_funding_rate * time_step_hours
        
        return funding_pnl
    
    def calculate_basis_pnl(self, spot_price: float, futures_price: float) -> float:
        """Calculate basis trading P&L."""
        if abs(self.spot_position) < 0.001 or abs(self.futures_position) < 0.001:
            return 0.0
        
        # Simple basis P&L calculation
        basis_spread = futures_price - spot_price
        basis_pnl = basis_spread * min(abs(self.spot_position), abs(self.futures_position)) * 0.01
        
        return basis_pnl
    
    def update_portfolio_metrics(self, current_price: float):
        """Update portfolio metrics and track performance."""
        # Calculate current portfolio value
        spot_value = self.spot_position * current_price
        unrealized_pnl = self.total_pnl
        
        current_portfolio_value = self.initial_capital + unrealized_pnl + spot_value
        
        # Update peak and drawdown
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Store history
        self.portfolio_history.append(current_portfolio_value)
        self.pnl_history.append(self.total_pnl)
        self.delta_history.append(self.net_delta)
        self.drawdown_history.append(current_drawdown)
        
        # Check risk limits
        if current_drawdown > 0.1:  # 10% drawdown limit
            self.risk_limit_breached = True
            logger.warning(f"Risk limit breached: {current_drawdown:.1%} drawdown")
    
    def run_backtest(self, start_date: str = "2022-01-01", end_date: str = "2025-01-01") -> Dict[str, Any]:
        """
        Run the comprehensive delta-neutral backtest.
        
        Returns:
            Dictionary with detailed backtest results
        """
        try:
            logger.info("🚀 Starting Delta-Neutral Market Making Backtest (2022-2025)")
            
            # Generate market data
            market_data = self.generate_market_data(start_date, end_date)
            
            if market_data.empty:
                logger.error("Failed to generate market data")
                return {}
            
            # Run simulation
            logger.info(f"Processing {len(market_data)} data points...")
            
            for i, row in market_data.iterrows():
                if i == 0:
                    continue  # Skip first row (no previous price)
                
                current_price = row['spot_price']
                previous_price = market_data.iloc[i-1]['spot_price']
                futures_price = row['futures_price']
                funding_rate = row['funding_rate']
                atr = row['atr']
                
                # 1. Simulate spot grid trading
                grid_profit, trades = self.simulate_spot_grid_trading(current_price, previous_price, atr)
                self.grid_pnl += grid_profit
                self.total_trades += trades
                self.grid_fills += trades
                
                if grid_profit > 0:
                    self.winning_trades += 1
                
                # 2. Simulate futures hedging
                hedge_cost, hedge_executed = self.simulate_futures_hedge(self.spot_position, futures_price)
                if hedge_executed:
                    self.hedge_executions += 1
                    self.grid_pnl -= hedge_cost  # Hedge costs reduce grid profits
                
                # 3. Calculate funding P&L
                funding_pnl = self.calculate_funding_pnl(funding_rate, 1.0)
                self.funding_pnl += funding_pnl
                
                # 4. Calculate basis P&L
                basis_pnl = self.calculate_basis_pnl(current_price, futures_price)
                self.basis_pnl += basis_pnl
                
                # 5. Update total P&L
                self.total_pnl = self.grid_pnl + self.funding_pnl + self.basis_pnl + self.spread_pnl
                
                # 6. Update portfolio metrics
                self.update_portfolio_metrics(current_price)
                
                # 7. Check for risk limit breach
                if self.risk_limit_breached:
                    logger.warning("Risk limits breached - stopping backtest")
                    break
                
                # Progress logging
                if i % 1000 == 0:
                    progress = (i / len(market_data)) * 100
                    logger.info(f"Progress: {progress:.1f}% - Portfolio: ${self.portfolio_history[-1]:,.0f}")
            
            # Calculate final metrics
            final_results = self._calculate_final_metrics(market_data, start_date, end_date)
            
            logger.info("✅ Backtest completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _calculate_final_metrics(self, market_data: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, Any]:
        """Calculate comprehensive final performance metrics."""
        try:
            # Basic performance metrics
            final_portfolio_value = self.portfolio_history[-1]
            total_return = final_portfolio_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Annualized return
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            years = (end_dt - start_dt).days / 365.25
            annualized_return = ((final_portfolio_value / self.initial_capital) ** (1/years) - 1) * 100
            
            # Risk metrics
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            volatility = np.std(returns) * np.sqrt(365*24) * 100  # Annualized
            
            sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0  # Risk-free rate = 2%
            
            # Downside deviation for Sortino ratio
            negative_returns = returns[returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(365*24) * 100 if len(negative_returns) > 0 else 0
            sortino_ratio = (annualized_return - 2) / downside_deviation if downside_deviation > 0 else 0
            
            # Win rate
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            # Benchmark comparison (BTC buy-and-hold)
            btc_start_price = market_data['spot_price'].iloc[0]
            btc_end_price = market_data['spot_price'].iloc[-1]
            btc_return = ((btc_end_price / btc_start_price) - 1) * 100
            
            # Monthly consistency
            monthly_returns = []
            for i in range(0, len(self.portfolio_history), 24*30):  # Monthly intervals
                if i + 24*30 < len(self.portfolio_history):
                    start_val = self.portfolio_history[i]
                    end_val = self.portfolio_history[i + 24*30]
                    monthly_ret = (end_val / start_val - 1) * 100
                    monthly_returns.append(monthly_ret)
            
            positive_months = sum(1 for ret in monthly_returns if ret > 0)
            positive_month_pct = (positive_months / max(1, len(monthly_returns))) * 100
            
            return {
                "performance": {
                    "initial_capital": self.initial_capital,
                    "final_value": final_portfolio_value,
                    "total_return": total_return,
                    "total_return_pct": total_return_pct,
                    "annualized_return": annualized_return,
                    "max_drawdown": self.max_drawdown * 100,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio
                },
                "pnl_attribution": {
                    "total_pnl": self.total_pnl,
                    "grid_pnl": self.grid_pnl,
                    "funding_pnl": self.funding_pnl,
                    "basis_pnl": self.basis_pnl,
                    "spread_pnl": self.spread_pnl
                },
                "trading_stats": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "win_rate": win_rate,
                    "grid_fills": self.grid_fills,
                    "hedge_executions": self.hedge_executions
                },
                "market_neutrality": {
                    "avg_delta_exposure": np.mean(np.abs(self.delta_history)),
                    "max_delta_exposure": np.max(np.abs(self.delta_history)),
                    "positive_months": positive_months,
                    "positive_month_pct": positive_month_pct
                },
                "benchmark_comparison": {
                    "strategy_return": total_return_pct,
                    "btc_buy_hold_return": btc_return,
                    "outperformance": total_return_pct - btc_return
                },
                "market_data": {
                    "start_price": btc_start_price,
                    "end_price": btc_end_price,
                    "data_points": len(market_data),
                    "period_years": years
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
            return {}
    
    def create_performance_visualization(self, results: Dict[str, Any]):
        """Create comprehensive performance visualization."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Delta-Neutral Market Making Strategy Performance (2022-2025)', fontsize=16, fontweight='bold')
            
            # 1. Portfolio Value Over Time
            axes[0,0].plot(self.portfolio_history, color='green', linewidth=2)
            axes[0,0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7)
            axes[0,0].set_title('Portfolio Value Over Time')
            axes[0,0].set_ylabel('Portfolio Value ($)')
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. P&L Attribution
            pnl_data = results["pnl_attribution"]
            pnl_sources = ['Grid', 'Funding', 'Basis', 'Spreads']
            pnl_values = [pnl_data['grid_pnl'], pnl_data['funding_pnl'], 
                         pnl_data['basis_pnl'], pnl_data['spread_pnl']]
            
            colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFD700']
            axes[0,1].bar(pnl_sources, pnl_values, color=colors)
            axes[0,1].set_title('P&L Attribution by Source')
            axes[0,1].set_ylabel('P&L ($)')
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Delta Exposure Over Time
            axes[0,2].plot(self.delta_history, color='red', alpha=0.7)
            axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0,2].axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='Rebalance Threshold')
            axes[0,2].axhline(y=-0.04, color='red', linestyle='--', alpha=0.7)
            axes[0,2].set_title('Delta Exposure Over Time')
            axes[0,2].set_ylabel('Net Delta')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # 4. Drawdown
            axes[1,0].fill_between(range(len(self.drawdown_history)), 
                                  [d*100 for d in self.drawdown_history], 
                                  alpha=0.6, color='red')
            axes[1,0].set_title('Drawdown Over Time')
            axes[1,0].set_ylabel('Drawdown (%)')
            axes[1,0].set_ylim(0, max(self.drawdown_history)*100*1.1)
            axes[1,0].grid(True, alpha=0.3)
            
            # 5. Performance Metrics Table
            axes[1,1].axis('off')
            metrics_data = [
                ['Total Return', f"{results['performance']['total_return_pct']:.1f}%"],
                ['Annualized Return', f"{results['performance']['annualized_return']:.1f}%"],
                ['Max Drawdown', f"{results['performance']['max_drawdown']:.1f}%"],
                ['Sharpe Ratio', f"{results['performance']['sharpe_ratio']:.2f}"],
                ['Win Rate', f"{results['trading_stats']['win_rate']:.1f}%"],
                ['Positive Months', f"{results['market_neutrality']['positive_month_pct']:.1f}%"]
            ]
            
            table = axes[1,1].table(cellText=metrics_data, 
                                   colLabels=['Metric', 'Value'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1,1].set_title('Key Performance Metrics')
            
            # 6. Strategy vs Benchmark
            strategy_return = results['benchmark_comparison']['strategy_return']
            btc_return = results['benchmark_comparison']['btc_buy_hold_return']
            
            comparison_data = ['Strategy', 'BTC Buy & Hold']
            comparison_values = [strategy_return, btc_return]
            comparison_colors = ['green' if strategy_return > btc_return else 'red', 'orange']
            
            axes[1,2].bar(comparison_data, comparison_values, color=comparison_colors)
            axes[1,2].set_title('Strategy vs Benchmark Returns')
            axes[1,2].set_ylabel('Total Return (%)')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('delta_neutral_backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("📊 Performance visualization created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

def main():
    """Run the comprehensive delta-neutral backtest."""
    print("="*80)
    print("🎯 DELTA-NEUTRAL MARKET MAKING BACKTEST (2022-2025)")
    print("="*80)
    
    # Initialize backtester
    backtester = DeltaNeutralBacktester(initial_capital=100000)
    
    # Run backtest
    results = backtester.run_backtest("2022-01-01", "2025-01-01")
    
    if results:
        # Print results
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Initial Capital:     ${results['performance']['initial_capital']:,.0f}")
        print(f"   Final Value:         ${results['performance']['final_value']:,.0f}")
        print(f"   Total Return:        {results['performance']['total_return_pct']:.1f}%")
        print(f"   Annualized Return:   {results['performance']['annualized_return']:.1f}%")
        print(f"   Max Drawdown:        {results['performance']['max_drawdown']:.1f}%")
        print(f"   Sharpe Ratio:        {results['performance']['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio:       {results['performance']['sortino_ratio']:.2f}")
        
        print(f"\n💰 P&L ATTRIBUTION:")
        print(f"   Grid Trading:        ${results['pnl_attribution']['grid_pnl']:,.0f}")
        print(f"   Funding Rates:       ${results['pnl_attribution']['funding_pnl']:,.0f}")
        print(f"   Basis Trading:       ${results['pnl_attribution']['basis_pnl']:,.0f}")
        print(f"   Market Making:       ${results['pnl_attribution']['spread_pnl']:,.0f}")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"   Total Trades:        {results['trading_stats']['total_trades']:,}")
        print(f"   Win Rate:            {results['trading_stats']['win_rate']:.1f}%")
        print(f"   Grid Fills:          {results['trading_stats']['grid_fills']:,}")
        print(f"   Hedge Executions:    {results['trading_stats']['hedge_executions']:,}")
        
        print(f"\n⚖️ MARKET NEUTRALITY:")
        print(f"   Avg Delta Exposure:  {results['market_neutrality']['avg_delta_exposure']:.3f}")
        print(f"   Max Delta Exposure:  {results['market_neutrality']['max_delta_exposure']:.3f}")
        print(f"   Positive Months:     {results['market_neutrality']['positive_month_pct']:.1f}%")
        
        print(f"\n🎯 BENCHMARK COMPARISON:")
        print(f"   Strategy Return:     {results['benchmark_comparison']['strategy_return']:.1f}%")
        print(f"   BTC Buy & Hold:      {results['benchmark_comparison']['btc_buy_hold_return']:.1f}%")
        print(f"   Outperformance:      {results['benchmark_comparison']['outperformance']:.1f}%")
        
        # Create visualization
        backtester.create_performance_visualization(results)
        
        print("\n" + "="*80)
        print("✅ Delta-Neutral Backtest Completed Successfully!")
        print("📊 Detailed charts saved to 'delta_neutral_backtest_results.png'")
        print("="*80)
    
    else:
        print("❌ Backtest failed to complete")

if __name__ == "__main__":
    main()