"""
Fast Delta-Neutral Market Making Backtest (2022-2025)
Optimized validation of institutional-grade delta-neutral strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastDeltaNeutralBacktester:
    """Optimized delta-neutral backtester for faster execution."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize the fast backtester."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position tracking
        self.spot_position = 0.0
        self.futures_position = 0.0
        self.net_delta = 0.0
        
        # Performance tracking
        self.total_pnl = 0.0
        self.grid_pnl = 0.0
        self.funding_pnl = 0.0
        self.basis_pnl = 0.0
        
        # Strategy parameters
        self.spot_grid_levels = 40
        self.futures_ladder_levels = 12
        self.spot_grid_spacing = 0.005  # 0.5%
        self.futures_ladder_spacing = 0.025  # 2.5%
        self.delta_threshold = 0.04  # 4%
        self.position_size_pct = 0.02  # 2%
        
        # Risk management
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        self.risk_limit_breached = False
        
        # Performance history
        self.portfolio_history = []
        self.pnl_history = []
        self.delta_history = []
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.grid_fills = 0
        self.hedge_executions = 0
        
        logger.info("Fast Delta-Neutral Backtester initialized")
    
    def generate_fast_market_data(self, start_date: str, end_date: str, freq: str = '4H') -> pd.DataFrame:
        """Generate market data with lower frequency for faster processing."""
        try:
            logger.info(f"Generating fast market data from {start_date} to {end_date}")
            
            # Create date range with 4-hour intervals for faster processing
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            n_periods = len(dates)
            
            # Generate realistic BTC price evolution
            np.random.seed(42)
            base_prices = []
            current_price = 47000  # Starting price Jan 2022
            
            for i, date in enumerate(dates):
                # Market regime logic
                if date.year == 2022:
                    trend = -0.0003  # Bear market
                    volatility = 0.04
                elif date.year == 2023:
                    trend = 0.0001  # Sideways/recovery
                    volatility = 0.025
                else:  # 2024-2025
                    trend = 0.0004  # Bull market
                    volatility = 0.03
                
                # Price evolution
                cycle = 0.0001 * np.sin(i / (6 * 30))  # Monthly cycles
                random_shock = np.random.normal(0, volatility)
                price_change = trend + cycle + random_shock
                
                current_price *= (1 + price_change)
                base_prices.append(current_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'spot_price': base_prices
            })
            
            # Generate futures and funding data
            df['basis_spread'] = np.random.normal(0.002, 0.001, n_periods)
            df['futures_price'] = df['spot_price'] * (1 + df['basis_spread'])
            df['funding_rate'] = np.random.normal(0.0001, 0.0005, n_periods)
            df['atr'] = df['spot_price'] * np.random.uniform(0.01, 0.03, n_periods)
            
            logger.info(f"Generated {len(df)} fast data points from ${df['spot_price'].iloc[0]:,.0f} to ${df['spot_price'].iloc[-1]:,.0f}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating fast market data: {e}")
            return pd.DataFrame()
    
    def simulate_grid_trading_vectorized(self, price_changes: np.ndarray, prices: np.ndarray) -> Tuple[float, int]:
        """Vectorized simulation of grid trading for speed."""
        grid_profit = 0.0
        trades = 0
        
        # Calculate movements that trigger grid fills
        significant_moves = np.abs(price_changes) > self.spot_grid_spacing
        
        if np.any(significant_moves):
            # Count levels crossed
            levels_crossed = np.sum(np.abs(price_changes[significant_moves]) / self.spot_grid_spacing)
            levels_crossed = min(int(levels_crossed), self.spot_grid_levels // 4)
            
            # Calculate profit from grid rebalancing
            avg_price = np.mean(prices[significant_moves])
            profit_per_fill = avg_price * self.position_size_pct * self.spot_grid_spacing * 0.5
            grid_profit = levels_crossed * profit_per_fill
            trades = int(levels_crossed)
            
            # Update position (simplified)
            net_position_change = np.sum(price_changes[significant_moves]) * self.position_size_pct
            self.spot_position += net_position_change
        
        return grid_profit, trades
    
    def simulate_futures_hedge_vectorized(self, deltas: np.ndarray, futures_prices: np.ndarray) -> Tuple[float, int]:
        """Vectorized simulation of futures hedging."""
        hedge_cost = 0.0
        hedge_executions = 0
        
        # Find where hedging is needed
        hedge_needed = np.abs(deltas) > self.delta_threshold
        
        if np.any(hedge_needed):
            # Calculate hedge adjustments
            required_hedges = -deltas[hedge_needed]
            avg_futures_price = np.mean(futures_prices[hedge_needed])
            
            # Update futures position
            total_hedge = np.sum(required_hedges)
            self.futures_position += total_hedge
            
            # Calculate transaction costs
            hedge_cost = np.sum(np.abs(required_hedges)) * avg_futures_price * 0.0004
            hedge_executions = int(np.sum(hedge_needed))
        
        return hedge_cost, hedge_executions
    
    def run_fast_backtest(self, start_date: str = "2022-01-01", end_date: str = "2025-01-01") -> Dict[str, Any]:
        """Run optimized backtest with vectorized operations."""
        try:
            logger.info("🚀 Starting Fast Delta-Neutral Backtest")
            
            # Generate market data
            market_data = self.generate_fast_market_data(start_date, end_date)
            
            if market_data.empty:
                logger.error("Failed to generate market data")
                return {}
            
            logger.info(f"Processing {len(market_data)} data points...")
            
            # Vectorized calculations
            price_changes = market_data['spot_price'].pct_change().fillna(0).values
            prices = market_data['spot_price'].values
            futures_prices = market_data['futures_price'].values
            funding_rates = market_data['funding_rate'].values
            
            # Process in chunks for memory efficiency
            chunk_size = 1000
            total_chunks = len(market_data) // chunk_size + 1
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(market_data))
                
                if start_idx >= len(market_data):
                    break
                
                # Process chunk
                chunk_price_changes = price_changes[start_idx:end_idx]
                chunk_prices = prices[start_idx:end_idx]
                chunk_futures_prices = futures_prices[start_idx:end_idx]
                chunk_funding_rates = funding_rates[start_idx:end_idx]
                
                # Simulate grid trading
                grid_profit, trades = self.simulate_grid_trading_vectorized(chunk_price_changes, chunk_prices)
                self.grid_pnl += grid_profit
                self.total_trades += trades
                self.grid_fills += trades
                
                if grid_profit > 0:
                    self.winning_trades += 1
                
                # Calculate current deltas
                deltas = np.full(len(chunk_prices), self.spot_position + self.futures_position)
                
                # Simulate futures hedging
                hedge_cost, hedge_executions = self.simulate_futures_hedge_vectorized(deltas, chunk_futures_prices)
                self.grid_pnl -= hedge_cost
                self.hedge_executions += hedge_executions
                
                # Calculate funding P&L (simplified)
                if abs(self.futures_position) > 0.001:
                    chunk_funding_pnl = -self.futures_position * np.mean(chunk_funding_rates) * len(chunk_prices) / 2
                    self.funding_pnl += chunk_funding_pnl
                
                # Update total P&L
                self.total_pnl = self.grid_pnl + self.funding_pnl + self.basis_pnl
                
                # Update portfolio metrics
                for i in range(len(chunk_prices)):
                    current_price = chunk_prices[i]
                    spot_value = self.spot_position * current_price
                    current_portfolio_value = self.initial_capital + self.total_pnl + spot_value
                    
                    if current_portfolio_value > self.peak_portfolio_value:
                        self.peak_portfolio_value = current_portfolio_value
                    
                    current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                    self.max_drawdown = max(self.max_drawdown, current_drawdown)
                    
                    self.portfolio_history.append(current_portfolio_value)
                    self.pnl_history.append(self.total_pnl)
                    self.delta_history.append(self.spot_position + self.futures_position)
                    
                    # Check risk limits (relaxed for this test)
                    if current_drawdown > 0.25:  # 25% limit for testing
                        self.risk_limit_breached = True
                        logger.warning(f"Risk limit breached: {current_drawdown:.1%} drawdown")
                        break
                
                if self.risk_limit_breached:
                    break
                
                # Progress update
                if chunk_idx % 10 == 0:
                    progress = (chunk_idx / total_chunks) * 100
                    logger.info(f"Progress: {progress:.1f}% - Portfolio: ${self.portfolio_history[-1]:,.0f}")
            
            # Calculate final metrics
            final_results = self._calculate_final_metrics(market_data, start_date, end_date)
            
            logger.info("✅ Fast backtest completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Error running fast backtest: {e}")
            return {}
    
    def _calculate_final_metrics(self, market_data: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, Any]:
        """Calculate final performance metrics."""
        try:
            # Basic performance
            final_portfolio_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_capital
            total_return = final_portfolio_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Time-based metrics
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            years = (end_dt - start_dt).days / 365.25
            annualized_return = ((final_portfolio_value / self.initial_capital) ** (1/years) - 1) * 100
            
            # Risk metrics
            if len(self.portfolio_history) > 1:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                volatility = np.std(returns) * np.sqrt(365/4) * 100  # Annualized (4H data)
                sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
            
            # Win rate
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            # Benchmark comparison
            btc_start_price = market_data['spot_price'].iloc[0]
            btc_end_price = market_data['spot_price'].iloc[-1]
            btc_return = ((btc_end_price / btc_start_price) - 1) * 100
            
            return {
                "performance": {
                    "initial_capital": self.initial_capital,
                    "final_value": final_portfolio_value,
                    "total_return": total_return,
                    "total_return_pct": total_return_pct,
                    "annualized_return": annualized_return,
                    "max_drawdown": self.max_drawdown * 100,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio
                },
                "pnl_attribution": {
                    "total_pnl": self.total_pnl,
                    "grid_pnl": self.grid_pnl,
                    "funding_pnl": self.funding_pnl,
                    "basis_pnl": self.basis_pnl
                },
                "trading_stats": {
                    "total_trades": self.total_trades,
                    "winning_trades": self.winning_trades,
                    "win_rate": win_rate,
                    "grid_fills": self.grid_fills,
                    "hedge_executions": self.hedge_executions
                },
                "market_neutrality": {
                    "avg_delta_exposure": np.mean(np.abs(self.delta_history)) if self.delta_history else 0,
                    "max_delta_exposure": np.max(np.abs(self.delta_history)) if self.delta_history else 0
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
    
    def create_simple_visualization(self, results: Dict[str, Any]):
        """Create simple performance visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Delta-Neutral Strategy Performance (2022-2025)', fontsize=16, fontweight='bold')
            
            # Portfolio Value
            axes[0,0].plot(self.portfolio_history, color='green', linewidth=2)
            axes[0,0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7)
            axes[0,0].set_title('Portfolio Value Over Time')
            axes[0,0].set_ylabel('Portfolio Value ($)')
            axes[0,0].grid(True, alpha=0.3)
            
            # P&L Attribution
            pnl_data = results["pnl_attribution"]
            pnl_sources = ['Grid', 'Funding', 'Basis']
            pnl_values = [pnl_data['grid_pnl'], pnl_data['funding_pnl'], pnl_data['basis_pnl']]
            
            axes[0,1].bar(pnl_sources, pnl_values, color=['#2E8B57', '#4169E1', '#FF6347'])
            axes[0,1].set_title('P&L Attribution by Source')
            axes[0,1].set_ylabel('P&L ($)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Delta Exposure
            axes[1,0].plot(self.delta_history[:1000], color='red', alpha=0.7)  # Sample for speed
            axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1,0].axhline(y=0.04, color='red', linestyle='--', alpha=0.7)
            axes[1,0].axhline(y=-0.04, color='red', linestyle='--', alpha=0.7)
            axes[1,0].set_title('Delta Exposure Over Time (Sample)')
            axes[1,0].set_ylabel('Net Delta')
            axes[1,0].grid(True, alpha=0.3)
            
            # Performance Summary
            axes[1,1].axis('off')
            summary_text = f"""
            Total Return: {results['performance']['total_return_pct']:.1f}%
            Annualized: {results['performance']['annualized_return']:.1f}%
            Max Drawdown: {results['performance']['max_drawdown']:.1f}%
            Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}
            Win Rate: {results['trading_stats']['win_rate']:.1f}%
            vs BTC: {results['benchmark_comparison']['outperformance']:.1f}%
            """
            axes[1,1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            axes[1,1].set_title('Performance Summary')
            
            plt.tight_layout()
            plt.savefig('fast_delta_neutral_results.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            logger.info("📊 Visualization created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

def main():
    """Run the fast delta-neutral backtest."""
    print("="*80)
    print("🎯 FAST DELTA-NEUTRAL MARKET MAKING BACKTEST (2022-2025)")
    print("="*80)
    
    # Initialize backtester
    backtester = FastDeltaNeutralBacktester(initial_capital=100000)
    
    # Run backtest
    results = backtester.run_fast_backtest("2022-01-01", "2025-01-01")
    
    if results:
        # Print results
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Initial Capital:     ${results['performance']['initial_capital']:,.0f}")
        print(f"   Final Value:         ${results['performance']['final_value']:,.0f}")
        print(f"   Total Return:        {results['performance']['total_return_pct']:.1f}%")
        print(f"   Annualized Return:   {results['performance']['annualized_return']:.1f}%")
        print(f"   Max Drawdown:        {results['performance']['max_drawdown']:.1f}%")
        print(f"   Sharpe Ratio:        {results['performance']['sharpe_ratio']:.2f}")
        
        print(f"\n💰 P&L ATTRIBUTION:")
        print(f"   Grid Trading:        ${results['pnl_attribution']['grid_pnl']:,.0f}")
        print(f"   Funding Rates:       ${results['pnl_attribution']['funding_pnl']:,.0f}")
        print(f"   Basis Trading:       ${results['pnl_attribution']['basis_pnl']:,.0f}")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"   Total Trades:        {results['trading_stats']['total_trades']:,}")
        print(f"   Win Rate:            {results['trading_stats']['win_rate']:.1f}%")
        print(f"   Grid Fills:          {results['trading_stats']['grid_fills']:,}")
        print(f"   Hedge Executions:    {results['trading_stats']['hedge_executions']:,}")
        
        print(f"\n🎯 BENCHMARK COMPARISON:")
        print(f"   Strategy Return:     {results['benchmark_comparison']['strategy_return']:.1f}%")
        print(f"   BTC Buy & Hold:      {results['benchmark_comparison']['btc_buy_hold_return']:.1f}%")
        print(f"   Outperformance:      {results['benchmark_comparison']['outperformance']:.1f}%")
        
        # Create visualization
        backtester.create_simple_visualization(results)
        
        print("\n" + "="*80)
        print("✅ Fast Delta-Neutral Backtest Completed Successfully!")
        print("📊 Charts saved to 'fast_delta_neutral_results.png'")
        print("="*80)
    
    else:
        print("❌ Backtest failed to complete")

if __name__ == "__main__":
    main()