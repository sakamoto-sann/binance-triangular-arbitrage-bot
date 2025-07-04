import asyncio
import logging
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional
from backtester import StrategyBacktester, BacktestConfig, BacktestResult
from data_fetcher import HistoricalDataFetcher
import matplotlib.pyplot as plt
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BTC2021BacktestRunner:
    """
    Specialized backtesting runner for BTC 2021 data with comprehensive analysis
    """
    
    def __init__(self):
        self.data_fetcher = HistoricalDataFetcher()
        self.btc_data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        
    async def setup_data(self, force_refetch: bool = False) -> bool:
        """Setup BTC 2021 data for backtesting"""
        try:
            data_file = "data/btc_2021_1h_binance.csv"
            
            # Check if data already exists
            if not force_refetch and os.path.exists(data_file):
                logging.info(f"Loading existing data from {data_file}")
                self.btc_data = self.data_fetcher.load_data(data_file)
                
                if self.btc_data is not None:
                    logging.info(f"Loaded {len(self.btc_data)} records from existing file")
                    return True
            
            # Fetch fresh data
            logging.info("Fetching fresh BTC 2021 data...")
            self.btc_data = self.data_fetcher.fetch_btc_2021_data(interval='1h')
            
            if self.btc_data is not None:
                logging.info(f"Successfully fetched {len(self.btc_data)} records")
                return True
            else:
                logging.error("Failed to fetch BTC 2021 data")
                return False
                
        except Exception as e:
            logging.error(f"Error setting up data: {e}")
            return False
    
    def create_backtest_configs(self) -> Dict[str, BacktestConfig]:
        """Create multiple backtest configurations for different strategies"""
        configs = {}
        
        # Conservative strategy
        configs['conservative'] = BacktestConfig(
            start_date='2021-01-01',
            end_date='2021-12-31',
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=8,
            grid_interval=1000,  # $1000 intervals
            order_size=0.001,    # Small position size
            commission_rate=0.001
        )
        
        # Aggressive strategy
        configs['aggressive'] = BacktestConfig(
            start_date='2021-01-01',
            end_date='2021-12-31',
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=20,
            grid_interval=500,   # $500 intervals
            order_size=0.002,    # Larger position size
            commission_rate=0.001
        )
        
        # Tight grid strategy
        configs['tight_grid'] = BacktestConfig(
            start_date='2021-01-01',
            end_date='2021-12-31',
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=15,
            grid_interval=250,   # $250 intervals
            order_size=0.0015,
            commission_rate=0.001
        )
        
        # Quarterly tests
        quarters = [
            ('Q1_2021', '2021-01-01', '2021-03-31'),
            ('Q2_2021', '2021-04-01', '2021-06-30'),
            ('Q3_2021', '2021-07-01', '2021-09-30'),
            ('Q4_2021', '2021-10-01', '2021-12-31')
        ]
        
        for quarter_name, start_date, end_date in quarters:
            configs[f'conservative_{quarter_name}'] = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000,
                symbol='BTCUSDT',
                grid_size=8,
                grid_interval=1000,
                order_size=0.001,
                commission_rate=0.001
            )
        
        return configs
    
    async def run_all_backtests(self) -> Dict[str, Any]:
        """Run backtests for all configurations"""
        if self.btc_data is None:
            logging.error("No data available for backtesting")
            return {}
        
        configs = self.create_backtest_configs()
        results = {}
        
        logging.info(f"Running {len(configs)} different backtest configurations...")
        
        for config_name, config in configs.items():
            logging.info(f"\n=== Running backtest: {config_name} ===")
            
            try:
                backtester = StrategyBacktester(config)
                backtester.load_historical_data(data=self.btc_data.copy())
                
                result = await backtester.run_backtest()
                
                if result:
                    results[config_name] = {
                        'config': config,
                        'result': result,
                        'report': backtester.generate_report()
                    }
                    
                    # Log individual results
                    logging.info(f"✅ {config_name} completed:")
                    logging.info(f"   Final Balance: {result.final_balance:.2f} USDT")
                    logging.info(f"   Total Return: {result.total_return_pct:.2f}%")
                    logging.info(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
                    logging.info(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
                    logging.info(f"   Win Rate: {result.win_rate:.1f}%")
                else:
                    logging.error(f"❌ {config_name} failed")
                    
            except Exception as e:
                logging.error(f"Error running backtest {config_name}: {e}")
        
        self.results = results
        return results
    
    def compare_strategies(self) -> Dict[str, Any]:
        """Compare results across all strategies"""
        if not self.results:
            return {}
        
        comparison = {
            'summary': {},
            'rankings': {},
            'best_performers': {}
        }
        
        # Extract key metrics for comparison
        metrics = ['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate']
        strategy_data = {}
        
        for strategy_name, data in self.results.items():
            result = data['result']
            strategy_data[strategy_name] = {
                'total_return_pct': result.total_return_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'final_balance': result.final_balance,
                'total_trades': result.total_trades
            }
        
        comparison['summary'] = strategy_data
        
        # Rankings
        for metric in metrics:
            ranking = sorted(
                strategy_data.items(),
                key=lambda x: x[1][metric],
                reverse=True if metric != 'max_drawdown_pct' else False
            )
            comparison['rankings'][metric] = ranking
        
        # Best performers
        comparison['best_performers'] = {
            'highest_return': max(strategy_data.items(), key=lambda x: x[1]['total_return_pct']),
            'lowest_drawdown': min(strategy_data.items(), key=lambda x: x[1]['max_drawdown_pct']),
            'best_sharpe': max(strategy_data.items(), key=lambda x: x[1]['sharpe_ratio']),
            'highest_win_rate': max(strategy_data.items(), key=lambda x: x[1]['win_rate'])
        }
        
        return comparison
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        if not self.btc_data or not self.results:
            return {}
        
        # Data summary
        data_summary = self.data_fetcher.get_data_summary(self.btc_data)
        
        # Strategy comparison
        strategy_comparison = self.compare_strategies()
        
        # Market analysis for 2021
        market_analysis = self._analyze_2021_market()
        
        # Overall report
        report = {
            'report_generated': datetime.now().isoformat(),
            'data_summary': data_summary,
            'market_analysis': market_analysis,
            'strategy_comparison': strategy_comparison,
            'individual_results': {
                name: data['report'] for name, data in self.results.items()
            },
            'conclusions': self._generate_conclusions()
        }
        
        return report
    
    def _analyze_2021_market(self) -> Dict[str, Any]:
        """Analyze BTC market conditions in 2021"""
        try:
            df = self.btc_data.copy()
            df['date'] = df['timestamp'].dt.date
            
            # Price analysis
            start_price = df.iloc[0]['price']
            end_price = df.iloc[-1]['price']
            min_price = df['price'].min()
            max_price = df['price'].max()
            
            # Monthly analysis
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_returns = df.groupby('month')['price'].agg(['first', 'last']).reset_index()
            monthly_returns['return_pct'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100
            
            # Volatility analysis
            df['daily_return'] = df['price'].pct_change()
            volatility = df['daily_return'].std() * np.sqrt(365) * 100  # Annualized volatility
            
            analysis = {
                'price_summary': {
                    'start_price': start_price,
                    'end_price': end_price,
                    'yearly_return_pct': ((end_price - start_price) / start_price) * 100,
                    'min_price': min_price,
                    'max_price': max_price,
                    'price_range_pct': ((max_price - min_price) / min_price) * 100
                },
                'volatility': {
                    'annualized_volatility_pct': volatility,
                    'daily_volatility_pct': df['daily_return'].std() * 100
                },
                'monthly_performance': monthly_returns.to_dict('records'),
                'best_month': monthly_returns.loc[monthly_returns['return_pct'].idxmax()].to_dict(),
                'worst_month': monthly_returns.loc[monthly_returns['return_pct'].idxmin()].to_dict()
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing 2021 market: {e}")
            return {}
    
    def _generate_conclusions(self) -> Dict[str, Any]:
        """Generate conclusions based on backtest results"""
        try:
            if not self.results:
                return {}
            
            comparison = self.compare_strategies()
            
            conclusions = {
                'key_findings': [],
                'recommendations': [],
                'risk_assessment': {},
                'market_insights': []
            }
            
            # Analyze results
            best_return = comparison['best_performers']['highest_return']
            lowest_dd = comparison['best_performers']['lowest_drawdown']
            best_sharpe = comparison['best_performers']['best_sharpe']
            
            conclusions['key_findings'] = [
                f"Best overall return: {best_return[0]} with {best_return[1]['total_return_pct']:.2f}%",
                f"Lowest drawdown: {lowest_dd[0]} with {lowest_dd[1]['max_drawdown_pct']:.2f}%",
                f"Best risk-adjusted return: {best_sharpe[0]} with Sharpe ratio {best_sharpe[1]['sharpe_ratio']:.2f}"
            ]
            
            # Generate recommendations based on results
            avg_return = np.mean([data['total_return_pct'] for data in comparison['summary'].values()])
            avg_drawdown = np.mean([data['max_drawdown_pct'] for data in comparison['summary'].values()])
            
            if avg_return > 0:
                conclusions['recommendations'].append("Grid trading showed positive returns on average in 2021")
            else:
                conclusions['recommendations'].append("Grid trading struggled in 2021 - consider strategy modifications")
            
            if avg_drawdown < 10:
                conclusions['recommendations'].append("Drawdowns were manageable across strategies")
            else:
                conclusions['recommendations'].append("High drawdowns observed - consider reducing position sizes")
            
            return conclusions
            
        except Exception as e:
            logging.error(f"Error generating conclusions: {e}")
            return {}
    
    def save_results(self, filename: str = None) -> str:
        """Save comprehensive results to file"""
        try:
            if filename is None:
                filename = f"btc_2021_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = self.generate_comprehensive_report()
            
            # Create results directory
            os.makedirs('results', exist_ok=True)
            filepath = os.path.join('results', filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            return None
    
    def print_summary(self):
        """Print a summary of all backtest results"""
        if not self.results:
            print("No backtest results available")
            return
        
        comparison = self.compare_strategies()
        
        print("\n" + "="*80)
        print("BTC 2021 BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        # Data info
        if self.btc_data is not None:
            print(f"Data Points: {len(self.btc_data)}")
            print(f"Date Range: {self.btc_data['timestamp'].min()} to {self.btc_data['timestamp'].max()}")
        
        print(f"Strategies Tested: {len(self.results)}")
        print()
        
        # Best performers
        best_performers = comparison['best_performers']
        print("BEST PERFORMERS:")
        print(f"🏆 Highest Return: {best_performers['highest_return'][0]} ({best_performers['highest_return'][1]['total_return_pct']:.2f}%)")
        print(f"🛡️  Lowest Drawdown: {best_performers['lowest_drawdown'][0]} ({best_performers['lowest_drawdown'][1]['max_drawdown_pct']:.2f}%)")
        print(f"📊 Best Sharpe Ratio: {best_performers['best_sharpe'][0]} ({best_performers['best_sharpe'][1]['sharpe_ratio']:.2f})")
        print(f"🎯 Highest Win Rate: {best_performers['highest_win_rate'][0]} ({best_performers['highest_win_rate'][1]['win_rate']:.1f}%)")
        print()
        
        # All strategies summary
        print("ALL STRATEGIES PERFORMANCE:")
        print(f"{'Strategy':<20} {'Return %':<10} {'Drawdown %':<12} {'Sharpe':<8} {'Win Rate %':<10}")
        print("-" * 70)
        
        for name, data in comparison['summary'].items():
            print(f"{name:<20} {data['total_return_pct']:>8.2f}% {data['max_drawdown_pct']:>10.2f}% {data['sharpe_ratio']:>6.2f} {data['win_rate']:>8.1f}%")
        
        print("="*80)

async def main():
    """Main function to run BTC 2021 backtesting"""
    runner = BTC2021BacktestRunner()
    
    # Setup data
    logging.info("Setting up BTC 2021 data...")
    if not await runner.setup_data():
        logging.error("Failed to setup data. Exiting.")
        return
    
    # Run all backtests
    logging.info("Running comprehensive backtests...")
    results = await runner.run_all_backtests()
    
    if not results:
        logging.error("No successful backtests completed")
        return
    
    # Print summary
    runner.print_summary()
    
    # Save results
    filepath = runner.save_results()
    if filepath:
        print(f"\n📁 Detailed results saved to: {filepath}")
    
    print(f"\n✅ Backtesting completed! Tested {len(results)} strategies on BTC 2021 data.")

if __name__ == "__main__":
    asyncio.run(main())