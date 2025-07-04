import asyncio
import logging
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from backtester import StrategyBacktester, BacktestConfig, BacktestResult
from data_fetcher import HistoricalDataFetcher
from visualization import BacktestVisualizer
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BTCMultiYearBacktestRunner:
    """
    Comprehensive backtesting runner for BTC 2021-2025 data with multi-year analysis
    """
    
    def __init__(self, start_year: int = 2021, end_year: int = 2025):
        self.start_year = start_year
        self.end_year = end_year
        self.data_fetcher = HistoricalDataFetcher()
        self.yearly_data: Dict[int, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        self.market_cycles = {}
        
    async def setup_multi_year_data(self, force_refetch: bool = False) -> bool:
        """Setup BTC data for multiple years (2021-2025)"""
        try:
            # Check if we have combined data file
            combined_file = f"data/btc_{self.start_year}_{self.end_year}_1h_combined.csv"
            
            if not force_refetch and os.path.exists(combined_file):
                logging.info(f"Loading existing combined data from {combined_file}")
                self.combined_data = self.data_fetcher.load_data(combined_file)
                
                if self.combined_data is not None:
                    logging.info(f"Loaded {len(self.combined_data)} records from combined file")
                    # Split data by year for analysis
                    self._split_data_by_year()
                    return True
            
            # Fetch data year by year
            logging.info(f"Fetching BTC data from {self.start_year} to {self.end_year}")
            self.yearly_data = self.data_fetcher.fetch_yearly_data_chunks(
                start_year=self.start_year,
                end_year=self.end_year,
                interval='1h'
            )
            
            if not self.yearly_data:
                logging.error("Failed to fetch yearly data")
                return False
            
            # Combine all years
            self.combined_data = self.data_fetcher.combine_yearly_data(self.yearly_data)
            
            if self.combined_data is not None:
                # Save combined data
                self.data_fetcher.save_data(
                    self.combined_data, 
                    f"btc_{self.start_year}_{self.end_year}_1h_combined.csv"
                )
                logging.info(f"Successfully combined {len(self.combined_data)} records")
                return True
            else:
                logging.error("Failed to combine yearly data")
                return False
                
        except Exception as e:
            logging.error(f"Error setting up multi-year data: {e}")
            return False
    
    def _split_data_by_year(self):
        """Split combined data back into yearly chunks for analysis"""
        try:
            if self.combined_data is None:
                return
            
            self.yearly_data = {}
            for year in range(self.start_year, self.end_year + 1):
                if year > datetime.now().year:
                    continue
                    
                year_mask = self.combined_data['timestamp'].dt.year == year
                year_data = self.combined_data[year_mask].copy().reset_index(drop=True)
                
                if len(year_data) > 0:
                    self.yearly_data[year] = year_data
                    logging.info(f"Split {len(year_data)} records for year {year}")
            
        except Exception as e:
            logging.error(f"Error splitting data by year: {e}")
    
    def detect_market_cycles(self) -> Dict[str, Any]:
        """Detect bull and bear market cycles in the data"""
        try:
            if self.combined_data is None:
                return {}
            
            df = self.combined_data.copy()
            
            # Calculate moving averages for trend detection
            df['ma_50'] = df['price'].rolling(window=50*24).mean()  # 50-day MA
            df['ma_200'] = df['price'].rolling(window=200*24).mean()  # 200-day MA
            
            # Detect trends
            df['trend'] = 'neutral'
            df.loc[df['ma_50'] > df['ma_200'], 'trend'] = 'bull'
            df.loc[df['ma_50'] < df['ma_200'], 'trend'] = 'bear'
            
            # Find cycle periods
            cycles = []
            current_trend = None
            start_date = None
            
            for idx, row in df.iterrows():
                if pd.isna(row['trend']) or row['trend'] == 'neutral':
                    continue
                    
                if current_trend != row['trend']:
                    if current_trend is not None and start_date is not None:
                        # End of previous cycle
                        end_idx = max(0, idx - 1)
                        end_date = df.iloc[end_idx]['timestamp']
                        start_price = df[df['timestamp'] >= start_date]['price'].iloc[0]
                        end_price = df.iloc[end_idx]['price']
                        
                        cycles.append({
                            'trend': current_trend,
                            'start_date': start_date,
                            'end_date': end_date,
                            'duration_days': (end_date - start_date).days,
                            'start_price': start_price,
                            'end_price': end_price,
                            'return_pct': ((end_price - start_price) / start_price) * 100
                        })
                    
                    # Start new cycle
                    current_trend = row['trend']
                    start_date = row['timestamp']
            
            # Handle last cycle
            if current_trend is not None and start_date is not None:
                end_date = df.iloc[-1]['timestamp']
                start_price = df[df['timestamp'] >= start_date]['price'].iloc[0]
                end_price = df.iloc[-1]['price']
                
                cycles.append({
                    'trend': current_trend,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': (end_date - start_date).days,
                    'start_price': start_price,
                    'end_price': end_price,
                    'return_pct': ((end_price - start_price) / start_price) * 100
                })
            
            self.market_cycles = {
                'cycles': cycles,
                'bull_cycles': [c for c in cycles if c['trend'] == 'bull'],
                'bear_cycles': [c for c in cycles if c['trend'] == 'bear'],
                'total_cycles': len(cycles)
            }
            
            logging.info(f"Detected {len(cycles)} market cycles")
            return self.market_cycles
            
        except Exception as e:
            logging.error(f"Error detecting market cycles: {e}")
            return {}
    
    def create_multi_year_backtest_configs(self) -> Dict[str, BacktestConfig]:
        """Create backtest configurations for multi-year analysis"""
        configs = {}
        
        # Full period strategies
        full_start = f"{self.start_year}-01-01"
        current_year = datetime.now().year
        actual_end_year = min(self.end_year, current_year)
        
        if actual_end_year == current_year:
            full_end = datetime.now().strftime('%Y-%m-%d')
        else:
            full_end = f"{actual_end_year}-12-31"
        
        # Conservative long-term strategy
        configs['conservative_full_period'] = BacktestConfig(
            start_date=full_start,
            end_date=full_end,
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=12,
            grid_interval=2000,  # $2000 intervals for long-term
            order_size=0.0005,   # Smaller size for long-term
            commission_rate=0.001
        )
        
        # Aggressive long-term strategy
        configs['aggressive_full_period'] = BacktestConfig(
            start_date=full_start,
            end_date=full_end,
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=20,
            grid_interval=1000,  # $1000 intervals
            order_size=0.001,
            commission_rate=0.001
        )
        
        # Adaptive strategy (medium grid)
        configs['adaptive_full_period'] = BacktestConfig(
            start_date=full_start,
            end_date=full_end,
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=15,
            grid_interval=1500,  # $1500 intervals
            order_size=0.0008,
            commission_rate=0.001
        )
        
        # Year-by-year strategies
        for year in range(self.start_year, min(self.end_year + 1, current_year + 1)):
            start_date = f"{year}-01-01"
            if year == current_year:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = f"{year}-12-31"
            
            # Standard strategy for each year
            configs[f'standard_{year}'] = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000,
                symbol='BTCUSDT',
                grid_size=10,
                grid_interval=1000,
                order_size=0.001,
                commission_rate=0.001
            )
            
            # Tight grid for volatile years
            configs[f'tight_grid_{year}'] = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000,
                symbol='BTCUSDT',
                grid_size=20,
                grid_interval=500,
                order_size=0.0005,
                commission_rate=0.001
            )
        
        # Bull market strategy (if we have bull cycles)
        if self.market_cycles and self.market_cycles.get('bull_cycles'):
            for i, cycle in enumerate(self.market_cycles['bull_cycles']):
                start_date = cycle['start_date'].strftime('%Y-%m-%d')
                end_date = cycle['end_date'].strftime('%Y-%m-%d')
                
                configs[f'bull_cycle_{i+1}'] = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=10000,
                    symbol='BTCUSDT',
                    grid_size=8,
                    grid_interval=2000,  # Wider intervals for bull markets
                    order_size=0.001,
                    commission_rate=0.001
                )
        
        # Bear market strategy
        if self.market_cycles and self.market_cycles.get('bear_cycles'):
            for i, cycle in enumerate(self.market_cycles['bear_cycles']):
                start_date = cycle['start_date'].strftime('%Y-%m-%d')
                end_date = cycle['end_date'].strftime('%Y-%m-%d')
                
                configs[f'bear_cycle_{i+1}'] = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=10000,
                    symbol='BTCUSDT',
                    grid_size=15,
                    grid_interval=800,   # Tighter intervals for bear markets
                    order_size=0.0008,
                    commission_rate=0.001
                )
        
        return configs
    
    async def run_all_backtests(self) -> Dict[str, Any]:
        """Run comprehensive multi-year backtests"""
        if self.combined_data is None:
            logging.error("No data available for backtesting")
            return {}
        
        # Detect market cycles first
        self.detect_market_cycles()
        
        # Create configurations
        configs = self.create_multi_year_backtest_configs()
        results = {}
        
        logging.info(f"Running {len(configs)} different backtest configurations across {self.start_year}-{self.end_year}")
        
        for config_name, config in configs.items():
            logging.info(f"\n=== Running backtest: {config_name} ===")
            
            try:
                backtester = StrategyBacktester(config)
                backtester.load_historical_data(data=self.combined_data.copy())
                
                result = await backtester.run_backtest()
                
                if result:
                    results[config_name] = {
                        'config': config,
                        'result': result,
                        'report': backtester.generate_report()
                    }
                    
                    # Log individual results
                    logging.info(f"✅ {config_name} completed:")
                    logging.info(f"   Period: {config.start_date} to {config.end_date}")
                    logging.info(f"   Final Balance: {result.final_balance:.2f} USDT")
                    logging.info(f"   Total Return: {result.total_return_pct:.2f}%")
                    logging.info(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
                    logging.info(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
                    logging.info(f"   Total Trades: {result.total_trades}")
                else:
                    logging.error(f"❌ {config_name} failed")
                    
            except Exception as e:
                logging.error(f"Error running backtest {config_name}: {e}")
        
        self.results = results
        return results
    
    def analyze_yearly_performance(self) -> Dict[str, Any]:
        """Analyze performance year by year"""
        try:
            yearly_analysis = {}
            
            # Market performance by year
            for year, data in self.yearly_data.items():
                if len(data) == 0:
                    continue
                
                start_price = data.iloc[0]['price']
                end_price = data.iloc[-1]['price']
                min_price = data['price'].min()
                max_price = data['price'].max()
                
                # Calculate volatility
                data_copy = data.copy()
                data_copy['daily_return'] = data_copy['price'].pct_change()
                volatility = data_copy['daily_return'].std() * np.sqrt(365) * 100
                
                yearly_analysis[year] = {
                    'market_performance': {
                        'start_price': start_price,
                        'end_price': end_price,
                        'yearly_return_pct': ((end_price - start_price) / start_price) * 100,
                        'min_price': min_price,
                        'max_price': max_price,
                        'volatility_pct': volatility,
                        'max_gain_pct': ((max_price - start_price) / start_price) * 100,
                        'max_drawdown_from_peak': ((min_price - max_price) / max_price) * 100
                    }
                }
                
                # Strategy performance for this year
                year_strategies = {}
                for strategy_name, strategy_data in self.results.items():
                    if str(year) in strategy_name and 'full_period' not in strategy_name:
                        year_strategies[strategy_name] = strategy_data['result']
                
                yearly_analysis[year]['strategy_performance'] = year_strategies
            
            return yearly_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing yearly performance: {e}")
            return {}
    
    def compare_strategies_across_periods(self) -> Dict[str, Any]:
        """Compare strategy performance across different time periods"""
        try:
            comparison = {
                'full_period_strategies': {},
                'yearly_strategies': {},
                'cycle_strategies': {},
                'best_performers': {},
                'consistency_analysis': {}
            }
            
            # Group results by strategy type
            for strategy_name, data in self.results.items():
                result = data['result']
                
                if 'full_period' in strategy_name:
                    comparison['full_period_strategies'][strategy_name] = {
                        'total_return_pct': result.total_return_pct,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate
                    }
                elif any(str(year) in strategy_name for year in range(self.start_year, self.end_year + 1)):
                    comparison['yearly_strategies'][strategy_name] = {
                        'total_return_pct': result.total_return_pct,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate
                    }
                elif 'cycle' in strategy_name:
                    comparison['cycle_strategies'][strategy_name] = {
                        'total_return_pct': result.total_return_pct,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate
                    }
            
            # Find best performers across categories
            all_strategies = {}
            all_strategies.update(comparison['full_period_strategies'])
            all_strategies.update(comparison['yearly_strategies'])
            all_strategies.update(comparison['cycle_strategies'])
            
            if all_strategies:
                comparison['best_performers'] = {
                    'highest_return': max(all_strategies.items(), key=lambda x: x[1]['total_return_pct']),
                    'lowest_drawdown': min(all_strategies.items(), key=lambda x: x[1]['max_drawdown_pct']),
                    'best_sharpe': max(all_strategies.items(), key=lambda x: x[1]['sharpe_ratio']),
                    'highest_win_rate': max(all_strategies.items(), key=lambda x: x[1]['win_rate'])
                }
                
                # Consistency analysis
                yearly_returns = []
                for strategy_name, data in comparison['yearly_strategies'].items():
                    yearly_returns.append(data['total_return_pct'])
                
                if yearly_returns:
                    comparison['consistency_analysis'] = {
                        'avg_yearly_return': np.mean(yearly_returns),
                        'yearly_return_std': np.std(yearly_returns),
                        'consistency_score': np.mean(yearly_returns) / np.std(yearly_returns) if np.std(yearly_returns) > 0 else 0,
                        'positive_years': len([r for r in yearly_returns if r > 0]),
                        'negative_years': len([r for r in yearly_returns if r < 0])
                    }
            
            return comparison
            
        except Exception as e:
            logging.error(f"Error comparing strategies: {e}")
            return {}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive multi-year analysis report"""
        if not self.combined_data or not self.results:
            return {}
        
        # Data summary
        data_summary = self.data_fetcher.get_data_summary(self.combined_data)
        
        # Yearly analysis
        yearly_analysis = self.analyze_yearly_performance()
        
        # Strategy comparison
        strategy_comparison = self.compare_strategies_across_periods()
        
        # Market cycles analysis
        cycles_analysis = self.market_cycles
        
        # Overall market statistics
        start_price = self.combined_data.iloc[0]['price']
        end_price = self.combined_data.iloc[-1]['price']
        total_return = ((end_price - start_price) / start_price) * 100
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'analysis_period': f"{self.start_year}-{self.end_year}",
            'data_summary': data_summary,
            'market_overview': {
                'start_price': start_price,
                'end_price': end_price,
                'total_market_return_pct': total_return,
                'total_data_points': len(self.combined_data),
                'years_analyzed': list(self.yearly_data.keys())
            },
            'market_cycles': cycles_analysis,
            'yearly_analysis': yearly_analysis,
            'strategy_comparison': strategy_comparison,
            'individual_results': {
                name: data['report'] for name, data in self.results.items()
            },
            'conclusions': self._generate_multi_year_conclusions()
        }
        
        return report
    
    def _generate_multi_year_conclusions(self) -> Dict[str, Any]:
        """Generate conclusions for multi-year analysis"""
        try:
            conclusions = {
                'key_findings': [],
                'recommendations': [],
                'market_insights': [],
                'strategy_insights': []
            }
            
            if not self.results:
                return conclusions
            
            # Market insights
            if self.market_cycles:
                bull_cycles = self.market_cycles.get('bull_cycles', [])
                bear_cycles = self.market_cycles.get('bear_cycles', [])
                
                if bull_cycles:
                    avg_bull_return = np.mean([c['return_pct'] for c in bull_cycles])
                    avg_bull_duration = np.mean([c['duration_days'] for c in bull_cycles])
                    conclusions['market_insights'].append(
                        f"Bull markets averaged {avg_bull_return:.1f}% returns over {avg_bull_duration:.0f} days"
                    )
                
                if bear_cycles:
                    avg_bear_return = np.mean([c['return_pct'] for c in bear_cycles])
                    avg_bear_duration = np.mean([c['duration_days'] for c in bear_cycles])
                    conclusions['market_insights'].append(
                        f"Bear markets averaged {avg_bear_return:.1f}% returns over {avg_bear_duration:.0f} days"
                    )
            
            # Strategy insights
            comparison = self.compare_strategies_across_periods()
            if comparison.get('best_performers'):
                best_return = comparison['best_performers']['highest_return']
                best_sharpe = comparison['best_performers']['best_sharpe']
                
                conclusions['key_findings'].append(
                    f"Best performing strategy: {best_return[0]} with {best_return[1]['total_return_pct']:.2f}% return"
                )
                conclusions['key_findings'].append(
                    f"Best risk-adjusted strategy: {best_sharpe[0]} with Sharpe ratio {best_sharpe[1]['sharpe_ratio']:.2f}"
                )
            
            # Generate recommendations
            if comparison.get('consistency_analysis'):
                consistency = comparison['consistency_analysis']
                if consistency['positive_years'] > consistency['negative_years']:
                    conclusions['recommendations'].append(
                        "Grid trading showed positive results in majority of years - strategy appears viable long-term"
                    )
                else:
                    conclusions['recommendations'].append(
                        "Mixed results across years - consider market condition adaptations"
                    )
            
            return conclusions
            
        except Exception as e:
            logging.error(f"Error generating conclusions: {e}")
            return {}
    
    def save_results(self, filename: str = None) -> str:
        """Save comprehensive multi-year results"""
        try:
            if filename is None:
                filename = f"btc_{self.start_year}_{self.end_year}_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = self.generate_comprehensive_report()
            
            os.makedirs('results', exist_ok=True)
            filepath = os.path.join('results', filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Multi-year results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            return None
    
    def print_multi_year_summary(self):
        """Print comprehensive multi-year summary"""
        if not self.results:
            print("No backtest results available")
            return
        
        print("\n" + "="*80)
        print(f"BTC {self.start_year}-{self.end_year} MULTI-YEAR BACKTEST RESULTS")
        print("="*80)
        
        # Data overview
        if self.combined_data is not None:
            start_date = self.combined_data['timestamp'].min()
            end_date = self.combined_data['timestamp'].max()
            start_price = self.combined_data.iloc[0]['price']
            end_price = self.combined_data.iloc[-1]['price']
            total_return = ((end_price - start_price) / start_price) * 100
            
            print(f"📊 Data Period: {start_date} to {end_date}")
            print(f"📈 Market Performance: ${start_price:,.0f} → ${end_price:,.0f} ({total_return:+.1f}%)")
            print(f"📋 Total Data Points: {len(self.combined_data):,}")
            print(f"🗓️  Years Analyzed: {len(self.yearly_data)}")
        
        # Market cycles
        if self.market_cycles:
            bull_cycles = len(self.market_cycles.get('bull_cycles', []))
            bear_cycles = len(self.market_cycles.get('bear_cycles', []))
            print(f"🐂 Bull Cycles: {bull_cycles}")
            print(f"🐻 Bear Cycles: {bear_cycles}")
        
        print(f"⚡ Strategies Tested: {len(self.results)}")
        print()
        
        # Best performers
        comparison = self.compare_strategies_across_periods()
        if comparison.get('best_performers'):
            best_performers = comparison['best_performers']
            print("🏆 BEST PERFORMERS:")
            print(f"   📊 Highest Return: {best_performers['highest_return'][0]} ({best_performers['highest_return'][1]['total_return_pct']:.2f}%)")
            print(f"   🛡️  Lowest Drawdown: {best_performers['lowest_drawdown'][0]} ({best_performers['lowest_drawdown'][1]['max_drawdown_pct']:.2f}%)")
            print(f"   📈 Best Sharpe: {best_performers['best_sharpe'][0]} ({best_performers['best_sharpe'][1]['sharpe_ratio']:.2f})")
            print(f"   🎯 Highest Win Rate: {best_performers['highest_win_rate'][0]} ({best_performers['highest_win_rate'][1]['win_rate']:.1f}%)")
        
        # Strategy type performance
        print("\n📋 STRATEGY PERFORMANCE BY TYPE:")
        
        full_period = comparison.get('full_period_strategies', {})
        if full_period:
            print("   🌍 Full Period Strategies:")
            for name, data in full_period.items():
                print(f"      {name}: {data['total_return_pct']:+.2f}% return, {data['max_drawdown_pct']:.2f}% drawdown")
        
        yearly = comparison.get('yearly_strategies', {})
        if yearly:
            print("   📅 Yearly Strategies (sample):")
            sample_strategies = list(yearly.items())[:5]  # Show first 5
            for name, data in sample_strategies:
                print(f"      {name}: {data['total_return_pct']:+.2f}% return")
        
        cycle_strategies = comparison.get('cycle_strategies', {})
        if cycle_strategies:
            print("   🔄 Market Cycle Strategies:")
            for name, data in cycle_strategies.items():
                print(f"      {name}: {data['total_return_pct']:+.2f}% return")
        
        print("="*80)

async def main():
    """Main function for multi-year backtesting"""
    runner = BTCMultiYearBacktestRunner(start_year=2021, end_year=2025)
    
    # Setup data
    logging.info("Setting up multi-year BTC data (2021-2025)...")
    if not await runner.setup_multi_year_data():
        logging.error("Failed to setup data. Exiting.")
        return
    
    # Run comprehensive backtests
    logging.info("Running comprehensive multi-year backtests...")
    results = await runner.run_all_backtests()
    
    if not results:
        logging.error("No successful backtests completed")
        return
    
    # Print summary
    runner.print_multi_year_summary()
    
    # Save results
    filepath = runner.save_results()
    if filepath:
        print(f"\n📁 Detailed results saved to: {filepath}")
    
    # Generate visualizations
    try:
        print("\n🎨 Generating performance charts...")
        visualizer = BacktestVisualizer()
        chart_paths = visualizer.generate_all_charts(runner.combined_data, results)
        
        if chart_paths:
            print(f"📈 Generated {len(chart_paths)} charts:")
            for chart_type, path in chart_paths.items():
                print(f"   • {chart_type}: {path}")
        
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
    
    print(f"\n✅ Multi-year backtest analysis completed! Analyzed {len(results)} strategies across {runner.start_year}-{runner.end_year}.")

if __name__ == "__main__":
    asyncio.run(main())