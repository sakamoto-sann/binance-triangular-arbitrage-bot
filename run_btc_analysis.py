#!/usr/bin/env python3
"""
BTC 2021 Analysis Runner
Easy-to-use script for running backtests and paper trading on BTC 2021 data
"""

import asyncio
import sys
import os
import argparse
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from btc_2021_backtest import BTC2021BacktestRunner
from btc_multi_year_backtest import BTCMultiYearBacktestRunner
from live_paper_trader import InteractivePaperTrader, LivePaperTrader
from visualization import BacktestVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_banner():
    """Print welcome banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    BTC 2021 Analysis Suite                   ║
    ║                                                              ║
    ║  📊 Backtest grid trading strategies on historical BTC data  ║
    ║  📈 Generate comprehensive performance reports               ║
    ║  🎯 Test strategies with live paper trading simulation      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

async def run_backtest():
    """Run comprehensive BTC 2021 backtest"""
    print("🚀 Starting BTC 2021 Comprehensive Backtest...")
    
    runner = BTC2021BacktestRunner()
    
    # Setup data
    print("📊 Setting up BTC 2021 data...")
    if not await runner.setup_data():
        print("❌ Failed to setup data. Please check your internet connection.")
        return False
    
    # Run backtests
    print("⚡ Running multiple strategy backtests...")
    results = await runner.run_all_backtests()
    
    if not results:
        print("❌ No successful backtests completed")
        return False
    
    # Print summary
    runner.print_summary()
    
    # Save results
    filepath = runner.save_results()
    if filepath:
        print(f"\n📁 Detailed results saved to: {filepath}")
    
    # Generate visualizations
    try:
        print("\n🎨 Generating performance charts...")
        visualizer = BacktestVisualizer()
        chart_paths = visualizer.generate_all_charts(runner.btc_data, results)
        
        if chart_paths:
            print(f"📈 Generated {len(chart_paths)} charts:")
            for chart_type, path in chart_paths.items():
                print(f"   • {chart_type}: {path}")
        
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        print("💡 Install matplotlib and seaborn: pip install matplotlib seaborn")
    
    print(f"\n✅ Backtest analysis completed! Tested {len(results)} strategies.")
    return True

async def run_multi_year_backtest():
    """Run comprehensive BTC 2021-2025 multi-year backtest"""
    print("🚀 Starting BTC 2021-2025 Multi-Year Comprehensive Backtest...")
    
    runner = BTCMultiYearBacktestRunner(start_year=2021, end_year=2025)
    
    # Setup data
    print("📊 Setting up multi-year BTC data (2021-2025)...")
    if not await runner.setup_multi_year_data():
        print("❌ Failed to setup data. Please check your internet connection.")
        return False
    
    # Run comprehensive backtests
    print("⚡ Running comprehensive multi-year backtests...")
    results = await runner.run_all_backtests()
    
    if not results:
        print("❌ No successful backtests completed")
        return False
    
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
        print("💡 Install matplotlib and seaborn: pip install matplotlib seaborn")
    
    print(f"\n✅ Multi-year backtest analysis completed! Analyzed {len(results)} strategies across 2021-2025.")
    return True

async def run_paper_trading():
    """Run interactive paper trading"""
    print("🎯 Starting Interactive Paper Trading...")
    
    interactive_trader = InteractivePaperTrader()
    await interactive_trader.run_interactive()

async def run_quick_paper_test():
    """Run a quick 15-minute paper trading test"""
    print("⚡ Running Quick Paper Trading Test (15 minutes)...")
    
    trader = LivePaperTrader(initial_balance=10000)
    
    print("📊 Loading simulation data...")
    if await trader.load_simulation_data():
        print("🚀 Starting 15-minute paper trading simulation...")
        await trader.simulate_live_trading(duration_minutes=15)
    else:
        print("❌ Failed to load simulation data")

def show_help():
    """Show help information"""
    print("""
Available Commands:

1. backtest       - Run comprehensive BTC 2021 backtests
                   Tests multiple grid trading strategies on 2021 data
                   Generates performance reports and visualizations

2. multi-year     - Run comprehensive BTC 2021-2025 multi-year backtests
                   Tests strategies across bull/bear cycles and multiple years
                   Includes market cycle analysis and yearly comparisons

3. paper          - Interactive paper trading simulation
                   Real-time simulation using historical data
                   Full control over trading parameters

4. quick-test     - Quick 15-minute paper trading test
                   Fast simulation to test the system

5. help          - Show this help message

Examples:
    python run_btc_analysis.py backtest      # Run 2021 backtest
    python run_btc_analysis.py multi-year    # Run 2021-2025 analysis
    python run_btc_analysis.py paper         # Interactive trading
    python run_btc_analysis.py quick-test    # Quick test

Requirements:
    - Python packages: pandas, numpy, requests
    - Optional for charts: matplotlib, seaborn
    - Internet connection (for fetching data)
    """)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='BTC Multi-Year Analysis Suite')
    parser.add_argument('command', nargs='?', default='help',
                       choices=['backtest', 'multi-year', 'paper', 'quick-test', 'help'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'backtest':
        await run_backtest()
    elif args.command == 'multi-year':
        await run_multi_year_backtest()
    elif args.command == 'paper':
        await run_paper_trading()
    elif args.command == 'quick-test':
        await run_quick_paper_test()
    elif args.command == 'help':
        show_help()
    else:
        print("❌ Unknown command. Use 'help' to see available commands.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)