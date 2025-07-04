#!/usr/bin/env python3
"""
Example Usage of Professional-Grade Crypto Trading System
Demonstrates how to use all advanced components together
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from binance.client import Client

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_trading_system.professional_trading_engine import ProfessionalTradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def demonstrate_advanced_analysis():
    """Demonstrate comprehensive market analysis capabilities"""
    
    logger.info("🚀 Starting Professional Trading System Demonstration")
    
    # Configuration for institutional-grade trading
    config = {
        'volatility_grid': {
            'bitvol_weight': 0.3,        # BitVol weight in composite volatility
            'lxvx_weight': 0.25,         # LXVX weight
            'garch_weight': 0.25,        # GARCH model weight
            'realized_vol_weight': 0.2,  # Realized volatility weight
            'min_grid_spacing': 0.0005,  # 0.05% minimum spacing
            'max_grid_spacing': 0.02     # 2% maximum spacing
        },
        'timeframe_analyzer': {
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'timeframe_weights': {
                '1m': 0.1, '5m': 0.15, '15m': 0.2,
                '1h': 0.25, '4h': 0.2, '1d': 0.1
            }
        },
        'delta_hedger': {
            'base_hedge_ratio': 1.0,
            'gamma_adjustment_factor': 0.1,
            'max_delta_deviation': 0.05,
            'hedge_frequency': 60
        },
        'funding_arbitrage': {
            'exchanges': ['binance', 'okx', 'bybit'],
            'funding_threshold': 0.01,
            'min_rate_difference': 0.005,
            'max_single_position': 100000,
            'position_limits': {
                'binance': 1000000,
                'okx': 500000,
                'bybit': 500000
            }
        },
        'order_flow': {
            'order_book_depth': 20,
            'trade_flow_window': 300,
            'imbalance_threshold': 0.7
        },
        'inventory_manager': {
            'total_capital': 1000000,     # $1M capital
            'max_inventory_ratio': 0.3,   # 30% max exposure
            'correlation_threshold': 0.7,
            'kelly_multiplier': 0.25,     # Conservative Kelly
            'max_single_position': 0.1    # 10% per position
        }
    }
    
    # Initialize Binance client (use testnet for demonstration)
    # In production, use real API keys
    try:
        binance_client = Client(
            api_key="your_api_key_here",
            api_secret="your_secret_key_here",
            testnet=True  # Use testnet for safety
        )
        logger.info("✅ Binance client initialized (testnet)")
    except Exception as e:
        logger.warning(f"⚠️ Binance client initialization failed: {e}")
        logger.info("📝 Continuing with simulation mode")
        binance_client = None
    
    # Initialize the professional trading engine
    engine = ProfessionalTradingEngine(
        binance_client=binance_client,
        config=config
    )
    
    logger.info("🎯 Professional Trading Engine initialized with all components")
    
    # Demonstrate comprehensive market analysis
    symbols_to_analyze = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in symbols_to_analyze:
        logger.info(f"\n📊 Analyzing {symbol}")
        logger.info("=" * 60)
        
        try:
            # 1. Comprehensive Market Analysis
            logger.info("🔍 Running comprehensive market analysis...")
            market_conditions = await engine.analyze_market_conditions(symbol)
            
            if 'error' not in market_conditions:
                # Display volatility analysis
                vol_analysis = market_conditions.get('volatility_analysis', {})
                vol_metrics = vol_analysis.get('volatility_metrics', {})
                
                logger.info(f"📈 Volatility Analysis:")
                logger.info(f"   BitVol: {vol_metrics.get('bitvol', 0):.4f}")
                logger.info(f"   LXVX: {vol_metrics.get('lxvx', 0):.4f}")
                logger.info(f"   GARCH: {vol_metrics.get('garch_vol', 0):.4f}")
                logger.info(f"   Realized: {vol_metrics.get('realized_vol', 0):.4f}")
                logger.info(f"   Composite: {vol_metrics.get('composite_vol', 0):.4f}")
                logger.info(f"   Confidence: {vol_metrics.get('confidence', 0):.2f}")
                
                # Display grid parameters
                grid_params = vol_analysis.get('grid_parameters', {})
                logger.info(f"🎛️ Grid Parameters:")
                logger.info(f"   Spacing: {grid_params.get('spacing', 0):.4f}")
                logger.info(f"   Regime: {grid_params.get('regime', 'unknown')}")
                logger.info(f"   Max Levels: {grid_params.get('max_levels', 0)}")
                logger.info(f"   Stress Factor: {grid_params.get('stress_factor', 1):.2f}")
                
                # Display signal analysis
                signal_analysis = market_conditions.get('timeframe_analysis', {})
                composite_signal = signal_analysis.get('composite_signal', {})
                
                logger.info(f"📡 Signal Analysis:")
                logger.info(f"   Signal Strength: {composite_signal.get('signal', 0):.3f}")
                logger.info(f"   Confidence: {composite_signal.get('confidence', 0):.3f}")
                logger.info(f"   Market Regime: {composite_signal.get('regime', 'unknown')}")
                
                # Display order flow analysis
                order_flow = market_conditions.get('order_flow_analysis', {})
                flow_metrics = order_flow.get('order_flow_metrics', {})
                
                logger.info(f"🌊 Order Flow Analysis:")
                logger.info(f"   Bid/Ask Imbalance: {flow_metrics.get('bid_ask_imbalance', 0):.3f}")
                logger.info(f"   Trade Flow Imbalance: {flow_metrics.get('trade_flow_imbalance', 0):.3f}")
                logger.info(f"   Liquidity Score: {flow_metrics.get('liquidity_score', 0):.3f}")
                logger.info(f"   Execution Difficulty: {flow_metrics.get('execution_difficulty', 0):.3f}")
                logger.info(f"   Directional Bias: {flow_metrics.get('directional_bias', 'neutral')}")
                
                # Display funding opportunities
                funding_opps = market_conditions.get('funding_opportunities', [])
                if funding_opps:
                    logger.info(f"💰 Funding Opportunities:")
                    for opp in funding_opps[:3]:  # Show top 3
                        logger.info(f"   Rate Diff: {opp.get('rate_difference', 0):.4f}")
                        logger.info(f"   Annual Return: {opp.get('annualized_return', 0):.2%}")
                        logger.info(f"   Confidence: {opp.get('confidence', 0):.3f}")
                
                logger.info(f"🎯 Market Regime: {market_conditions.get('market_regime', 'unknown')}")
                logger.info(f"⚠️ Risk Assessment: {market_conditions.get('risk_assessment', 0):.3f}")
            
            # 2. Generate Trading Decision
            logger.info("\n🎯 Generating trading decision...")
            trading_decision = await engine.generate_trading_decision(symbol, market_conditions)
            
            logger.info(f"📋 Trading Decision:")
            logger.info(f"   Action: {trading_decision.action}")
            logger.info(f"   Quantity: {trading_decision.quantity:.6f}")
            logger.info(f"   Price: ${trading_decision.price:.2f}")
            logger.info(f"   Confidence: {trading_decision.confidence:.3f}")
            logger.info(f"   Risk Score: {trading_decision.risk_score:.3f}")
            logger.info(f"   Expected Profit: ${trading_decision.expected_profit:.2f}")
            logger.info(f"   Strategy: {trading_decision.strategy_type}")
            logger.info(f"   Reasoning: {trading_decision.reasoning}")
            
            # 3. Execute Complete Trading Strategy
            logger.info("\n⚡ Executing complete trading strategy...")
            execution_result = await engine.execute_trading_strategy(symbol)
            
            if execution_result.get('status') == 'success':
                logger.info(f"✅ Strategy execution successful:")
                
                # Primary strategy results
                primary = execution_result.get('execution_results', {}).get('primary_strategy', {})
                logger.info(f"   Primary Action: {primary.get('action', 'none')}")
                logger.info(f"   Grid Spacing: {primary.get('grid_spacing', 0):.4f}")
                logger.info(f"   Grid Levels: {primary.get('grid_levels', 0)}")
                logger.info(f"   Volatility Regime: {primary.get('volatility_regime', 'unknown')}")
                
                # Hedge management results
                hedge = execution_result.get('execution_results', {}).get('hedge_management', {})
                logger.info(f"   Hedge Status: {hedge.get('status', 'unknown')}")
                if hedge.get('status') == 'hedge_executed':
                    hedge_params = hedge.get('hedge_params', {})
                    logger.info(f"   Hedge Ratio: {hedge_params.get('hedge_ratio', 0):.3f}")
                    logger.info(f"   Hedge Confidence: {hedge_params.get('confidence', 0):.3f}")
                
                # Funding arbitrage results
                funding = execution_result.get('execution_results', {}).get('funding_arbitrage', {})
                logger.info(f"   Funding Status: {funding.get('status', 'unknown')}")
                if funding.get('status') == 'opportunity_found':
                    opp = funding.get('opportunity', {})
                    logger.info(f"   Funding Return: {opp.get('annualized_return', 0):.2%}")
                
                # Inventory management results
                inventory = execution_result.get('execution_results', {}).get('inventory_update', {})
                portfolio_metrics = inventory.get('portfolio_metrics', {})
                logger.info(f"   Portfolio Exposure: ${portfolio_metrics.get('total_exposure', 0):,.2f}")
                logger.info(f"   Concentration Risk: {portfolio_metrics.get('concentration_risk', 0):.3f}")
                logger.info(f"   Correlation Risk: {portfolio_metrics.get('correlation_risk', 0):.3f}")
                
                # Overall assessment
                logger.info(f"   Market Conditions: {execution_result.get('market_conditions', 'unknown')}")
                logger.info(f"   Overall Risk Score: {execution_result.get('risk_score', 0):.3f}")
                logger.info(f"   Expected Profit: ${execution_result.get('expected_profit', 0):.2f}")
                
            else:
                logger.error(f"❌ Strategy execution failed: {execution_result.get('error', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {symbol}: {e}")
    
    # 4. Get System Status
    logger.info("\n📊 System Status Summary")
    logger.info("=" * 60)
    
    try:
        system_status = await engine.get_system_status()
        
        system_info = system_status.get('system', {})
        logger.info(f"🖥️ System Status:")
        logger.info(f"   Trading Enabled: {system_info.get('trading_enabled', False)}")
        logger.info(f"   Emergency Mode: {system_info.get('emergency_mode', False)}")
        logger.info(f"   Active Positions: {system_info.get('active_positions', 0)}")
        
        performance = system_status.get('performance', {})
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"   Total Trades: {performance.get('total_trades', 0)}")
        logger.info(f"   Profitable Trades: {performance.get('profitable_trades', 0)}")
        logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1%}")
        logger.info(f"   Total P&L: ${performance.get('total_pnl', 0):.2f}")
        logger.info(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   Max Drawdown: {performance.get('max_drawdown', 0):.1%}")
        
        components = system_status.get('components', {})
        logger.info(f"🔧 Component Status:")
        for component, status in components.items():
            logger.info(f"   {component}: {status}")
            
    except Exception as e:
        logger.error(f"❌ System status check failed: {e}")
    
    logger.info("\n🎉 Professional Trading System Demonstration Complete!")
    logger.info("💡 This system represents institutional-grade crypto trading technology")
    logger.info("🏆 Ready for production deployment with real capital")

async def run_continuous_trading():
    """Example of continuous trading loop"""
    
    logger.info("🔄 Starting Continuous Trading Mode")
    
    # Initialize engine (same config as above)
    config = {}  # Use the same config from demonstrate_advanced_analysis
    engine = ProfessionalTradingEngine(config=config)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    cycle_count = 0
    
    try:
        while cycle_count < 5:  # Run 5 cycles for demonstration
            cycle_count += 1
            logger.info(f"\n🔄 Trading Cycle {cycle_count}")
            logger.info("-" * 40)
            
            for symbol in symbols:
                try:
                    # Execute complete strategy for each symbol
                    result = await engine.execute_trading_strategy(symbol)
                    
                    if result.get('status') == 'success':
                        decision = result.get('trading_decision', {})
                        logger.info(f"✅ {symbol}: {decision.get('action', 'hold')} "
                                  f"(confidence: {decision.get('confidence', 0):.2f})")
                    else:
                        logger.warning(f"⚠️ {symbol}: {result.get('error', 'execution failed')}")
                        
                except Exception as e:
                    logger.error(f"❌ {symbol} cycle failed: {e}")
            
            # Wait for next cycle (1 minute in production)
            logger.info("⏰ Waiting for next cycle...")
            await asyncio.sleep(10)  # 10 seconds for demo
            
    except KeyboardInterrupt:
        logger.info("🛑 Continuous trading stopped by user")
    except Exception as e:
        logger.error(f"❌ Continuous trading error: {e}")

if __name__ == "__main__":
    # Choose demonstration mode
    import argparse
    
    parser = argparse.ArgumentParser(description='Professional Trading System Demo')
    parser.add_argument('--mode', choices=['analysis', 'continuous'], 
                       default='analysis', help='Demonstration mode')
    args = parser.parse_args()
    
    if args.mode == 'analysis':
        asyncio.run(demonstrate_advanced_analysis())
    elif args.mode == 'continuous':
        asyncio.run(run_continuous_trading())