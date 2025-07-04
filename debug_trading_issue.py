#!/usr/bin/env python3
"""
Debug Trading Issue - Identify why enhanced strategy isn't trading
"""

import sys
import os
sys.path.append('.')
sys.path.append('src')
sys.path.append('src/advanced')
sys.path.append('advanced')

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

# Import components
sys.path.insert(0, 'advanced')
sys.path.insert(0, 'src/advanced')

from advanced.atr_grid_optimizer import ATRConfig
from src.advanced.atr_supertrend_optimizer import ATRSupertrendOptimizer, SupertrendConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_trading_conditions():
    """Debug why the enhanced strategy isn't trading."""
    try:
        print("🔍 DEBUGGING TRADING CONDITIONS")
        print("=" * 50)
        
        # Initialize configurations
        atr_config = ATRConfig()
        supertrend_config = SupertrendConfig()
        optimizer = ATRSupertrendOptimizer(atr_config, supertrend_config)
        
        # Load sample data
        data_files = ['btc_2021_2025_1h_combined.csv', 'btc_2024_2024_1h_binance.csv']
        price_data = None
        
        for data_file in data_files:
            if os.path.exists(data_file):
                price_data = pd.read_csv(data_file)
                if 'timestamp' in price_data.columns:
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                    price_data.set_index('timestamp', inplace=True)
                price_data = price_data.tail(200)  # Small sample for debugging
                print(f"✅ Loaded {len(price_data)} rows from {data_file}")
                break
        
        if price_data is None:
            print("❌ No data file found")
            return
        
        # Test several periods
        start_idx = 50
        trading_allowed_count = 0
        signal_agreement_count = 0
        analysis_errors = 0
        
        print(f"\n🔄 Testing {len(price_data) - start_idx} periods...")
        
        for idx in range(start_idx, min(start_idx + 20, len(price_data))):  # Test first 20 periods
            current_price = price_data['close'].iloc[idx]
            hist_data = price_data.iloc[max(0, idx-100):idx+1]
            
            try:
                # Get integrated analysis
                analysis = optimizer.analyze_market_conditions(hist_data)
                enhanced_params = optimizer.get_enhanced_grid_parameters(current_price, analysis)
                
                print(f"\n📊 Period {idx}:")
                print(f"   Price: ${current_price:.2f}")
                print(f"   Trading Allowed: {analysis.trading_allowed}")
                print(f"   Signal Agreement: {analysis.signal_agreement}")
                print(f"   Enhanced Confidence: {analysis.enhanced_confidence:.3f}")
                print(f"   Grid Spacing: {enhanced_params['spacing_pct']:.4f}")
                print(f"   Market Regime: {analysis.market_regime}")
                
                if analysis.trading_allowed:
                    trading_allowed_count += 1
                
                if analysis.signal_agreement:
                    signal_agreement_count += 1
                    
            except Exception as e:
                analysis_errors += 1
                print(f"❌ Analysis error at {idx}: {e}")
        
        print(f"\n📈 SUMMARY:")
        print(f"Periods tested: 20")
        print(f"Trading allowed: {trading_allowed_count}")
        print(f"Signal agreements: {signal_agreement_count}")
        print(f"Analysis errors: {analysis_errors}")
        
        if trading_allowed_count == 0:
            print(f"\n🚨 ISSUE IDENTIFIED: No trading allowed in any period!")
            print(f"This explains why no trades are executed.")
        
        if signal_agreement_count > 0:
            print(f"\n✅ Signal detection working: {signal_agreement_count} agreements found")
        
        return {
            'trading_allowed_count': trading_allowed_count,
            'signal_agreement_count': signal_agreement_count,
            'analysis_errors': analysis_errors
        }
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        return None

if __name__ == "__main__":
    results = debug_trading_conditions()