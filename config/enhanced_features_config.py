#!/usr/bin/env python3
"""
Enhanced Features Configuration
Implements Claude's winning conservative parameter strategy
"""

from typing import Dict, Any

# Claude's Conservative Parameter Implementation
# Winner of Claude vs Gemini optimization comparison
ENHANCED_FEATURES_CONFIG = {
    # Feature 1: Portfolio-Level Drawdown Control
    'drawdown_control': {
        'enabled': True,
        'halt_threshold': 0.20,       # 20% drawdown halt (conservative)
        'recovery_threshold': 0.10,   # 10% recovery threshold
        'emergency_threshold': 0.25,  # 25% emergency halt
        'high_water_mark_file': 'data/portfolio_hwm.json',
        'logging_enabled': True
    },
    
    # Feature 2: Volatility-Adjusted Position Sizing
    'volatility_sizing': {
        'enabled': True,
        'high_vol_threshold': 0.06,   # 6% daily volatility threshold
        'low_vol_threshold': 0.02,    # 2% daily volatility threshold
        'high_vol_multiplier': 1.00,  # Neutral sizing (no reduction) - conservative
        'low_vol_multiplier': 1.00,   # Neutral sizing (no increase) - conservative
        'max_position_size': 0.15,    # 15% max single position
        'portfolio_vol_limit': 0.03,  # 3% portfolio volatility limit
        'kelly_multiplier': 0.25      # Conservative Kelly fraction
    },
    
    # Feature 3: Smart Order Execution
    'smart_execution': {
        'enabled': True,
        'execution_efficiency': 1.0000,     # Neutral execution - conservative
        'order_scaling_threshold': 100000,  # $100k+ orders get scaled
        'scaling_slices': 3,                # Split large orders into 3 slices
        'timeout_seconds': 30,              # 30 second order timeout
        'max_slippage_pct': 0.5,            # 0.5% max acceptable slippage
        'maker_rebate_optimization': True,   # Optimize for maker rebates
        'tick_size_rounding': True          # Round to appropriate tick sizes
    },
    
    # Feature 4: ATR-Based Dynamic Grid Optimization
    'atr_grid_optimization': {
        'enabled': True,
        'atr_period': 14,                   # 14-period ATR calculation
        'regime_lookback': 100,             # 100 periods for regime detection
        'update_frequency_hours': 2,        # Update grid spacing every 2 hours
        
        # Conservative grid spacing multipliers
        'low_vol_multiplier': 0.08,         # 8% of ATR for low volatility
        'normal_vol_multiplier': 0.12,      # 12% of ATR for normal volatility  
        'high_vol_multiplier': 0.15,        # 15% of ATR for high volatility
        'extreme_vol_multiplier': 0.20,     # 20% of ATR for extreme volatility
        
        # Volatility regime thresholds (percentiles)
        'low_threshold': 0.25,              # 25th percentile
        'high_threshold': 0.75,             # 75th percentile
        'extreme_threshold': 0.95,          # 95th percentile
        
        # Safety limits
        'min_grid_spacing': 0.001,          # Minimum 0.1% spacing
        'max_grid_spacing': 0.05,           # Maximum 5% spacing
        
        # Performance tracking
        'enable_performance_tracking': True,
        'log_regime_changes': True,
        'fallback_to_static': True          # Fallback to static grid if ATR fails
    },
    
    # Integration Settings
    'integration': {
        'supertrend_preservation': True,    # Preserve existing Supertrend logic
        'baseline_performance_target': {
            'total_return': 250.2,          # Target baseline performance
            'sharpe_ratio': 5.74,
            'max_drawdown': 0.324
        },
        'monitoring': {
            'performance_check_frequency': 3600,  # Check every hour
            'alert_performance_degradation': True,
            'revert_threshold': 0.90        # Revert if performance < 90% of baseline
        }
    },
    
    # Conservative approach: Minimal feature impact with ATR optimization
    'philosophy': 'conservative_atr_enhanced',
    'strategy_name': 'Claude Conservative Enhancement with ATR Grid Optimization',
    'description': 'Conservative parameter strategy with dynamic ATR-based grid spacing for improved performance',
    'version': '4.0.0',
    'optimization_results': {
        'total_return': 183.2,
        'sharpe_ratio': 4.26,
        'max_drawdown': 0.042,
        'optimization_score': 0.790
    }
}

def get_config() -> Dict[str, Any]:
    """Get the enhanced features configuration"""
    return ENHANCED_FEATURES_CONFIG

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled"""
    return ENHANCED_FEATURES_CONFIG.get(feature_name, {}).get('enabled', False)

def get_feature_config(feature_name: str) -> Dict[str, Any]:
    """Get configuration for a specific feature"""
    return ENHANCED_FEATURES_CONFIG.get(feature_name, {})

def update_config(feature_name: str, updates: Dict[str, Any]) -> None:
    """Update configuration for a specific feature"""
    if feature_name in ENHANCED_FEATURES_CONFIG:
        ENHANCED_FEATURES_CONFIG[feature_name].update(updates)
    else:
        ENHANCED_FEATURES_CONFIG[feature_name] = updates

# Export for easy import
__all__ = ['ENHANCED_FEATURES_CONFIG', 'get_config', 'is_feature_enabled', 'get_feature_config', 'update_config']