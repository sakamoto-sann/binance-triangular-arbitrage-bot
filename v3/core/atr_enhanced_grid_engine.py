"""
ATR-Enhanced Grid Engine v4.0
Integrates ATR-based dynamic grid spacing with existing grid trading logic.
Maintains backward compatibility while adding intelligent volatility-based optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Import existing components
from .grid_engine import GridType, GridLevel
from .market_analyzer import MarketAnalysis, MarketRegime
from ..utils.config_manager import BotConfig
from ...advanced.atr_grid_optimizer import ATRGridOptimizer, ATRConfig, VolatilityRegime
from ...config.enhanced_features_config import get_config, is_feature_enabled, get_feature_config

logger = logging.getLogger(__name__)

@dataclass
class ATRGridParameters:
    """ATR-enhanced grid parameters."""
    base_spacing: float          # Original static spacing
    atr_spacing: float          # ATR-calculated spacing
    final_spacing: float        # Final spacing used
    volatility_regime: str      # Current volatility regime
    atr_value: float           # Current ATR value
    confidence: float          # Confidence in ATR calculation
    fallback_used: bool        # Whether fallback to static was used
    timestamp: datetime

class ATREnhancedGridEngine:
    """
    Enhanced grid engine with ATR-based dynamic spacing optimization.
    
    Features:
    - Backward compatible with existing grid engine
    - ATR-based dynamic grid spacing
    - Volatility regime detection
    - Conservative fallback mechanisms
    - Performance tracking and monitoring
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize ATR-enhanced grid engine.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        
        # Enhanced features configuration
        self.enhanced_config = get_config()
        self.atr_enabled = is_feature_enabled('atr_grid_optimization')
        self.atr_config_dict = get_feature_config('atr_grid_optimization')
        
        # Initialize ATR optimizer if enabled
        if self.atr_enabled:
            try:
                atr_config = ATRConfig(
                    atr_period=self.atr_config_dict.get('atr_period', 14),
                    regime_lookback=self.atr_config_dict.get('regime_lookback', 100),
                    update_frequency_hours=self.atr_config_dict.get('update_frequency_hours', 2),
                    low_vol_multiplier=self.atr_config_dict.get('low_vol_multiplier', 0.08),
                    normal_vol_multiplier=self.atr_config_dict.get('normal_vol_multiplier', 0.12),
                    high_vol_multiplier=self.atr_config_dict.get('high_vol_multiplier', 0.15),
                    extreme_vol_multiplier=self.atr_config_dict.get('extreme_vol_multiplier', 0.20),
                    low_threshold=self.atr_config_dict.get('low_threshold', 0.25),
                    high_threshold=self.atr_config_dict.get('high_threshold', 0.75),
                    extreme_threshold=self.atr_config_dict.get('extreme_threshold', 0.95),
                    min_grid_spacing=self.atr_config_dict.get('min_grid_spacing', 0.001),
                    max_grid_spacing=self.atr_config_dict.get('max_grid_spacing', 0.05)
                )
                
                self.atr_optimizer = ATRGridOptimizer(atr_config)
                logger.info("ATR Grid Optimizer initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize ATR optimizer: {e}")
                self.atr_enabled = False
                self.atr_optimizer = None
        else:
            self.atr_optimizer = None
            logger.info("ATR optimization disabled - using static grid spacing")
        
        # Grid state
        self.active_grids: Dict[str, List[GridLevel]] = {}
        self.grid_parameters_history: List[ATRGridParameters] = []
        
        # Performance tracking
        self.static_performance_baseline = 0.0
        self.atr_performance_current = 0.0
        self.fallback_count = 0
        self.optimization_count = 0
        
        # Default grid configuration
        self.default_grid_config = {
            'spacing_pct': 0.005,  # 0.5% default spacing
            'levels_count': 10,    # Number of grid levels
            'max_position_size': 0.15,  # 15% max position
            'rebalance_threshold': 0.02  # 2% rebalance threshold
        }
        
        logger.info(f"ATR-Enhanced Grid Engine initialized (ATR: {'enabled' if self.atr_enabled else 'disabled'})")
    
    def calculate_dynamic_grid_spacing(self, symbol: str, current_price: float, 
                                     price_history: pd.DataFrame) -> ATRGridParameters:
        """
        Calculate dynamic grid spacing using ATR optimization.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            price_history: Historical OHLC data
            
        Returns:
            ATR grid parameters with spacing recommendation
        """
        try:
            # Default spacing from configuration
            base_spacing = self.default_grid_config['spacing_pct']
            
            # If ATR is disabled or not available, use static spacing
            if not self.atr_enabled or not self.atr_optimizer:
                return ATRGridParameters(
                    base_spacing=base_spacing,
                    atr_spacing=base_spacing,
                    final_spacing=base_spacing,
                    volatility_regime='static',
                    atr_value=0.0,
                    confidence=1.0,
                    fallback_used=False,
                    timestamp=datetime.now()
                )
            
            # Validate price history
            if len(price_history) < self.atr_optimizer.config.atr_period:
                logger.warning(f"Insufficient price history for ATR calculation: {len(price_history)} < {self.atr_optimizer.config.atr_period}")
                self.fallback_count += 1
                return ATRGridParameters(
                    base_spacing=base_spacing,
                    atr_spacing=base_spacing,
                    final_spacing=base_spacing,
                    volatility_regime='insufficient_data',
                    atr_value=0.0,
                    confidence=0.0,
                    fallback_used=True,
                    timestamp=datetime.now()
                )
            
            # Run ATR analysis
            volatility_analysis = self.atr_optimizer.analyze_market_conditions(price_history)
            
            # Get ATR-optimized grid parameters
            grid_params = self.atr_optimizer.get_grid_parameters(current_price)
            
            atr_spacing = grid_params['spacing_pct']
            
            # Conservative validation: ensure ATR spacing is reasonable
            spacing_change_ratio = atr_spacing / base_spacing
            max_change_ratio = 3.0  # Allow up to 3x change from base spacing
            
            if spacing_change_ratio > max_change_ratio or spacing_change_ratio < (1/max_change_ratio):
                logger.warning(f"ATR spacing change too extreme: {spacing_change_ratio:.2f}x, using fallback")
                final_spacing = base_spacing
                fallback_used = True
                self.fallback_count += 1
            else:
                final_spacing = atr_spacing
                fallback_used = False
            
            # Create parameters object
            atr_params = ATRGridParameters(
                base_spacing=base_spacing,
                atr_spacing=atr_spacing,
                final_spacing=final_spacing,
                volatility_regime=volatility_analysis.regime.value,
                atr_value=volatility_analysis.current_atr,
                confidence=volatility_analysis.confidence,
                fallback_used=fallback_used,
                timestamp=datetime.now()
            )
            
            # Track parameters history
            self.grid_parameters_history.append(atr_params)
            if len(self.grid_parameters_history) > 1000:
                self.grid_parameters_history = self.grid_parameters_history[-1000:]
            
            self.optimization_count += 1
            
            logger.info(f"Dynamic grid spacing calculated: {final_spacing*100:.2f}% "
                       f"(base: {base_spacing*100:.2f}%, ATR: {atr_spacing*100:.2f}%, "
                       f"regime: {volatility_analysis.regime.value}, "
                       f"confidence: {volatility_analysis.confidence:.2f})")
            
            return atr_params
            
        except Exception as e:
            logger.error(f"Error calculating dynamic grid spacing: {e}")
            self.fallback_count += 1
            return ATRGridParameters(
                base_spacing=base_spacing,
                atr_spacing=base_spacing,
                final_spacing=base_spacing,
                volatility_regime='error',
                atr_value=0.0,
                confidence=0.0,
                fallback_used=True,
                timestamp=datetime.now()
            )
    
    def generate_grid_levels(self, symbol: str, center_price: float, 
                           price_history: pd.DataFrame, 
                           grid_type: GridType = GridType.MEDIUM) -> List[GridLevel]:
        """
        Generate grid levels with ATR-optimized spacing.
        
        Args:
            symbol: Trading symbol
            center_price: Center price for grid
            price_history: Historical price data
            grid_type: Type of grid to generate
            
        Returns:
            List of optimized grid levels
        """
        try:
            # Calculate dynamic spacing
            atr_params = self.calculate_dynamic_grid_spacing(symbol, center_price, price_history)
            
            # Use final spacing (ATR-optimized or fallback)
            spacing_pct = atr_params.final_spacing
            
            # Grid configuration based on type
            grid_configs = {
                GridType.FAST: {'levels': 6, 'quantity_base': 0.001},
                GridType.MEDIUM: {'levels': 10, 'quantity_base': 0.001},
                GridType.SLOW: {'levels': 16, 'quantity_base': 0.0008}
            }
            
            config = grid_configs.get(grid_type, grid_configs[GridType.MEDIUM])
            levels_count = config['levels']
            base_quantity = config['quantity_base']
            
            # Generate grid levels
            grid_levels = []
            current_time = datetime.now()
            
            # Calculate level prices around center
            for i in range(levels_count):
                level_offset = (i - levels_count // 2) * spacing_pct
                level_price = center_price * (1 + level_offset)
                
                # Determine side based on price relative to center
                side = 'buy' if level_price < center_price else 'sell'
                
                # Adjust quantity based on distance from center (closer = larger quantity)
                distance_factor = 1.0 - abs(level_offset) / (levels_count * spacing_pct / 2)
                level_quantity = base_quantity * (0.5 + 0.5 * distance_factor)
                
                # Calculate confidence based on ATR analysis
                base_confidence = atr_params.confidence
                distance_penalty = abs(level_offset) * 2  # Reduce confidence for distant levels
                level_confidence = max(0.1, base_confidence - distance_penalty)
                
                # Determine priority (closer to center = higher priority)
                priority = int(abs(level_offset) * 100) + 1
                
                # Create grid level
                level = GridLevel(
                    level_id=f"{symbol}_{grid_type.value}_{i}_{int(current_time.timestamp())}",
                    price=level_price,
                    side=side,
                    quantity=level_quantity,
                    grid_type=grid_type,
                    confidence=level_confidence,
                    priority=priority,
                    created_at=current_time,
                    is_active=True
                )
                
                grid_levels.append(level)
            
            # Store grid for tracking
            grid_key = f"{symbol}_{grid_type.value}"
            self.active_grids[grid_key] = grid_levels
            
            logger.info(f"Generated {len(grid_levels)} grid levels for {symbol} "
                       f"({grid_type.value}) with {spacing_pct*100:.2f}% spacing "
                       f"(regime: {atr_params.volatility_regime})")
            
            return grid_levels
            
        except Exception as e:
            logger.error(f"Error generating grid levels: {e}")
            return []
    
    def should_rebalance_grid(self, symbol: str, current_price: float, 
                            grid_type: GridType = GridType.MEDIUM) -> bool:
        """
        Determine if grid should be rebalanced based on price movement and ATR changes.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            grid_type: Grid type to check
            
        Returns:
            True if grid should be rebalanced
        """
        try:
            grid_key = f"{symbol}_{grid_type.value}"
            
            if grid_key not in self.active_grids:
                return True  # No grid exists, should create one
            
            grid_levels = self.active_grids[grid_key]
            if not grid_levels:
                return True  # Empty grid, should recreate
            
            # Check if price has moved significantly from grid center
            prices = [level.price for level in grid_levels]
            grid_center = sum(prices) / len(prices)
            
            price_deviation = abs(current_price - grid_center) / grid_center
            
            # Check if ATR optimizer suggests rebalancing
            atr_rebalance = False
            if self.atr_enabled and self.atr_optimizer:
                atr_rebalance = self.atr_optimizer.should_update_grid()
            
            # Rebalancing criteria
            rebalance_threshold = self.default_grid_config['rebalance_threshold']
            
            should_rebalance = (
                price_deviation > rebalance_threshold or  # Price moved too far
                atr_rebalance or  # ATR suggests update
                len([l for l in grid_levels if l.is_active]) < len(grid_levels) * 0.5  # Too many levels filled
            )
            
            if should_rebalance:
                logger.info(f"Grid rebalancing triggered for {symbol}: "
                           f"price_deviation={price_deviation:.3f}, "
                           f"atr_rebalance={atr_rebalance}")
            
            return should_rebalance
            
        except Exception as e:
            logger.error(f"Error checking grid rebalancing: {e}")
            return False
    
    def get_grid_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for ATR-enhanced grid system.
        
        Returns:
            Comprehensive performance metrics
        """
        try:
            metrics = {
                'atr_enabled': self.atr_enabled,
                'optimization_count': self.optimization_count,
                'fallback_count': self.fallback_count,
                'fallback_rate': self.fallback_count / max(1, self.optimization_count),
                'active_grids': len(self.active_grids),
                'total_grid_levels': sum(len(levels) for levels in self.active_grids.values())
            }
            
            # ATR optimizer metrics
            if self.atr_enabled and self.atr_optimizer:
                atr_metrics = self.atr_optimizer.get_performance_metrics()
                metrics.update({
                    'atr_regime_changes': atr_metrics.get('regime_changes', 0),
                    'atr_average_confidence': atr_metrics.get('average_confidence', 0.0),
                    'atr_current_regime': atr_metrics.get('current_regime', 'none'),
                    'atr_regime_distribution': atr_metrics.get('regime_distribution', {})
                })
            
            # Grid parameters history analysis
            if self.grid_parameters_history:
                recent_params = self.grid_parameters_history[-50:]  # Last 50 calculations
                
                spacing_values = [p.final_spacing for p in recent_params]
                confidence_values = [p.confidence for p in recent_params]
                
                metrics.update({
                    'average_spacing_pct': np.mean(spacing_values) * 100,
                    'spacing_std_pct': np.std(spacing_values) * 100,
                    'min_spacing_pct': min(spacing_values) * 100,
                    'max_spacing_pct': max(spacing_values) * 100,
                    'average_confidence': np.mean(confidence_values),
                    'parameters_history_length': len(self.grid_parameters_history)
                })
                
                # Regime distribution from parameters
                regime_counts = {}
                for param in recent_params:
                    regime = param.volatility_regime
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                regime_distribution = {
                    regime: count / len(recent_params) * 100
                    for regime, count in regime_counts.items()
                }
                
                metrics['spacing_regime_distribution'] = regime_distribution
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting grid performance metrics: {e}")
            return {
                'atr_enabled': self.atr_enabled,
                'error': str(e)
            }
    
    def get_current_grid_status(self, symbol: str, 
                              grid_type: GridType = GridType.MEDIUM) -> Dict[str, Any]:
        """
        Get current status of grid for a symbol.
        
        Args:
            symbol: Trading symbol
            grid_type: Grid type
            
        Returns:
            Grid status information
        """
        try:
            grid_key = f"{symbol}_{grid_type.value}"
            
            if grid_key not in self.active_grids:
                return {
                    'symbol': symbol,
                    'grid_type': grid_type.value,
                    'exists': False,
                    'levels_count': 0,
                    'active_levels': 0
                }
            
            grid_levels = self.active_grids[grid_key]
            active_levels = [level for level in grid_levels if level.is_active]
            
            # Calculate grid statistics
            prices = [level.price for level in grid_levels]
            quantities = [level.quantity for level in grid_levels]
            confidences = [level.confidence for level in grid_levels]
            
            # Get latest ATR parameters
            latest_params = None
            if self.grid_parameters_history:
                latest_params = self.grid_parameters_history[-1]
            
            status = {
                'symbol': symbol,
                'grid_type': grid_type.value,
                'exists': True,
                'levels_count': len(grid_levels),
                'active_levels': len(active_levels),
                'price_range': {
                    'min': min(prices) if prices else 0,
                    'max': max(prices) if prices else 0,
                    'center': sum(prices) / len(prices) if prices else 0
                },
                'total_quantity': sum(quantities),
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'created_at': grid_levels[0].created_at.isoformat() if grid_levels else None
            }
            
            # Add ATR information if available
            if latest_params:
                status['atr_info'] = {
                    'spacing_pct': latest_params.final_spacing * 100,
                    'volatility_regime': latest_params.volatility_regime,
                    'atr_value': latest_params.atr_value,
                    'confidence': latest_params.confidence,
                    'fallback_used': latest_params.fallback_used
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting grid status: {e}")
            return {
                'symbol': symbol,
                'grid_type': grid_type.value,
                'exists': False,
                'error': str(e)
            }

# Backward compatibility function
def create_enhanced_grid_engine(config: BotConfig) -> ATREnhancedGridEngine:
    """
    Create ATR-enhanced grid engine with backward compatibility.
    
    Args:
        config: Bot configuration
        
    Returns:
        Enhanced grid engine instance
    """
    return ATREnhancedGridEngine(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample configuration
    class MockBotConfig:
        def __init__(self):
            self.trading = type('', (), {
                'min_order_size': 0.0001,
                'max_order_size': 1.0
            })()
    
    config = MockBotConfig()
    
    # Create enhanced grid engine
    engine = ATREnhancedGridEngine(config)
    
    # Generate sample price data
    def create_sample_data(periods=200):
        base_price = 55000.0
        dates = [datetime.now() - timedelta(hours=periods-i) for i in range(periods)]
        
        prices = []
        current_price = base_price
        
        for i in range(periods):
            # Random walk with volatility clustering
            volatility = np.random.uniform(0.01, 0.05)
            change = np.random.normal(0, volatility)
            current_price *= (1 + change)
            
            # Generate OHLC
            daily_vol = np.random.uniform(0.005, 0.02)
            high = current_price * (1 + np.random.uniform(0, daily_vol))
            low = current_price * (1 - np.random.uniform(0, daily_vol))
            open_price = current_price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
            
            prices.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': np.random.uniform(1000, 10000)
            })
        
        return pd.DataFrame(prices)
    
    print("=== ATR-Enhanced Grid Engine Test ===")
    
    # Generate test data
    price_data = create_sample_data(200)
    current_price = price_data['close'].iloc[-1]
    symbol = "BTCUSDT"
    
    print(f"Generated {len(price_data)} price periods")
    print(f"Current price: ${current_price:.2f}")
    print(f"Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # Generate grid levels
    print(f"\n=== Generating Grid Levels ===")
    grid_levels = engine.generate_grid_levels(symbol, current_price, price_data, GridType.MEDIUM)
    
    print(f"Generated {len(grid_levels)} grid levels:")
    for i, level in enumerate(grid_levels[:5]):  # Show first 5 levels
        print(f"  Level {i+1}: {level.side.upper()} {level.quantity:.4f} at ${level.price:.2f} "
              f"(confidence: {level.confidence:.2f})")
    
    # Get grid status
    status = engine.get_current_grid_status(symbol, GridType.MEDIUM)
    print(f"\n=== Grid Status ===")
    print(f"Exists: {status['exists']}")
    print(f"Levels: {status['levels_count']} total, {status['active_levels']} active")
    print(f"Price range: ${status['price_range']['min']:.2f} - ${status['price_range']['max']:.2f}")
    print(f"Center: ${status['price_range']['center']:.2f}")
    
    if 'atr_info' in status:
        atr_info = status['atr_info']
        print(f"ATR spacing: {atr_info['spacing_pct']:.2f}%")
        print(f"Volatility regime: {atr_info['volatility_regime']}")
        print(f"Confidence: {atr_info['confidence']:.2f}")
    
    # Get performance metrics
    metrics = engine.get_grid_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"ATR enabled: {metrics['atr_enabled']}")
    print(f"Optimizations: {metrics['optimization_count']}")
    print(f"Fallback rate: {metrics['fallback_rate']:.1%}")
    
    if 'average_spacing_pct' in metrics:
        print(f"Average spacing: {metrics['average_spacing_pct']:.2f}%")
        print(f"Current regime: {metrics['atr_current_regime']}")
    
    print(f"\n✅ ATR-Enhanced Grid Engine test completed successfully!")