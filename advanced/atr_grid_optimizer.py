"""
ATR-Based Dynamic Grid Spacing Optimizer
Advanced enhancement for crypto grid trading bot - v4.0.0

This module implements dynamic grid spacing based on Average True Range (ATR)
and volatility regime detection for improved performance and risk management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ATRConfig:
    """ATR calculation and optimization configuration."""
    atr_period: int = 14
    regime_lookback: int = 100
    update_frequency_hours: int = 2
    
    # Grid spacing multipliers based on volatility regime
    low_vol_multiplier: float = 0.08      # 8% of ATR
    normal_vol_multiplier: float = 0.12   # 12% of ATR  
    high_vol_multiplier: float = 0.15     # 15% of ATR
    extreme_vol_multiplier: float = 0.20  # 20% of ATR
    
    # Volatility regime thresholds (percentiles)
    low_threshold: float = 0.25      # 25th percentile
    high_threshold: float = 0.75     # 75th percentile
    extreme_threshold: float = 0.95  # 95th percentile
    
    # Safety limits
    min_grid_spacing: float = 0.001  # Minimum 0.1% spacing
    max_grid_spacing: float = 0.05   # Maximum 5% spacing

@dataclass 
class VolatilityAnalysis:
    """Volatility analysis results."""
    current_atr: float
    atr_percentile: float
    regime: VolatilityRegime
    recommended_spacing: float
    confidence: float
    timestamp: datetime

class ATRGridOptimizer:
    """
    Advanced ATR-based grid spacing optimizer with volatility regime detection.
    
    Features:
    - 14-period ATR calculation
    - Dynamic volatility regime classification
    - Adaptive grid spacing recommendations
    - Conservative risk management integration
    - Real-time optimization capabilities
    """
    
    def __init__(self, config: ATRConfig = None):
        """
        Initialize ATR grid optimizer.
        
        Args:
            config: ATR configuration parameters
        """
        self.config = config or ATRConfig()
        
        # Historical data storage
        self.price_history: List[Dict] = []
        self.atr_history: List[float] = []
        self.volatility_history: List[VolatilityAnalysis] = []
        
        # Current state
        self.current_analysis: Optional[VolatilityAnalysis] = None
        self.last_update: Optional[datetime] = None
        
        # Performance tracking
        self.optimization_count: int = 0
        self.regime_changes: int = 0
        self.last_regime: Optional[VolatilityRegime] = None
        
        logger.info("ATR Grid Optimizer initialized with conservative parameters")
    
    def calculate_atr(self, price_data: pd.DataFrame) -> float:
        """
        Calculate Average True Range (ATR) for given price data.
        
        Args:
            price_data: DataFrame with OHLC columns
            
        Returns:
            Current ATR value
        """
        try:
            if len(price_data) < self.config.atr_period:
                logger.warning(f"Insufficient data for ATR calculation: {len(price_data)} < {self.config.atr_period}")
                return 0.0
            
            # Calculate True Range components
            high_low = price_data['high'] - price_data['low']
            high_close_prev = abs(price_data['high'] - price_data['close'].shift(1))
            low_close_prev = abs(price_data['low'] - price_data['close'].shift(1))
            
            # True Range is the maximum of the three components
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # Calculate ATR as simple moving average of True Range
            atr = true_range.rolling(window=self.config.atr_period).mean().iloc[-1]
            
            # Validation
            if pd.isna(atr) or atr <= 0:
                logger.error(f"Invalid ATR calculation: {atr}")
                return 0.0
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def classify_volatility_regime(self, current_atr: float, price: float) -> Tuple[VolatilityRegime, float]:
        """
        Classify current volatility regime based on ATR percentiles.
        
        Args:
            current_atr: Current ATR value
            price: Current price for normalization
            
        Returns:
            Tuple of (regime, confidence_score)
        """
        try:
            if len(self.atr_history) < self.config.regime_lookback:
                # Not enough history, assume normal regime
                return VolatilityRegime.NORMAL, 0.5
            
            # Calculate ATR as percentage of price for normalization
            atr_pct = (current_atr / price) * 100 if price > 0 else 0
            
            # Get recent ATR history for percentile calculation
            recent_atr_history = self.atr_history[-self.config.regime_lookback:]
            
            # Calculate percentile of current ATR
            atr_percentile = np.percentile(recent_atr_history, 
                                         [q for q in range(101) if np.percentile(recent_atr_history, q) <= current_atr][-1] if recent_atr_history else 50)
            
            # Classify regime based on percentiles
            if atr_percentile >= self.config.extreme_threshold * 100:
                regime = VolatilityRegime.EXTREME
                confidence = min(0.95, (atr_percentile - 95) / 5 + 0.8)
            elif atr_percentile >= self.config.high_threshold * 100:
                regime = VolatilityRegime.HIGH
                confidence = min(0.9, (atr_percentile - 75) / 20 + 0.7)
            elif atr_percentile <= self.config.low_threshold * 100:
                regime = VolatilityRegime.LOW
                confidence = min(0.9, (25 - atr_percentile) / 25 + 0.7)
            else:
                regime = VolatilityRegime.NORMAL
                confidence = 0.8 - abs(atr_percentile - 50) / 50 * 0.3
            
            return regime, max(0.5, confidence)
            
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {e}")
            return VolatilityRegime.NORMAL, 0.5
    
    def calculate_optimal_grid_spacing(self, atr: float, regime: VolatilityRegime, 
                                     price: float) -> float:
        """
        Calculate optimal grid spacing based on ATR and volatility regime.
        
        Args:
            atr: Current ATR value
            regime: Volatility regime
            price: Current price
            
        Returns:
            Optimal grid spacing as percentage
        """
        try:
            # Select multiplier based on regime
            multipliers = {
                VolatilityRegime.LOW: self.config.low_vol_multiplier,
                VolatilityRegime.NORMAL: self.config.normal_vol_multiplier,
                VolatilityRegime.HIGH: self.config.high_vol_multiplier,
                VolatilityRegime.EXTREME: self.config.extreme_vol_multiplier
            }
            
            multiplier = multipliers.get(regime, self.config.normal_vol_multiplier)
            
            # Calculate spacing as percentage of price
            spacing_amount = atr * multiplier
            spacing_pct = (spacing_amount / price) if price > 0 else self.config.min_grid_spacing
            
            # Apply safety limits
            spacing_pct = max(self.config.min_grid_spacing, 
                            min(self.config.max_grid_spacing, spacing_pct))
            
            logger.debug(f"Grid spacing calculated: {spacing_pct:.4f} ({spacing_pct*100:.2f}%) for {regime.value} volatility")
            
            return spacing_pct
            
        except Exception as e:
            logger.error(f"Error calculating grid spacing: {e}")
            return self.config.normal_vol_multiplier
    
    def analyze_market_conditions(self, price_data: pd.DataFrame) -> VolatilityAnalysis:
        """
        Perform comprehensive volatility analysis and grid optimization.
        
        Args:
            price_data: Recent OHLC price data
            
        Returns:
            Complete volatility analysis results
        """
        try:
            current_price = float(price_data['close'].iloc[-1])
            
            # Calculate current ATR
            current_atr = self.calculate_atr(price_data)
            
            # Add to ATR history
            self.atr_history.append(current_atr)
            
            # Keep history within limits
            if len(self.atr_history) > self.config.regime_lookback * 2:
                self.atr_history = self.atr_history[-self.config.regime_lookback:]
            
            # Classify volatility regime
            regime, confidence = self.classify_volatility_regime(current_atr, current_price)
            
            # Calculate optimal grid spacing
            recommended_spacing = self.calculate_optimal_grid_spacing(
                current_atr, regime, current_price
            )
            
            # Calculate ATR percentile for analysis
            if len(self.atr_history) >= 10:
                atr_percentile = (np.searchsorted(sorted(self.atr_history[-100:]), current_atr) / 
                                len(self.atr_history[-100:]) * 100)
            else:
                atr_percentile = 50.0
            
            # Create analysis result
            analysis = VolatilityAnalysis(
                current_atr=current_atr,
                atr_percentile=atr_percentile,
                regime=regime,
                recommended_spacing=recommended_spacing,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Track regime changes
            if self.last_regime and self.last_regime != regime:
                self.regime_changes += 1
                logger.info(f"Volatility regime changed: {self.last_regime.value} -> {regime.value}")
            
            self.last_regime = regime
            self.current_analysis = analysis
            self.last_update = datetime.now()
            self.optimization_count += 1
            
            # Add to history
            self.volatility_history.append(analysis)
            if len(self.volatility_history) > 1000:
                self.volatility_history = self.volatility_history[-1000:]
            
            logger.info(f"Market analysis complete: {regime.value} volatility, "
                       f"{recommended_spacing*100:.2f}% grid spacing recommended")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            # Return safe default analysis
            return VolatilityAnalysis(
                current_atr=0.01,
                atr_percentile=50.0,
                regime=VolatilityRegime.NORMAL,
                recommended_spacing=self.config.normal_vol_multiplier,
                confidence=0.5,
                timestamp=datetime.now()
            )
    
    def should_update_grid(self) -> bool:
        """
        Determine if grid spacing should be updated based on time and conditions.
        
        Returns:
            True if grid should be updated
        """
        try:
            if not self.last_update:
                return True
            
            # Check time-based update
            time_since_update = datetime.now() - self.last_update
            if time_since_update >= timedelta(hours=self.config.update_frequency_hours):
                return True
            
            # Check regime change
            if self.current_analysis and self.last_regime:
                if self.current_analysis.regime != self.last_regime:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking grid update requirement: {e}")
            return False
    
    def get_grid_parameters(self, base_price: float) -> Dict[str, float]:
        """
        Get current grid parameters for implementation.
        
        Args:
            base_price: Base price for grid calculation
            
        Returns:
            Dictionary of grid parameters
        """
        try:
            if not self.current_analysis:
                # Return default parameters
                return {
                    'spacing_pct': self.config.normal_vol_multiplier,
                    'spacing_amount': base_price * self.config.normal_vol_multiplier,
                    'regime': 'normal',
                    'confidence': 0.5,
                    'atr': 0.0
                }
            
            analysis = self.current_analysis
            spacing_amount = base_price * analysis.recommended_spacing
            
            return {
                'spacing_pct': analysis.recommended_spacing,
                'spacing_amount': spacing_amount,
                'regime': analysis.regime.value,
                'confidence': analysis.confidence,
                'atr': analysis.current_atr,
                'atr_percentile': analysis.atr_percentile
            }
            
        except Exception as e:
            logger.error(f"Error getting grid parameters: {e}")
            return {
                'spacing_pct': self.config.normal_vol_multiplier,
                'spacing_amount': base_price * self.config.normal_vol_multiplier,
                'regime': 'error',
                'confidence': 0.0,
                'atr': 0.0
            }
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """
        Get optimizer performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        try:
            total_optimizations = self.optimization_count
            
            if len(self.volatility_history) == 0:
                return {
                    'total_optimizations': total_optimizations,
                    'regime_changes': self.regime_changes,
                    'average_confidence': 0.0,
                    'regime_distribution': {},
                    'current_regime': 'none'
                }
            
            # Calculate regime distribution
            regime_counts = {}
            total_confidence = 0.0
            
            for analysis in self.volatility_history[-100:]:  # Last 100 analyses
                regime = analysis.regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                total_confidence += analysis.confidence
            
            # Calculate percentages
            total_samples = len(self.volatility_history[-100:])
            regime_distribution = {
                regime: count / total_samples * 100 
                for regime, count in regime_counts.items()
            }
            
            average_confidence = total_confidence / total_samples if total_samples > 0 else 0.0
            
            return {
                'total_optimizations': total_optimizations,
                'regime_changes': self.regime_changes,
                'average_confidence': average_confidence,
                'regime_distribution': regime_distribution,
                'current_regime': self.current_analysis.regime.value if self.current_analysis else 'none',
                'atr_history_length': len(self.atr_history),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                'total_optimizations': 0,
                'regime_changes': 0,
                'average_confidence': 0.0,
                'regime_distribution': {},
                'current_regime': 'error'
            }

def create_sample_price_data(periods: int = 100) -> pd.DataFrame:
    """
    Create sample price data for testing ATR optimizer.
    
    Args:
        periods: Number of periods to generate
        
    Returns:
        Sample OHLC DataFrame
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic crypto price data
    base_price = 55000.0  # Starting price similar to BTC
    
    data = []
    current_price = base_price
    
    for i in range(periods):
        # Random walk with volatility clustering
        volatility = np.random.uniform(0.01, 0.08)  # 1-8% daily volatility
        price_change = np.random.normal(0, volatility)
        
        current_price *= (1 + price_change)
        
        # Generate OHLC from current price
        daily_volatility = np.random.uniform(0.005, 0.03)
        
        high = current_price * (1 + np.random.uniform(0, daily_volatility))
        low = current_price * (1 - np.random.uniform(0, daily_volatility))
        open_price = current_price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
        close_price = current_price
        
        data.append({
            'timestamp': datetime.now() - timedelta(hours=periods-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create ATR optimizer
    config = ATRConfig(
        atr_period=14,
        update_frequency_hours=1,
        low_vol_multiplier=0.08,
        normal_vol_multiplier=0.12,
        high_vol_multiplier=0.15
    )
    
    optimizer = ATRGridOptimizer(config)
    
    # Generate sample data
    price_data = create_sample_price_data(200)
    
    print("=== ATR Grid Optimizer Test ===")
    print(f"Generated {len(price_data)} price periods")
    print(f"Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # Run analysis
    analysis = optimizer.analyze_market_conditions(price_data)
    
    print(f"\n=== Analysis Results ===")
    print(f"Current ATR: {analysis.current_atr:.2f}")
    print(f"ATR Percentile: {analysis.atr_percentile:.1f}%")
    print(f"Volatility Regime: {analysis.regime.value}")
    print(f"Recommended Grid Spacing: {analysis.recommended_spacing*100:.2f}%")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    # Get grid parameters
    current_price = price_data['close'].iloc[-1]
    grid_params = optimizer.get_grid_parameters(current_price)
    
    print(f"\n=== Grid Parameters ===")
    print(f"Base Price: ${current_price:.2f}")
    print(f"Grid Spacing: {grid_params['spacing_pct']*100:.2f}% (${grid_params['spacing_amount']:.2f})")
    print(f"Regime: {grid_params['regime']}")
    print(f"Confidence: {grid_params['confidence']:.2f}")
    
    # Performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"\n=== Performance Metrics ===")
    print(f"Total Optimizations: {metrics['total_optimizations']}")
    print(f"Regime Changes: {metrics['regime_changes']}")
    print(f"Average Confidence: {metrics['average_confidence']:.2f}")
    print(f"Current Regime: {metrics['current_regime']}")
    
    print("\n✅ ATR Grid Optimizer test completed successfully!")