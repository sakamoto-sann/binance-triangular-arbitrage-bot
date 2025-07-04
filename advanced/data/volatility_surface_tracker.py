"""
Advanced Trading System - Volatility Surface Tracker
Volatility surface construction and analysis for optimal grid spacing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class VolatilityData:
    """Volatility data point."""
    symbol: str
    timeframe: str  # "1m", "5m", "1h", "1d"
    realized_vol: float
    implied_vol: float = 0.0
    atr: float = 0.0
    parkinson_vol: float = 0.0
    garman_klass_vol: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class VolatilityRegime:
    """Volatility regime classification."""
    regime: str  # "low", "medium", "high", "extreme"
    current_vol: float
    historical_percentile: float
    regime_confidence: float
    expected_persistence_hours: int
    regime_change_probability: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GridSpacingRecommendation:
    """Grid spacing recommendation based on volatility."""
    symbol: str
    current_spacing: float
    recommended_spacing: float
    spacing_multiplier: float
    confidence: float
    rationale: str
    expected_fills_24h: int
    risk_adjustment: float
    timestamp: datetime = field(default_factory=datetime.now)

class VolatilitySurfaceTracker:
    """
    Advanced volatility surface tracking and analysis.
    
    Provides real-time volatility analysis across multiple timeframes
    and generates optimal grid spacing recommendations.
    """
    
    def __init__(self, market_data_aggregator):
        """
        Initialize volatility surface tracker.
        
        Args:
            market_data_aggregator: Market data aggregation system
        """
        self.market_data = market_data_aggregator
        
        # Data storage
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.current_volatilities: Dict[str, VolatilityData] = {}
        self.volatility_regimes: Dict[str, VolatilityRegime] = {}
        self.grid_recommendations: Dict[str, GridSpacingRecommendation] = {}
        
        # Analysis parameters
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.lookback_periods = {"1m": 1440, "5m": 288, "15m": 96, "1h": 168, "4h": 168, "1d": 30}
        
        # Volatility thresholds (annualized)
        self.vol_thresholds = {
            "low": 0.3,      # 30% annual
            "medium": 0.6,   # 60% annual
            "high": 1.0,     # 100% annual
            "extreme": 1.5   # 150% annual
        }
        
        # Processing tasks
        self.analysis_tasks: List[asyncio.Task] = []
        self.is_active = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("VolatilitySurfaceTracker initialized")
    
    async def start(self) -> bool:
        """Start volatility analysis."""
        try:
            self.logger.info("Starting volatility surface tracking...")
            
            # Start analysis tasks
            self.analysis_tasks = [
                asyncio.create_task(self._volatility_calculator()),
                asyncio.create_task(self._regime_detector()),
                asyncio.create_task(self._grid_spacing_optimizer()),
                asyncio.create_task(self._surface_updater())
            ]
            
            self.is_active = True
            self.logger.info("Volatility surface tracking started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting volatility tracking: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop volatility analysis."""
        try:
            self.logger.info("Stopping volatility surface tracking...")
            
            self.is_active = False
            
            # Cancel analysis tasks
            for task in self.analysis_tasks:
                task.cancel()
            
            self.logger.info("Volatility surface tracking stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping volatility tracking: {e}")
            return False
    
    async def _volatility_calculator(self):
        """Calculate various volatility measures."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    # Get historical data
                    historical_data = await self.market_data.get_historical_data(symbol, "binance", 1000)
                    
                    if len(historical_data) >= 30:
                        # Calculate volatilities for different timeframes
                        for timeframe in self.timeframes:
                            vol_data = await self._calculate_timeframe_volatility(
                                symbol, timeframe, historical_data
                            )
                            if vol_data:
                                key = f"{symbol}_{timeframe}"
                                self.current_volatilities[key] = vol_data
                                self.volatility_history[key].append(vol_data)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in volatility calculator: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_timeframe_volatility(self, symbol: str, timeframe: str, 
                                           historical_data: List[Any]) -> Optional[VolatilityData]:
        """Calculate volatility for specific timeframe."""
        try:
            if len(historical_data) < 10:
                return None
            
            # Convert to price series
            prices = [point.price for point in historical_data]
            
            # Resample data based on timeframe
            resampled_prices = self._resample_prices(prices, timeframe)
            
            if len(resampled_prices) < 5:
                return None
            
            # Calculate simple realized volatility
            returns = []
            for i in range(1, len(resampled_prices)):
                if resampled_prices[i-1] > 0:
                    ret = (resampled_prices[i] - resampled_prices[i-1]) / resampled_prices[i-1]
                    returns.append(ret)
            
            if not returns:
                return None
            
            # Annualization factor based on timeframe
            annualization_factors = {
                "1m": np.sqrt(365 * 24 * 60),
                "5m": np.sqrt(365 * 24 * 12),
                "15m": np.sqrt(365 * 24 * 4),
                "1h": np.sqrt(365 * 24),
                "4h": np.sqrt(365 * 6),
                "1d": np.sqrt(365)
            }
            
            factor = annualization_factors.get(timeframe, np.sqrt(365))
            
            returns_array = np.array(returns)
            realized_vol = np.std(returns_array) * factor
            
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(resampled_prices)
            
            # Calculate Parkinson volatility (if we had OHLC data)
            parkinson_vol = realized_vol  # Simplified
            
            # Calculate Garman-Klass volatility (if we had OHLC data)
            garman_klass_vol = realized_vol  # Simplified
            
            vol_data = VolatilityData(
                symbol=symbol,
                timeframe=timeframe,
                realized_vol=realized_vol,
                atr=atr,
                parkinson_vol=parkinson_vol,
                garman_klass_vol=garman_klass_vol
            )
            
            return vol_data
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}/{timeframe}: {e}")
            return None
    
    def _resample_prices(self, prices: List[float], timeframe: str) -> List[float]:
        """Resample prices to specified timeframe."""
        try:
            # Simple resampling (in production, would use proper OHLC resampling)
            resample_factors = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "1h": 60,
                "4h": 240,
                "1d": 1440
            }
            
            factor = resample_factors.get(timeframe, 1)
            
            if factor == 1:
                return prices
            
            # Take every nth price point
            return prices[::factor]
            
        except Exception as e:
            self.logger.error(f"Error resampling prices: {e}")
            return prices
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            # Simplified ATR calculation (assumes prices are highs/lows)
            ranges = []
            for i in range(1, len(prices)):
                true_range = abs(prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
                ranges.append(true_range)
            
            if len(ranges) >= period:
                return np.mean(ranges[-period:])
            
            return np.mean(ranges) if ranges else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def _regime_detector(self):
        """Detect volatility regime changes."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    # Use 1-hour volatility for regime detection
                    key = f"{symbol}_1h"
                    
                    if key in self.volatility_history and len(self.volatility_history[key]) >= 50:
                        regime = await self._detect_volatility_regime(symbol)
                        if regime:
                            self.volatility_regimes[symbol] = regime
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in regime detector: {e}")
                await asyncio.sleep(300)
    
    async def _detect_volatility_regime(self, symbol: str) -> Optional[VolatilityRegime]:
        """Detect current volatility regime for symbol."""
        try:
            key = f"{symbol}_1h"
            history = list(self.volatility_history[key])
            
            if len(history) < 50:
                return None
            
            # Get current and historical volatilities
            current_vol = history[-1].realized_vol
            historical_vols = [v.realized_vol for v in history[-500:]]  # Last 500 periods
            
            # Calculate percentile
            percentile = np.percentile(historical_vols, 
                                     sum(1 for v in historical_vols if v <= current_vol) / len(historical_vols) * 100)
            
            # Classify regime
            if current_vol <= self.vol_thresholds["low"]:
                regime = "low"
            elif current_vol <= self.vol_thresholds["medium"]:
                regime = "medium"
            elif current_vol <= self.vol_thresholds["high"]:
                regime = "high"
            else:
                regime = "extreme"
            
            # Calculate regime confidence
            regime_stability = self._calculate_regime_stability(history[-20:])
            
            # Estimate persistence
            persistence_hours = self._estimate_regime_persistence(history, current_vol)
            
            # Calculate regime change probability
            change_probability = self._calculate_regime_change_probability(history)
            
            volatility_regime = VolatilityRegime(
                regime=regime,
                current_vol=current_vol,
                historical_percentile=percentile,
                regime_confidence=regime_stability,
                expected_persistence_hours=persistence_hours,
                regime_change_probability=change_probability
            )
            
            return volatility_regime
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime for {symbol}: {e}")
            return None
    
    def _calculate_regime_stability(self, recent_history: List[VolatilityData]) -> float:
        """Calculate regime stability score."""
        try:
            if len(recent_history) < 5:
                return 0.5
            
            vols = [v.realized_vol for v in recent_history]
            cv = np.std(vols) / np.mean(vols) if np.mean(vols) > 0 else 1.0
            
            # Lower coefficient of variation = higher stability
            stability = max(0, 1 - cv)
            return min(stability, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime stability: {e}")
            return 0.5
    
    def _estimate_regime_persistence(self, history: List[VolatilityData], current_vol: float) -> int:
        """Estimate how long current regime might persist."""
        try:
            # Simple heuristic: look at historical regime durations
            regime_durations = []
            current_regime = self._classify_vol_regime(current_vol)
            
            i = len(history) - 1
            while i >= 0:
                # Find regime changes
                if self._classify_vol_regime(history[i].realized_vol) != current_regime:
                    # Count how long the regime lasted
                    duration = 0
                    j = i + 1
                    while (j < len(history) and 
                           self._classify_vol_regime(history[j].realized_vol) == current_regime):
                        duration += 1
                        j += 1
                    
                    if duration > 0:
                        regime_durations.append(duration)
                    
                    # Skip to start of this regime
                    while (i >= 0 and 
                           self._classify_vol_regime(history[i].realized_vol) != current_regime):
                        i -= 1
                else:
                    i -= 1
            
            if regime_durations:
                avg_duration_periods = np.mean(regime_durations)
                return int(avg_duration_periods)  # Assuming 1-hour periods
            
            return 24  # Default 24 hours
            
        except Exception as e:
            self.logger.error(f"Error estimating regime persistence: {e}")
            return 24
    
    def _classify_vol_regime(self, vol: float) -> str:
        """Classify volatility into regime."""
        if vol <= self.vol_thresholds["low"]:
            return "low"
        elif vol <= self.vol_thresholds["medium"]:
            return "medium"
        elif vol <= self.vol_thresholds["high"]:
            return "high"
        else:
            return "extreme"
    
    def _calculate_regime_change_probability(self, history: List[VolatilityData]) -> float:
        """Calculate probability of regime change."""
        try:
            if len(history) < 20:
                return 0.2  # Default 20%
            
            # Look at recent volatility changes
            recent_vols = [v.realized_vol for v in history[-10:]]
            vol_changes = np.diff(recent_vols)
            
            # High volatility of volatility suggests regime change
            vol_of_vol = np.std(vol_changes)
            normalized_vov = min(vol_of_vol / 0.1, 1.0)  # Normalize to 0-1
            
            return normalized_vov
            
        except Exception as e:
            self.logger.error(f"Error calculating regime change probability: {e}")
            return 0.2
    
    async def _grid_spacing_optimizer(self):
        """Optimize grid spacing based on volatility."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    if symbol in self.volatility_regimes:
                        recommendation = await self._calculate_optimal_spacing(symbol)
                        if recommendation:
                            self.grid_recommendations[symbol] = recommendation
                
                await asyncio.sleep(180)  # Update every 3 minutes
                
            except Exception as e:
                self.logger.error(f"Error in grid spacing optimizer: {e}")
                await asyncio.sleep(180)
    
    async def _calculate_optimal_spacing(self, symbol: str) -> Optional[GridSpacingRecommendation]:
        """Calculate optimal grid spacing for symbol."""
        try:
            if symbol not in self.volatility_regimes:
                return None
            
            regime = self.volatility_regimes[symbol]
            current_vol = regime.current_vol
            
            # Base spacing parameters
            base_spacing = 0.005  # 0.5% base
            
            # Volatility-based adjustment
            vol_multipliers = {
                "low": 0.6,      # Tighter spacing in low vol
                "medium": 1.0,   # Normal spacing
                "high": 1.5,     # Wider spacing in high vol
                "extreme": 2.0   # Much wider in extreme vol
            }
            
            vol_multiplier = vol_multipliers.get(regime.regime, 1.0)
            
            # ATR-based adjustment
            atr_key = f"{symbol}_1h"
            if atr_key in self.current_volatilities:
                atr = self.current_volatilities[atr_key].atr
                # Convert ATR to percentage
                current_price = await self._get_current_price(symbol)
                if current_price > 0 and atr > 0:
                    atr_pct = atr / current_price
                    # Use ATR as additional spacing guidance
                    atr_multiplier = max(0.5, min(atr_pct / 0.01, 2.0))  # 1% ATR = 1.0 multiplier
                    vol_multiplier = (vol_multiplier + atr_multiplier) / 2
            
            # Calculate recommended spacing
            recommended_spacing = base_spacing * vol_multiplier
            
            # Ensure reasonable bounds
            recommended_spacing = max(0.002, min(recommended_spacing, 0.02))  # 0.2% to 2%
            
            # Calculate confidence
            confidence = regime.regime_confidence * (1 - regime.regime_change_probability)
            
            # Estimate expected fills
            expected_fills = self._estimate_daily_fills(symbol, recommended_spacing, current_vol)
            
            # Risk adjustment for regime uncertainty
            risk_adjustment = 1.0 - (regime.regime_change_probability * 0.3)
            
            # Generate rationale
            rationale = f"Vol regime: {regime.regime} ({current_vol:.1%}), " \
                       f"Regime confidence: {regime.regime_confidence:.1%}, " \
                       f"Change prob: {regime.regime_change_probability:.1%}"
            
            recommendation = GridSpacingRecommendation(
                symbol=symbol,
                current_spacing=base_spacing,
                recommended_spacing=recommended_spacing,
                spacing_multiplier=vol_multiplier,
                confidence=confidence,
                rationale=rationale,
                expected_fills_24h=expected_fills,
                risk_adjustment=risk_adjustment
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal spacing for {symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            current_data = await self.market_data.get_current_data()
            prices = current_data.get("prices", {})
            
            for key, price_data in prices.items():
                if symbol in key:
                    return price_data.get("price", 0.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _estimate_daily_fills(self, symbol: str, spacing: float, volatility: float) -> int:
        """Estimate number of daily grid fills."""
        try:
            # Simple heuristic: higher volatility and tighter spacing = more fills
            base_fills = 50  # Base fills per day
            
            # Volatility factor
            vol_factor = volatility / 0.6  # Normalize to 60% vol
            
            # Spacing factor (tighter spacing = more fills)
            spacing_factor = 0.005 / spacing  # Normalize to 0.5% spacing
            
            estimated_fills = int(base_fills * vol_factor * spacing_factor)
            
            # Reasonable bounds
            return max(5, min(estimated_fills, 500))
            
        except Exception as e:
            self.logger.error(f"Error estimating daily fills: {e}")
            return 50
    
    async def _surface_updater(self):
        """Update volatility surface visualization data."""
        while self.is_active:
            try:
                # Update surface data for visualization
                # This would typically create 3D surface data for display
                
                await asyncio.sleep(900)  # Update every 15 minutes
                
            except Exception as e:
                self.logger.error(f"Error in surface updater: {e}")
                await asyncio.sleep(900)
    
    async def get_current_volatility(self, symbol: str, timeframe: str = "1h") -> Optional[VolatilityData]:
        """Get current volatility for symbol and timeframe."""
        key = f"{symbol}_{timeframe}"
        return self.current_volatilities.get(key)
    
    async def get_volatility_regime(self, symbol: str) -> Optional[VolatilityRegime]:
        """Get current volatility regime for symbol."""
        return self.volatility_regimes.get(symbol)
    
    async def get_grid_recommendation(self, symbol: str) -> Optional[GridSpacingRecommendation]:
        """Get grid spacing recommendation for symbol."""
        return self.grid_recommendations.get(symbol)
    
    async def get_volatility_surface(self, symbol: str) -> Dict[str, Any]:
        """Get volatility surface data for symbol."""
        try:
            surface_data = {}
            
            for timeframe in self.timeframes:
                key = f"{symbol}_{timeframe}"
                if key in self.current_volatilities:
                    vol_data = self.current_volatilities[key]
                    surface_data[timeframe] = {
                        "realized_vol": vol_data.realized_vol,
                        "atr": vol_data.atr,
                        "timestamp": vol_data.timestamp.isoformat()
                    }
            
            return {
                "symbol": symbol,
                "surface": surface_data,
                "regime": self.volatility_regimes.get(symbol),
                "recommendation": self.grid_recommendations.get(symbol)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting volatility surface for {symbol}: {e}")
            return {}
    
    def get_tracker_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        return {
            "is_active": self.is_active,
            "volatilities_tracked": len(self.current_volatilities),
            "regimes_detected": len(self.volatility_regimes),
            "recommendations_generated": len(self.grid_recommendations),
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "current_regimes": {
                symbol: regime.regime for symbol, regime in self.volatility_regimes.items()
            }
        }