"""
Grid Trading Bot v3.0 - Market Analyzer
Advanced market regime detection and analysis system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..utils.indicators import TechnicalIndicators
from ..utils.config_manager import BotConfig

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics."""
    current_volatility: float
    volatility_percentile: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'
    atr_normalized: float
    volatility_forecast: float

@dataclass
class TrendMetrics:
    """Trend analysis metrics."""
    trend_strength: float  # -1 to 1 scale
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_confidence: float  # 0 to 1 scale
    trend_duration: int  # Number of periods
    trend_momentum: float
    reversal_probability: float

@dataclass
class MarketAnalysis:
    """Complete market analysis result."""
    regime: MarketRegime
    confidence: float
    volatility_metrics: VolatilityMetrics
    trend_metrics: TrendMetrics
    price_level: float
    support_levels: List[float]
    resistance_levels: List[float]
    breakout_probability: float
    mean_reversion_probability: float
    timestamp: datetime

class MarketAnalyzer:
    """Advanced market regime detection and analysis system."""
    
    def __init__(self, config: BotConfig):
        """
        Initialize market analyzer.
        
        Args:
            config: Bot configuration.
        """
        self.config = config
        self.indicators = TechnicalIndicators()
        
        # Analysis parameters
        self.ma_fast = config.market_regime.ma_fast
        self.ma_slow = config.market_regime.ma_slow
        self.atr_period = config.market_regime.atr_period
        self.volatility_periods = config.market_regime.volatility_periods
        self.breakout_threshold = config.market_regime.breakout_threshold
        self.sideways_threshold = config.market_regime.sideways_threshold
        
        # Historical analysis data
        self.analysis_history: List[MarketAnalysis] = []
        self.regime_transitions: List[Tuple[datetime, MarketRegime, MarketRegime]] = []
        
        logger.info("Market analyzer initialized")
    
    def analyze_market(self, price_data: pd.DataFrame) -> MarketAnalysis:
        """
        Perform comprehensive market analysis.
        
        Args:
            price_data: DataFrame with OHLCV data.
            
        Returns:
            Complete market analysis.
        """
        try:
            # OPTIMIZED: Reduced minimum data requirement for faster trading start
            min_required_data = min(50, max(self.ma_slow, max(self.volatility_periods)))
            if len(price_data) < min_required_data:
                logger.warning(f"Insufficient data for analysis: {len(price_data)} rows, need {min_required_data}")
                return self._create_unknown_analysis(price_data.iloc[-1]['close'])
            
            # Detect market regime
            regime, confidence = self._detect_market_regime(price_data)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(price_data)
            
            # Calculate trend metrics
            trend_metrics = self._calculate_trend_metrics(price_data)
            
            # Identify support/resistance levels
            support_levels, resistance_levels = self._identify_levels(price_data)
            
            # Calculate probabilities
            breakout_prob = self._calculate_breakout_probability(price_data, volatility_metrics)
            mean_reversion_prob = self._calculate_mean_reversion_probability(price_data, trend_metrics)
            
            # Create analysis result
            analysis = MarketAnalysis(
                regime=regime,
                confidence=confidence,
                volatility_metrics=volatility_metrics,
                trend_metrics=trend_metrics,
                price_level=float(price_data.iloc[-1]['close']),
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                breakout_probability=breakout_prob,
                mean_reversion_probability=mean_reversion_prob,
                timestamp=datetime.now()
            )
            
            # Update history
            self._update_analysis_history(analysis)
            
            logger.debug(f"Market analysis completed: {regime.value} regime with {confidence:.2f} confidence")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._create_unknown_analysis(price_data.iloc[-1]['close'])
    
    def _detect_market_regime(self, price_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.
        
        Args:
            price_data: DataFrame with OHLCV data.
            
        Returns:
            Tuple of (regime, confidence).
        """
        try:
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Calculate moving averages
            ma_fast = self.indicators.ema(close_prices, self.ma_fast)
            ma_slow = self.indicators.ema(close_prices, self.ma_slow)
            
            # Current values (most recent non-NaN)
            current_price = close_prices[-1]
            current_ma_fast = ma_fast[-1] if not np.isnan(ma_fast[-1]) else current_price
            current_ma_slow = ma_slow[-1] if not np.isnan(ma_slow[-1]) else current_price
            
            # Calculate ATR for volatility assessment
            atr = self.indicators.atr(high_prices, low_prices, close_prices, self.atr_period)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
            
            # Trend analysis
            trend_direction = 1 if current_ma_fast > current_ma_slow else -1
            
            # MA separation (normalized by price)
            ma_separation = abs(current_ma_fast - current_ma_slow) / current_price
            
            # Price position relative to MAs
            price_above_fast = current_price > current_ma_fast
            price_above_slow = current_price > current_ma_slow
            
            # Calculate price range for sideways detection
            lookback_period = min(self.ma_slow, len(close_prices))
            recent_high = np.max(high_prices[-lookback_period:])
            recent_low = np.min(low_prices[-lookback_period:])
            price_range = (recent_high - recent_low) / current_price
            
            # Breakout detection
            breakout_threshold_price = current_atr * self.breakout_threshold / current_price
            
            # Initialize confidence
            confidence = 0.5
            
            # Regime detection logic
            if price_range <= self.sideways_threshold:
                # Sideways market: low price range
                regime = MarketRegime.SIDEWAYS
                confidence = min(0.9, 0.5 + (self.sideways_threshold - price_range) * 2)
                
            elif ma_separation > breakout_threshold_price:
                # Potential breakout: significant MA separation
                if current_price > recent_high * 0.995:  # Breaking above recent high
                    regime = MarketRegime.BREAKOUT
                    confidence = min(0.9, 0.6 + ma_separation * 5)
                elif current_price < recent_low * 1.005:  # Breaking below recent low
                    regime = MarketRegime.BREAKOUT
                    confidence = min(0.9, 0.6 + ma_separation * 5)
                else:
                    # Strong trend
                    regime = MarketRegime.BULL if trend_direction > 0 else MarketRegime.BEAR
                    confidence = min(0.9, 0.6 + ma_separation * 3)
                    
            elif trend_direction > 0 and price_above_fast and price_above_slow:
                # Bull market: price above both MAs, fast MA above slow MA
                regime = MarketRegime.BULL
                confidence = min(0.85, 0.5 + ma_separation * 2)
                
                # Increase confidence if trend is consistent
                if len(ma_fast) >= 10:
                    recent_trend = np.mean(np.diff(ma_fast[-10:]))
                    if recent_trend > 0:
                        confidence += 0.1
                        
            elif trend_direction < 0 and not price_above_fast and not price_above_slow:
                # Bear market: price below both MAs, fast MA below slow MA
                regime = MarketRegime.BEAR
                confidence = min(0.85, 0.5 + ma_separation * 2)
                
                # Increase confidence if trend is consistent
                if len(ma_fast) >= 10:
                    recent_trend = np.mean(np.diff(ma_fast[-10:]))
                    if recent_trend < 0:
                        confidence += 0.1
                        
            else:
                # Transitional or unclear regime
                if abs(trend_direction) < 0.1:  # Very weak trend
                    regime = MarketRegime.SIDEWAYS
                    confidence = 0.4
                else:
                    regime = MarketRegime.BULL if trend_direction > 0 else MarketRegime.BEAR
                    confidence = 0.3 + ma_separation
            
            # Adjust confidence based on volatility
            volatility_factor = min(current_atr / current_price * 100, 5) / 5  # Normalize to 0-1
            if regime in [MarketRegime.BULL, MarketRegime.BEAR]:
                confidence *= (1 - volatility_factor * 0.2)  # Reduce confidence in high volatility
            elif regime == MarketRegime.BREAKOUT:
                confidence *= (1 + volatility_factor * 0.2)  # Increase confidence in high volatility
            
            confidence = max(0.1, min(0.95, confidence))
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN, 0.1
    
    def _calculate_volatility_metrics(self, price_data: pd.DataFrame) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics.
        
        Args:
            price_data: DataFrame with OHLCV data.
            
        Returns:
            Volatility metrics.
        """
        try:
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Calculate ATR
            atr = self.indicators.atr(high_prices, low_prices, close_prices, self.atr_period)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
            
            # Normalize ATR by price
            atr_normalized = current_atr / close_prices[-1] if close_prices[-1] > 0 else 0
            
            # Calculate volatility for different periods
            volatilities = {}
            for period in self.volatility_periods:
                if len(close_prices) >= period:
                    vol = self.indicators.volatility(close_prices[-period:], period // 4, annualize=True)
                    volatilities[period] = vol[-1] if not np.isnan(vol[-1]) else 0
            
            current_volatility = volatilities.get(self.volatility_periods[0], atr_normalized)
            
            # Calculate volatility percentile (based on recent history)
            if len(close_prices) >= max(self.volatility_periods):
                hist_vol = self.indicators.volatility(close_prices, 20, annualize=True)
                valid_vol = hist_vol[~np.isnan(hist_vol)]
                if len(valid_vol) > 0:
                    volatility_percentile = (np.sum(valid_vol <= current_volatility) / len(valid_vol))
                else:
                    volatility_percentile = 0.5
            else:
                volatility_percentile = 0.5
            
            # Determine volatility trend
            if len(volatilities) >= 2:
                short_vol = volatilities.get(self.volatility_periods[0], 0)
                long_vol = volatilities.get(self.volatility_periods[-1], 0)
                
                if short_vol > long_vol * 1.1:
                    volatility_trend = "increasing"
                elif short_vol < long_vol * 0.9:
                    volatility_trend = "decreasing"
                else:
                    volatility_trend = "stable"
            else:
                volatility_trend = "stable"
            
            # Classify volatility regime
            if volatility_percentile < 0.2:
                volatility_regime = "low"
            elif volatility_percentile < 0.4:
                volatility_regime = "normal"
            elif volatility_percentile < 0.8:
                volatility_regime = "high"
            else:
                volatility_regime = "extreme"
            
            # Simple volatility forecast (exponential smoothing)
            alpha = 0.3
            if len(self.analysis_history) > 0:
                prev_vol = self.analysis_history[-1].volatility_metrics.current_volatility
                volatility_forecast = alpha * current_volatility + (1 - alpha) * prev_vol
            else:
                volatility_forecast = current_volatility
            
            return VolatilityMetrics(
                current_volatility=current_volatility,
                volatility_percentile=volatility_percentile,
                volatility_trend=volatility_trend,
                volatility_regime=volatility_regime,
                atr_normalized=atr_normalized,
                volatility_forecast=volatility_forecast
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return VolatilityMetrics(0, 0.5, "stable", "normal", 0, 0)
    
    def _calculate_trend_metrics(self, price_data: pd.DataFrame) -> TrendMetrics:
        """
        Calculate comprehensive trend metrics.
        
        Args:
            price_data: DataFrame with OHLCV data.
            
        Returns:
            Trend metrics.
        """
        try:
            close_prices = price_data['close'].values
            
            # Calculate moving averages
            ma_fast = self.indicators.ema(close_prices, self.ma_fast)
            ma_slow = self.indicators.ema(close_prices, self.ma_slow)
            
            # Current values
            current_price = close_prices[-1]
            current_ma_fast = ma_fast[-1] if not np.isnan(ma_fast[-1]) else current_price
            current_ma_slow = ma_slow[-1] if not np.isnan(ma_slow[-1]) else current_price
            
            # Calculate trend strength (-1 to 1)
            ma_diff = current_ma_fast - current_ma_slow
            price_range = np.std(close_prices[-20:]) if len(close_prices) >= 20 else np.std(close_prices)
            trend_strength = np.clip(ma_diff / (price_range + 1e-10), -1, 1)
            
            # Determine trend direction
            if trend_strength > 0.1:
                trend_direction = "up"
            elif trend_strength < -0.1:
                trend_direction = "down"
            else:
                trend_direction = "sideways"
            
            # Calculate trend confidence
            # Based on consistency of MA relationship and price position
            price_above_fast = current_price > current_ma_fast
            price_above_slow = current_price > current_ma_slow
            ma_trend_consistent = (trend_strength > 0 and price_above_fast and price_above_slow) or \
                                (trend_strength < 0 and not price_above_fast and not price_above_slow)
            
            trend_confidence = abs(trend_strength) * (1.5 if ma_trend_consistent else 0.8)
            trend_confidence = max(0, min(1, trend_confidence))
            
            # Calculate trend duration
            trend_duration = self._calculate_trend_duration(ma_fast, ma_slow)
            
            # Calculate momentum using ROC (Rate of Change)
            roc_period = min(10, len(close_prices) - 1)
            if roc_period > 0:
                momentum = (current_price - close_prices[-roc_period-1]) / close_prices[-roc_period-1]
                momentum = np.clip(momentum, -0.5, 0.5)  # Limit extreme values
            else:
                momentum = 0
            
            # Calculate reversal probability
            reversal_probability = self._calculate_reversal_probability(
                close_prices, trend_strength, trend_duration, momentum
            )
            
            return TrendMetrics(
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                trend_confidence=trend_confidence,
                trend_duration=trend_duration,
                trend_momentum=momentum,
                reversal_probability=reversal_probability
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return TrendMetrics(0, "sideways", 0.5, 0, 0, 0.5)
    
    def _calculate_trend_duration(self, ma_fast: np.ndarray, ma_slow: np.ndarray) -> int:
        """
        Calculate how long the current trend has been in place.
        
        Args:
            ma_fast: Fast moving average array.
            ma_slow: Slow moving average array.
            
        Returns:
            Trend duration in periods.
        """
        try:
            if len(ma_fast) < 2 or len(ma_slow) < 2:
                return 0
            
            # Find where MAs crossed
            current_trend = ma_fast[-1] > ma_slow[-1]
            duration = 1
            
            for i in range(len(ma_fast) - 2, -1, -1):
                if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
                    break
                
                past_trend = ma_fast[i] > ma_slow[i]
                if past_trend == current_trend:
                    duration += 1
                else:
                    break
            
            return duration
            
        except Exception as e:
            logger.error(f"Error calculating trend duration: {e}")
            return 0
    
    def _calculate_reversal_probability(self, prices: np.ndarray, trend_strength: float, 
                                      trend_duration: int, momentum: float) -> float:
        """
        Calculate probability of trend reversal.
        
        Args:
            prices: Price array.
            trend_strength: Current trend strength.
            trend_duration: How long trend has lasted.
            momentum: Current momentum.
            
        Returns:
            Reversal probability (0-1).
        """
        try:
            # Base probability starts at 0.5
            reversal_prob = 0.5
            
            # Strong trends are less likely to reverse
            reversal_prob -= abs(trend_strength) * 0.3
            
            # Long trends are more likely to reverse (mean reversion)
            if trend_duration > 20:
                reversal_prob += (trend_duration - 20) * 0.01
            
            # Diverging momentum increases reversal probability
            if trend_strength > 0 and momentum < 0:
                reversal_prob += abs(momentum) * 0.5
            elif trend_strength < 0 and momentum > 0:
                reversal_prob += momentum * 0.5
            
            # Check for oversold/overbought conditions
            if len(prices) >= 14:
                rsi = self.indicators.rsi(prices, 14)
                current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
                
                if current_rsi > 80:  # Overbought
                    reversal_prob += (current_rsi - 80) * 0.005
                elif current_rsi < 20:  # Oversold
                    reversal_prob += (20 - current_rsi) * 0.005
            
            return max(0.1, min(0.9, reversal_prob))
            
        except Exception as e:
            logger.error(f"Error calculating reversal probability: {e}")
            return 0.5
    
    def _identify_levels(self, price_data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Identify key support and resistance levels.
        
        Args:
            price_data: DataFrame with OHLCV data.
            
        Returns:
            Tuple of (support_levels, resistance_levels).
        """
        try:
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Use technical indicators to find levels
            support_array, resistance_array = self.indicators.support_resistance_levels(
                high_prices, low_prices, window=20, min_touches=2
            )
            
            # Convert to lists and sort
            support_levels = sorted(support_array.tolist()) if len(support_array) > 0 else []
            resistance_levels = sorted(resistance_array.tolist(), reverse=True) if len(resistance_array) > 0 else []
            
            # Keep only the most relevant levels (closest to current price)
            current_price = price_data.iloc[-1]['close']
            
            # Filter support levels (below current price)
            support_levels = [level for level in support_levels if level < current_price]
            support_levels = sorted(support_levels, reverse=True)[:3]  # Top 3 closest
            
            # Filter resistance levels (above current price)
            resistance_levels = [level for level in resistance_levels if level > current_price]
            resistance_levels = sorted(resistance_levels)[:3]  # Top 3 closest
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {e}")
            return [], []
    
    def _calculate_breakout_probability(self, price_data: pd.DataFrame, 
                                      volatility_metrics: VolatilityMetrics) -> float:
        """
        Calculate probability of price breakout.
        
        Args:
            price_data: DataFrame with OHLCV data.
            volatility_metrics: Volatility analysis.
            
        Returns:
            Breakout probability (0-1).
        """
        try:
            close_prices = price_data['close'].values
            volume = price_data['volume'].values
            
            # Base probability
            breakout_prob = 0.3
            
            # High volatility increases breakout probability
            if volatility_metrics.volatility_regime in ["high", "extreme"]:
                breakout_prob += 0.2
            
            # Increasing volatility increases probability
            if volatility_metrics.volatility_trend == "increasing":
                breakout_prob += 0.15
            
            # Volume analysis
            if len(volume) >= 20:
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                if current_volume > avg_volume * 1.5:  # High volume
                    breakout_prob += 0.2
            
            # Price compression (low range) increases breakout probability
            if len(close_prices) >= 20:
                recent_range = (np.max(close_prices[-20:]) - np.min(close_prices[-20:])) / close_prices[-1]
                if recent_range < 0.05:  # Less than 5% range
                    breakout_prob += 0.25
            
            return max(0.1, min(0.9, breakout_prob))
            
        except Exception as e:
            logger.error(f"Error calculating breakout probability: {e}")
            return 0.3
    
    def _calculate_mean_reversion_probability(self, price_data: pd.DataFrame, 
                                            trend_metrics: TrendMetrics) -> float:
        """
        Calculate probability of mean reversion.
        
        Args:
            price_data: DataFrame with OHLCV data.
            trend_metrics: Trend analysis.
            
        Returns:
            Mean reversion probability (0-1).
        """
        try:
            close_prices = price_data['close'].values
            
            # Base probability (opposite of breakout)
            mean_reversion_prob = 0.7
            
            # Strong trends reduce mean reversion probability
            mean_reversion_prob -= abs(trend_metrics.trend_strength) * 0.3
            
            # Long trends increase mean reversion probability
            if trend_metrics.trend_duration > 30:
                mean_reversion_prob += (trend_metrics.trend_duration - 30) * 0.005
            
            # Check for extreme price levels
            if len(close_prices) >= 50:
                # Z-score of current price vs recent average
                z_scores = self.indicators.z_score(close_prices, 50)
                current_z = z_scores[-1] if not np.isnan(z_scores[-1]) else 0
                
                # Extreme z-scores increase mean reversion probability
                if abs(current_z) > 2:
                    mean_reversion_prob += (abs(current_z) - 2) * 0.1
            
            # High reversal probability increases mean reversion
            mean_reversion_prob += trend_metrics.reversal_probability * 0.2
            
            return max(0.1, min(0.9, mean_reversion_prob))
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion probability: {e}")
            return 0.7
    
    def _update_analysis_history(self, analysis: MarketAnalysis) -> None:
        """
        Update analysis history and track regime transitions.
        
        Args:
            analysis: New market analysis.
        """
        try:
            # Check for regime transition
            if len(self.analysis_history) > 0:
                prev_regime = self.analysis_history[-1].regime
                if prev_regime != analysis.regime:
                    self.regime_transitions.append((
                        analysis.timestamp, prev_regime, analysis.regime
                    ))
                    logger.info(f"Market regime transition: {prev_regime.value} -> {analysis.regime.value}")
            
            # Add to history
            self.analysis_history.append(analysis)
            
            # Keep only recent history (last 1000 analyses)
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-1000:]
            
            # Keep only recent transitions (last 100)
            if len(self.regime_transitions) > 100:
                self.regime_transitions = self.regime_transitions[-100:]
                
        except Exception as e:
            logger.error(f"Error updating analysis history: {e}")
    
    def _create_unknown_analysis(self, current_price: float) -> MarketAnalysis:
        """
        Create unknown/default analysis for error cases.
        
        Args:
            current_price: Current price level.
            
        Returns:
            Default market analysis.
        """
        return MarketAnalysis(
            regime=MarketRegime.UNKNOWN,
            confidence=0.1,
            volatility_metrics=VolatilityMetrics(0, 0.5, "stable", "normal", 0, 0),
            trend_metrics=TrendMetrics(0, "sideways", 0.5, 0, 0, 0.5),
            price_level=current_price,
            support_levels=[],
            resistance_levels=[],
            breakout_probability=0.3,
            mean_reversion_probability=0.7,
            timestamp=datetime.now()
        )
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regime detection performance.
        
        Returns:
            Statistics dictionary.
        """
        try:
            if not self.analysis_history:
                return {}
            
            # Count regime occurrences
            regime_counts = {}
            confidence_by_regime = {}
            
            for analysis in self.analysis_history:
                regime = analysis.regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                if regime not in confidence_by_regime:
                    confidence_by_regime[regime] = []
                confidence_by_regime[regime].append(analysis.confidence)
            
            # Calculate average confidence by regime
            avg_confidence = {}
            for regime, confidences in confidence_by_regime.items():
                avg_confidence[regime] = np.mean(confidences)
            
            # Transition statistics
            transition_counts = {}
            for _, from_regime, to_regime in self.regime_transitions:
                transition = f"{from_regime.value}->{to_regime.value}"
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
            
            return {
                'total_analyses': len(self.analysis_history),
                'regime_distribution': regime_counts,
                'average_confidence_by_regime': avg_confidence,
                'total_transitions': len(self.regime_transitions),
                'transition_counts': transition_counts,
                'most_recent_regime': self.analysis_history[-1].regime.value,
                'current_confidence': self.analysis_history[-1].confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return {}
    
    def is_regime_stable(self, lookback_periods: int = 10) -> bool:
        """
        Check if current regime has been stable recently.
        
        Args:
            lookback_periods: Number of recent periods to check.
            
        Returns:
            Whether regime has been stable.
        """
        try:
            if len(self.analysis_history) < lookback_periods:
                return False
            
            recent_analyses = self.analysis_history[-lookback_periods:]
            current_regime = recent_analyses[-1].regime
            
            # Check if all recent analyses show same regime
            stable = all(analysis.regime == current_regime for analysis in recent_analyses)
            
            # Also check confidence levels
            if stable:
                avg_confidence = np.mean([analysis.confidence for analysis in recent_analyses])
                stable = stable and avg_confidence > 0.6
            
            return stable
            
        except Exception as e:
            logger.error(f"Error checking regime stability: {e}")
            return False