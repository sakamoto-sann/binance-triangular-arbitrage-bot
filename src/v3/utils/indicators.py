"""
Grid Trading Bot v3.0 - Technical Indicators
Complete implementation of technical indicators for market analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union
# from numba import jit  # Commented out for compatibility
import warnings

logger = logging.getLogger(__name__)

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', module='numba')

class TechnicalIndicators:
    """Complete technical indicators implementation with optimized calculations."""
    
    @staticmethod
    # @jit(nopython=True)  # Commented out for compatibility
    def _ema_calculation(values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Optimized EMA calculation using numba JIT compilation.
        
        Args:
            values: Input values array.
            alpha: Smoothing factor.
            
        Returns:
            EMA values array.
        """
        result = np.empty_like(values)
        result[0] = values[0]
        
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        
        return result
    
    @staticmethod
    # @jit(nopython=True)  # Commented out for compatibility
    def _sma_calculation(values: np.ndarray, window: int) -> np.ndarray:
        """
        Optimized SMA calculation using numba JIT compilation.
        
        Args:
            values: Input values array.
            window: Moving average window.
            
        Returns:
            SMA values array.
        """
        result = np.full_like(values, np.nan)
        
        for i in range(window - 1, len(values)):
            result[i] = np.mean(values[i-window+1:i+1])
        
        return result
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
        """
        Simple Moving Average.
        
        Args:
            data: Input data.
            window: Moving average window.
            
        Returns:
            SMA values.
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        if len(values) < window:
            logger.warning(f"Data length {len(values)} is less than window {window}")
            return np.full_like(values, np.nan)
        
        return TechnicalIndicators._sma_calculation(values, window)
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
        """
        Exponential Moving Average.
        
        Args:
            data: Input data.
            window: EMA window.
            
        Returns:
            EMA values.
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        if len(values) == 0:
            return np.array([])
        
        alpha = 2.0 / (window + 1)
        return TechnicalIndicators._ema_calculation(values, alpha)
    
    @staticmethod
    def atr(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            window: int = 14) -> np.ndarray:
        """
        Average True Range.
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: ATR window.
            
        Returns:
            ATR values.
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if len(high) < 2:
            return np.full_like(high, np.nan)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # Set first value to high - low (no previous close)
        tr2[0] = 0
        tr3[0] = 0
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using EMA
        return TechnicalIndicators.ema(true_range, window)
    
    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], window: int = 14) -> np.ndarray:
        """
        Relative Strength Index.
        
        Args:
            data: Input data (typically close prices).
            window: RSI window.
            
        Returns:
            RSI values.
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        if len(values) < window + 1:
            return np.full_like(values, np.nan)
        
        # Calculate price changes
        delta = np.diff(values)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gains = TechnicalIndicators.ema(gains, window)
        avg_losses = TechnicalIndicators.ema(losses, window)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Add small value to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN to match original array length
        return np.concatenate([np.array([np.nan]), rsi])
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                       window: int = 20, 
                       num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands.
        
        Args:
            data: Input data.
            window: Moving average window.
            num_std: Number of standard deviations.
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band).
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        # Calculate middle band (SMA)
        middle_band = TechnicalIndicators.sma(values, window)
        
        # Calculate standard deviation
        std_dev = np.full_like(values, np.nan)
        for i in range(window - 1, len(values)):
            std_dev[i] = np.std(values[i-window+1:i+1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], 
             fast_window: int = 12, 
             slow_window: int = 26, 
             signal_window: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Input data.
            fast_window: Fast EMA window.
            slow_window: Slow EMA window.
            signal_window: Signal line EMA window.
            
        Returns:
            Tuple of (macd_line, signal_line, histogram).
        """
        # Calculate EMAs
        fast_ema = TechnicalIndicators.ema(data, fast_window)
        slow_ema = TechnicalIndicators.ema(data, slow_window)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal_window)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray], 
                  k_window: int = 14, 
                  d_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            k_window: %K window.
            d_window: %D window.
            
        Returns:
            Tuple of (%K, %D).
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        # Calculate %K
        k_percent = np.full_like(close, np.nan)
        
        for i in range(k_window - 1, len(close)):
            lowest_low = np.min(low[i-k_window+1:i+1])
            highest_high = np.max(high[i-k_window+1:i+1])
            
            if highest_high != lowest_low:
                k_percent[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_percent[i] = 50  # Neutral value when range is zero
        
        # Calculate %D (SMA of %K)
        d_percent = TechnicalIndicators.sma(k_percent, d_window)
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray], 
                  window: int = 14) -> np.ndarray:
        """
        Williams %R.
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: Williams %R window.
            
        Returns:
            Williams %R values.
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        williams_r = np.full_like(close, np.nan)
        
        for i in range(window - 1, len(close)):
            highest_high = np.max(high[i-window+1:i+1])
            lowest_low = np.min(low[i-window+1:i+1])
            
            if highest_high != lowest_low:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                williams_r[i] = -50  # Neutral value when range is zero
        
        return williams_r
    
    @staticmethod
    def cci(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            window: int = 20) -> np.ndarray:
        """
        Commodity Channel Index.
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: CCI window.
            
        Returns:
            CCI values.
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate SMA of typical price
        sma_tp = TechnicalIndicators.sma(typical_price, window)
        
        # Calculate mean deviation
        mean_deviation = np.full_like(typical_price, np.nan)
        
        for i in range(window - 1, len(typical_price)):
            tp_window = typical_price[i-window+1:i+1]
            mean_deviation[i] = np.mean(np.abs(tp_window - sma_tp[i]))
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def adx(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            window: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average Directional Index.
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: ADX window.
            
        Returns:
            Tuple of (adx, plus_di, minus_di).
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if len(high) < 2:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # Calculate True Range and Directional Movement
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        # Set DM to zero when the other DM is larger
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
        
        # Calculate smoothed TR and DM
        tr_smooth = TechnicalIndicators.ema(tr, window)
        plus_dm_smooth = TechnicalIndicators.ema(plus_dm, window)
        minus_dm_smooth = TechnicalIndicators.ema(minus_dm, window)
        
        # Calculate DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # Calculate ADX
        adx = TechnicalIndicators.ema(dx, window)
        
        # Prepend NaN to match original array length
        adx = np.concatenate([np.array([np.nan]), adx])
        plus_di = np.concatenate([np.array([np.nan]), plus_di])
        minus_di = np.concatenate([np.array([np.nan]), minus_di])
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def obv(close: Union[pd.Series, np.ndarray], 
            volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        On-Balance Volume.
        
        Args:
            close: Close prices.
            volume: Volume data.
            
        Returns:
            OBV values.
        """
        # Convert to numpy arrays
        if isinstance(close, pd.Series):
            close = close.values
        if isinstance(volume, pd.Series):
            volume = volume.values
        
        if len(close) < 2:
            return np.full_like(close, np.nan)
        
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """
        Calculate pivot points and support/resistance levels.
        
        Args:
            high: Previous period high.
            low: Previous period low.
            close: Previous period close.
            
        Returns:
            Dictionary with pivot points and levels.
        """
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def volatility(data: Union[pd.Series, np.ndarray], 
                  window: int = 20, 
                  annualize: bool = True) -> np.ndarray:
        """
        Calculate rolling volatility.
        
        Args:
            data: Input data (typically returns).
            window: Rolling window.
            annualize: Whether to annualize volatility.
            
        Returns:
            Volatility values.
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        # Calculate returns if input is prices
        if np.all(values > 0):  # Likely prices
            returns = np.diff(np.log(values))
        else:  # Likely returns
            returns = values[1:]  # Skip first NaN from diff
        
        # Calculate rolling standard deviation
        vol = np.full(len(values), np.nan)
        
        for i in range(window - 1, len(returns)):
            vol[i + 1] = np.std(returns[i-window+1:i+1])
        
        # Annualize if requested (assuming daily data)
        if annualize:
            vol = vol * np.sqrt(365)
        
        return vol
    
    @staticmethod
    def correlation(x: Union[pd.Series, np.ndarray], 
                   y: Union[pd.Series, np.ndarray], 
                   window: int = 20) -> np.ndarray:
        """
        Calculate rolling correlation.
        
        Args:
            x: First series.
            y: Second series.
            window: Rolling window.
            
        Returns:
            Correlation values.
        """
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if len(x) != len(y):
            raise ValueError("Series must have same length")
        
        corr = np.full_like(x, np.nan)
        
        for i in range(window - 1, len(x)):
            x_window = x[i-window+1:i+1]
            y_window = y[i-window+1:i+1]
            
            # Calculate correlation coefficient
            corr_matrix = np.corrcoef(x_window, y_window)
            corr[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
        
        return corr
    
    @staticmethod
    def z_score(data: Union[pd.Series, np.ndarray], 
                window: int = 20) -> np.ndarray:
        """
        Calculate rolling z-score.
        
        Args:
            data: Input data.
            window: Rolling window.
            
        Returns:
            Z-score values.
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
        
        z_scores = np.full_like(values, np.nan)
        
        for i in range(window - 1, len(values)):
            window_data = values[i-window+1:i+1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                z_scores[i] = (values[i] - mean) / std
            else:
                z_scores[i] = 0
        
        return z_scores
    
    @staticmethod
    def support_resistance_levels(high: Union[pd.Series, np.ndarray], 
                                 low: Union[pd.Series, np.ndarray], 
                                 window: int = 20, 
                                 min_touches: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify support and resistance levels.
        
        Args:
            high: High prices.
            low: Low prices.
            window: Lookback window.
            min_touches: Minimum touches to confirm level.
            
        Returns:
            Tuple of (support_levels, resistance_levels).
        """
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(high) - window):
            # Check for resistance (local high)
            if high[i] == np.max(high[i-window:i+window+1]):
                # Count touches within tolerance
                level = high[i]
                tolerance = level * 0.01  # 1% tolerance
                touches = np.sum(np.abs(high[max(0, i-100):i+100] - level) <= tolerance)
                
                if touches >= min_touches:
                    resistance_levels.append(level)
            
            # Check for support (local low)
            if low[i] == np.min(low[i-window:i+window+1]):
                # Count touches within tolerance
                level = low[i]
                tolerance = level * 0.01  # 1% tolerance
                touches = np.sum(np.abs(low[max(0, i-100):i+100] - level) <= tolerance)
                
                if touches >= min_touches:
                    support_levels.append(level)
        
        return np.array(support_levels), np.array(resistance_levels)