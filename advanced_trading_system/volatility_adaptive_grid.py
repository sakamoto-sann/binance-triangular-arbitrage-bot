"""
Professional-Grade Volatility Adaptive Grid Management System
Implements BitVol, LXVX, GARCH, and Parkinson volatility indicators for institutional trading
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("ARCH package not available. GARCH models will use fallback implementation.")

@dataclass
class VolatilityMetrics:
    """Data class for volatility metrics"""
    bitvol: float
    lxvx: float
    garch_vol: float
    realized_vol: float
    parkinson_vol: float
    composite_vol: float
    regime: str
    confidence: float

@dataclass
class GridParameters:
    """Data class for grid parameters"""
    spacing: float
    density_multiplier: float
    max_levels: int
    regime: str
    composite_volatility: float
    stress_factor: float
    liquidity_factor: float

class VolatilityAdaptiveGrid:
    """
    Professional-grade volatility adaptive grid management system.
    Integrates multiple institutional volatility indicators for optimal grid spacing.
    """
    
    def __init__(self, binance_client=None, config: Dict = None):
        """Initialize the volatility adaptive grid system"""
        self.binance_client = binance_client
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Traditional volatility measures
        self.atr_period = self.config.get('atr_period', 14)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 2.0)
        
        # Professional volatility indicator weights
        self.bitvol_weight = self.config.get('bitvol_weight', 0.3)
        self.lxvx_weight = self.config.get('lxvx_weight', 0.25)
        self.garch_weight = self.config.get('garch_weight', 0.25)
        self.realized_vol_weight = self.config.get('realized_vol_weight', 0.2)
        
        # Grid parameters
        self.min_grid_spacing = self.config.get('min_grid_spacing', 0.0005)  # 0.05%
        self.max_grid_spacing = self.config.get('max_grid_spacing', 0.02)    # 2.0%
        
        # Volatility regime thresholds (annualized)
        self.volatility_regimes = {
            'ultra_low': 0.15,    # <15% annualized
            'low': 0.25,          # 15-25% annualized
            'normal': 0.50,       # 25-50% annualized
            'high': 0.75,         # 50-75% annualized
            'extreme': 1.0        # >75% annualized
        }
        
        # Grid density adjustments per regime
        self.regime_grid_params = {
            'ultra_low': {'spacing': 0.0005, 'density': 1.5, 'levels': 50},
            'low': {'spacing': 0.001, 'density': 1.2, 'levels': 40},
            'normal': {'spacing': 0.002, 'density': 1.0, 'levels': 30},
            'high': {'spacing': 0.005, 'density': 0.8, 'levels': 20},
            'extreme': {'spacing': 0.01, 'density': 0.5, 'levels': 15}
        }
        
        # Cache for volatility data
        self._vol_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Market cap weights for LXVX calculation
        self.lxvx_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.market_cap_weights = {
            'BTCUSDT': 0.4,
            'ETHUSDT': 0.25,
            'BNBUSDT': 0.15,
            'ADAUSDT': 0.1,
            'SOLUSDT': 0.1
        }

    async def fetch_bitvol_data(self) -> float:
        """
        Fetch BitVol (Bitcoin Volatility Index) from Deribit options data.
        BitVol represents market's expectation of 30-day volatility.
        """
        try:
            # Check cache first
            cache_key = 'bitvol'
            if self._is_cache_valid(cache_key):
                return self._vol_cache[cache_key]['value']
            
            # Fetch from Deribit API
            url = "https://www.deribit.com/api/v2/public/get_index"
            params = {"currency": "BTC"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract BitVol if available
                        for index in data.get('result', {}).get('BTC', []):
                            if 'VOL' in index:
                                bitvol_value = float(index['price']) / 100
                                self._update_cache(cache_key, bitvol_value)
                                return bitvol_value
            
            # Fallback: Calculate implied volatility from options chain
            bitvol_value = await self._calculate_implied_volatility_index('BTC')
            self._update_cache(cache_key, bitvol_value)
            return bitvol_value
            
        except Exception as e:
            self.logger.warning(f"BitVol fetch failed: {e}. Using fallback calculation.")
            # Fallback to historical volatility
            return await self._calculate_historical_volatility('BTCUSDT', period=30)

    async def _calculate_implied_volatility_index(self, currency: str = 'BTC') -> float:
        """Calculate implied volatility index from options chain"""
        try:
            url = "https://www.deribit.com/api/v2/public/get_instruments"
            params = {"currency": currency, "kind": "option", "expired": False}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        instruments = data.get('result', [])
                        
                        # Filter for near-term ATM options (30 days)
                        atm_options = []
                        current_price = await self._get_current_price(f'{currency}USDT')
                        
                        for instrument in instruments:
                            strike = instrument.get('strike')
                            expiration = instrument.get('expiration_timestamp')
                            
                            if strike and expiration:
                                days_to_expiry = (expiration - datetime.now().timestamp() * 1000) / (1000 * 86400)
                                moneyness = abs(strike - current_price) / current_price
                                
                                if 25 <= days_to_expiry <= 35 and moneyness <= 0.05:  # Near ATM, ~30 days
                                    atm_options.append(instrument)
                        
                        # Calculate average implied volatility
                        if atm_options:
                            iv_sum = 0
                            count = 0
                            
                            for option in atm_options:
                                # Get option market data
                                ticker_url = "https://www.deribit.com/api/v2/public/ticker"
                                ticker_params = {"instrument_name": option['instrument_name']}
                                
                                async with session.get(ticker_url, params=ticker_params) as ticker_response:
                                    if ticker_response.status == 200:
                                        ticker_data = await ticker_response.json()
                                        result = ticker_data.get('result', {})
                                        mark_iv = result.get('mark_iv')
                                        
                                        if mark_iv:
                                            iv_sum += mark_iv
                                            count += 1
                            
                            if count > 0:
                                return iv_sum / count / 100  # Convert percentage to decimal
            
            # Fallback if options data unavailable
            return await self._calculate_historical_volatility('BTCUSDT', period=30)
            
        except Exception as e:
            self.logger.error(f"Implied volatility calculation failed: {e}")
            return await self._calculate_historical_volatility('BTCUSDT', period=30)

    async def fetch_lxvx_data(self) -> float:
        """
        Fetch LXVX (Liquid Index Volatility) equivalent.
        Creates equivalent using basket of top crypto volatilities weighted by market cap.
        """
        try:
            cache_key = 'lxvx'
            if self._is_cache_valid(cache_key):
                return self._vol_cache[cache_key]['value']
            
            volatilities = []
            
            # Calculate volatility for each symbol in the basket
            for symbol in self.lxvx_symbols:
                try:
                    vol = await self._calculate_realized_volatility(symbol, period=30)
                    weight = self.market_cap_weights.get(symbol, 0.1)
                    volatilities.append(vol * weight)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate volatility for {symbol}: {e}")
                    continue
            
            # Calculate weighted average volatility
            if volatilities:
                lxvx_equivalent = sum(volatilities)
                self._update_cache(cache_key, lxvx_equivalent)
                return lxvx_equivalent
            else:
                # Fallback to BTC volatility
                fallback_vol = await self._calculate_realized_volatility('BTCUSDT', period=30)
                self._update_cache(cache_key, fallback_vol)
                return fallback_vol
                
        except Exception as e:
            self.logger.error(f"LXVX calculation failed: {e}")
            # Fallback to BTC volatility
            return await self._calculate_realized_volatility('BTCUSDT', period=30)

    def calculate_garch_volatility(self, price_returns: np.ndarray, lookback: int = 252) -> float:
        """
        Calculate GARCH(1,1) volatility forecast.
        GARCH model captures volatility clustering popular in institutional risk management.
        """
        try:
            if not ARCH_AVAILABLE:
                return self._fallback_garch_calculation(price_returns)
            
            # Ensure we have enough data
            if len(price_returns) < 50:
                return np.std(price_returns) * np.sqrt(365)
            
            # Convert to percentage returns for ARCH model
            returns_pct = price_returns * 100
            
            # Remove any infinite or NaN values
            returns_pct = returns_pct[np.isfinite(returns_pct)]
            
            if len(returns_pct) < 50:
                return np.std(price_returns) * np.sqrt(365)
            
            # Fit GARCH(1,1) model
            model = arch_model(returns_pct, vol='Garch', p=1, q=1, rescale=False)
            
            # Fit with error handling
            try:
                fitted_model = model.fit(disp='off', show_warning=False)
                
                # Forecast next period volatility
                forecast = fitted_model.forecast(horizon=1)
                forecasted_variance = forecast.variance.iloc[-1, 0]
                
                # Convert to daily volatility (annualized)
                if forecasted_variance > 0:
                    daily_vol = np.sqrt(forecasted_variance / 100)
                    annualized_vol = daily_vol * np.sqrt(365)
                    
                    # Sanity check - cap volatility at reasonable levels
                    return min(max(annualized_vol, 0.01), 5.0)
                else:
                    return np.std(price_returns) * np.sqrt(365)
                    
            except Exception as fit_error:
                self.logger.warning(f"GARCH model fitting failed: {fit_error}")
                return self._fallback_garch_calculation(price_returns)
                
        except Exception as e:
            self.logger.error(f"GARCH volatility calculation failed: {e}")
            return self._fallback_garch_calculation(price_returns)

    def _fallback_garch_calculation(self, price_returns: np.ndarray) -> float:
        """Fallback GARCH calculation using exponential smoothing"""
        try:
            # Use exponentially weighted moving average as GARCH substitute
            returns_squared = price_returns ** 2
            alpha = 0.06  # GARCH alpha parameter approximation
            beta = 0.93   # GARCH beta parameter approximation
            
            # Initialize with sample variance
            var_estimate = np.var(price_returns)
            
            # Update variance estimate using GARCH-like formula
            for r_squared in returns_squared[-min(100, len(returns_squared)):]:
                var_estimate = alpha * r_squared + beta * var_estimate
            
            # Convert to annualized volatility
            return min(max(np.sqrt(var_estimate * 365), 0.01), 5.0)
            
        except Exception:
            return np.std(price_returns) * np.sqrt(365)

    async def _calculate_realized_volatility(self, symbol: str, period: int = 30) -> float:
        """
        Calculate realized volatility using high-frequency data with Parkinson estimator.
        Uses minute-by-minute data for accurate volatility calculation.
        """
        try:
            cache_key = f'realized_vol_{symbol}_{period}'
            if self._is_cache_valid(cache_key):
                return self._vol_cache[cache_key]['value']
            
            # Fetch high-frequency price data (1-minute intervals)
            price_data = await self._get_price_data(symbol, interval='1m', limit=period * 1440)
            
            if price_data is None or len(price_data) < 100:
                self.logger.warning(f"Insufficient data for {symbol}, using daily data")
                price_data = await self._get_price_data(symbol, interval='1d', limit=period)
                if price_data is None:
                    return 0.3  # Default fallback volatility
            
            # Calculate Parkinson volatility estimator for better accuracy
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Parkinson estimator: more efficient than close-to-close returns
            log_hl_ratios = np.log(high_prices / low_prices) ** 2
            parkinson_var = np.sum(log_hl_ratios) / (4 * np.log(2) * len(log_hl_ratios))
            parkinson_vol = np.sqrt(parkinson_var)
            
            # Also calculate close-to-close returns for comparison
            close_prices = price_data['close'].values
            log_returns = np.log(close_prices[1:] / close_prices[:-1])
            close_vol = np.std(log_returns)
            
            # Use the more stable estimate
            if len(log_returns) > 50:
                # Weight Parkinson estimator more heavily for high-frequency data
                realized_vol = 0.7 * parkinson_vol + 0.3 * close_vol
            else:
                realized_vol = close_vol
            
            # Annualize the volatility based on data frequency
            if price_data.index.freq and 'min' in str(price_data.index.freq):
                # Minute data
                annualized_vol = realized_vol * np.sqrt(365 * 1440)
            else:
                # Daily data
                annualized_vol = realized_vol * np.sqrt(365)
            
            # Sanity check and cache result
            final_vol = min(max(annualized_vol, 0.01), 5.0)
            self._update_cache(cache_key, final_vol)
            return final_vol
            
        except Exception as e:
            self.logger.error(f"Realized volatility calculation failed for {symbol}: {e}")
            return 0.3  # Default fallback volatility

    async def _calculate_historical_volatility(self, symbol: str, period: int = 30) -> float:
        """Calculate simple historical volatility as fallback"""
        try:
            price_data = await self._get_price_data(symbol, interval='1d', limit=period)
            if price_data is None or len(price_data) < 10:
                return 0.3  # Default fallback
            
            close_prices = price_data['close'].values
            returns = np.log(close_prices[1:] / close_prices[:-1])
            
            # Annualized standard deviation
            return np.std(returns) * np.sqrt(365)
            
        except Exception:
            return 0.3  # Default fallback

    async def calculate_composite_volatility(self, symbol: str) -> VolatilityMetrics:
        """
        Calculate weighted composite volatility using all professional indicators.
        Returns comprehensive volatility metrics with confidence scoring.
        """
        try:
            # Fetch all volatility measures
            bitvol = await self.fetch_bitvol_data()
            lxvx = await self.fetch_lxvx_data()
            
            # Get price returns for GARCH calculation
            price_data = await self._get_price_data(symbol, interval='1h', limit=168)
            if price_data is not None and len(price_data) > 50:
                close_prices = price_data['close'].values
                returns = np.log(close_prices[1:] / close_prices[:-1])
                garch_vol = self.calculate_garch_volatility(returns)
            else:
                garch_vol = await self._calculate_historical_volatility(symbol)
            
            realized_vol = await self._calculate_realized_volatility(symbol)
            
            # Calculate Parkinson volatility separately for comparison
            parkinson_vol = await self._calculate_parkinson_volatility(symbol)
            
            # Calculate weighted composite
            composite_volatility = (
                bitvol * self.bitvol_weight +
                lxvx * self.lxvx_weight +
                garch_vol * self.garch_weight +
                realized_vol * self.realized_vol_weight
            )
            
            # Determine volatility regime
            regime = self._determine_volatility_regime(composite_volatility)
            
            # Calculate confidence score based on consistency of measures
            vol_measures = [bitvol, lxvx, garch_vol, realized_vol]
            vol_std = np.std(vol_measures)
            vol_mean = np.mean(vol_measures)
            confidence = max(0.1, 1.0 - (vol_std / vol_mean)) if vol_mean > 0 else 0.5
            
            return VolatilityMetrics(
                bitvol=bitvol,
                lxvx=lxvx,
                garch_vol=garch_vol,
                realized_vol=realized_vol,
                parkinson_vol=parkinson_vol,
                composite_vol=composite_volatility,
                regime=regime,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Composite volatility calculation failed: {e}")
            # Return fallback metrics
            fallback_vol = 0.3
            return VolatilityMetrics(
                bitvol=fallback_vol,
                lxvx=fallback_vol,
                garch_vol=fallback_vol,
                realized_vol=fallback_vol,
                parkinson_vol=fallback_vol,
                composite_vol=fallback_vol,
                regime='normal',
                confidence=0.3
            )

    async def _calculate_parkinson_volatility(self, symbol: str, period: int = 30) -> float:
        """Calculate Parkinson volatility estimator specifically"""
        try:
            price_data = await self._get_price_data(symbol, interval='1d', limit=period)
            if price_data is None or len(price_data) < 10:
                return 0.3
            
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            log_hl_ratios = np.log(high_prices / low_prices) ** 2
            parkinson_var = np.sum(log_hl_ratios) / (4 * np.log(2) * len(log_hl_ratios))
            
            return min(max(np.sqrt(parkinson_var * 365), 0.01), 5.0)
            
        except Exception:
            return 0.3

    def _determine_volatility_regime(self, composite_volatility: float) -> str:
        """Determine current volatility regime based on composite volatility"""
        for regime, threshold in self.volatility_regimes.items():
            if composite_volatility <= threshold:
                return regime
        return 'extreme'

    async def calculate_dynamic_spacing(self, symbol: str, current_price: float) -> GridParameters:
        """
        Calculate optimal grid spacing based on professional volatility analysis.
        Returns comprehensive grid parameters optimized for current market conditions.
        """
        try:
            # Get comprehensive volatility metrics
            vol_metrics = await self.calculate_composite_volatility(symbol)
            
            # Get regime-specific parameters
            regime_params = self.regime_grid_params[vol_metrics.regime]
            
            # Calculate adaptive spacing
            base_spacing = regime_params['spacing']
            density_multiplier = regime_params['density']
            max_levels = regime_params['levels']
            
            # Fine-tune based on current market conditions
            stress_factor = await self._calculate_market_stress_factor(symbol)
            liquidity_factor = await self._calculate_liquidity_factor(symbol)
            
            # Volatility adjustment factor
            vol_adjustment = min(2.0, vol_metrics.composite_vol / 0.3)  # Normalize to 30% baseline
            
            # Final spacing calculation with all factors
            final_spacing = (
                base_spacing * 
                stress_factor * 
                liquidity_factor * 
                vol_adjustment
            )
            
            # Apply bounds
            final_spacing = max(self.min_grid_spacing, min(final_spacing, self.max_grid_spacing))
            
            # Adjust density based on confidence
            confidence_adjusted_density = density_multiplier * vol_metrics.confidence
            
            # Adjust levels based on volatility
            volatility_adjusted_levels = int(max_levels * (1.0 / vol_adjustment))
            volatility_adjusted_levels = max(10, min(volatility_adjusted_levels, 100))
            
            return GridParameters(
                spacing=final_spacing,
                density_multiplier=confidence_adjusted_density,
                max_levels=volatility_adjusted_levels,
                regime=vol_metrics.regime,
                composite_volatility=vol_metrics.composite_vol,
                stress_factor=stress_factor,
                liquidity_factor=liquidity_factor
            )
            
        except Exception as e:
            self.logger.error(f"Dynamic spacing calculation failed: {e}")
            # Return safe default parameters
            return GridParameters(
                spacing=0.002,  # 0.2% default
                density_multiplier=1.0,
                max_levels=30,
                regime='normal',
                composite_volatility=0.3,
                stress_factor=1.0,
                liquidity_factor=1.0
            )

    async def _calculate_market_stress_factor(self, symbol: str) -> float:
        """
        Calculate market stress factor using multiple indicators.
        Returns multiplier for grid spacing (>1.0 means wider spacing needed).
        """
        try:
            stress_components = []
            
            # 1. Funding rate stress
            funding_stress = await self._calculate_funding_rate_stress(symbol)
            stress_components.append(funding_stress)
            
            # 2. Correlation stress (breakdown in normal correlations)
            correlation_stress = await self._calculate_correlation_stress(symbol)
            stress_components.append(correlation_stress)
            
            # 3. Liquidity stress (order book depth reduction)
            liquidity_stress = await self._calculate_liquidity_stress(symbol)
            stress_components.append(liquidity_stress)
            
            # 4. Volume stress (unusual volume patterns)
            volume_stress = await self._calculate_volume_stress(symbol)
            stress_components.append(volume_stress)
            
            # Calculate composite stress factor
            if stress_components:
                avg_stress = np.mean(stress_components)
                # Stress factor ranges from 1.0 (no stress) to 2.0 (maximum stress)
                stress_factor = 1.0 + min(avg_stress, 1.0)
                return stress_factor
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Market stress calculation failed: {e}")
            return 1.0  # Default to no stress adjustment

    async def _calculate_funding_rate_stress(self, symbol: str) -> float:
        """Calculate stress factor based on funding rate extremes"""
        try:
            if self.binance_client:
                # Get current funding rate
                funding_info = self.binance_client.futures_funding_rate(symbol=symbol, limit=24)
                if funding_info:
                    current_funding = float(funding_info[-1]['fundingRate'])
                    
                    # Calculate stress based on absolute funding rate
                    # Normal funding rates are typically ±0.01% to ±0.1%
                    abs_funding = abs(current_funding)
                    
                    if abs_funding > 0.005:  # >0.5% funding rate
                        return min(1.0, abs_funding * 100)  # Cap at 100% stress
                    else:
                        return abs_funding * 20  # Scale lower rates
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _calculate_correlation_stress(self, symbol: str) -> float:
        """Calculate stress based on correlation breakdown"""
        try:
            # Get price data for correlation calculation
            symbols_for_correlation = ['BTCUSDT', 'ETHUSDT']
            if symbol not in symbols_for_correlation:
                symbols_for_correlation.append(symbol)
            
            correlations = []
            base_symbol = 'BTCUSDT'  # Use BTC as base
            
            for corr_symbol in symbols_for_correlation:
                if corr_symbol != base_symbol:
                    corr_coef = await self._calculate_correlation(base_symbol, corr_symbol)
                    if corr_coef is not None:
                        correlations.append(abs(corr_coef))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                # Normal crypto correlations are typically 0.6-0.8
                # Stress occurs when correlations break down (become very low)
                if avg_correlation < 0.3:
                    return (0.3 - avg_correlation) * 2  # Stress factor
                else:
                    return 0.0
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _calculate_liquidity_stress(self, symbol: str) -> float:
        """Calculate stress based on order book depth reduction"""
        try:
            if self.binance_client:
                # Get order book depth
                order_book = self.binance_client.get_order_book(symbol=symbol, limit=20)
                
                # Calculate bid-ask spread
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                spread = (best_ask - best_bid) / best_bid
                
                # Calculate total depth
                bid_depth = sum(float(level[1]) for level in order_book['bids'][:10])
                ask_depth = sum(float(level[1]) for level in order_book['asks'][:10])
                total_depth = bid_depth + ask_depth
                
                # Stress increases with wider spreads and lower depth
                spread_stress = min(1.0, spread * 1000)  # Scale spread stress
                
                # Depth stress (assumes normal depth levels)
                # This would need calibration based on typical depths for each symbol
                depth_stress = 0.0  # Placeholder - would need historical depth data
                
                return (spread_stress + depth_stress) / 2
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _calculate_volume_stress(self, symbol: str) -> float:
        """Calculate stress based on unusual volume patterns"""
        try:
            # Get recent volume data
            price_data = await self._get_price_data(symbol, interval='1h', limit=24)
            if price_data is None or len(price_data) < 12:
                return 0.0
            
            volumes = price_data['volume'].values
            current_volume = volumes[-1]
            
            # Calculate volume statistics
            mean_volume = np.mean(volumes[:-1])
            volume_std = np.std(volumes[:-1])
            
            if volume_std > 0:
                # Z-score of current volume
                volume_zscore = abs(current_volume - mean_volume) / volume_std
                
                # Stress factor based on how unusual current volume is
                return min(1.0, volume_zscore / 5.0)  # Cap at 100% stress
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _calculate_liquidity_factor(self, symbol: str) -> float:
        """
        Calculate liquidity factor for grid spacing adjustment.
        Returns multiplier where <1.0 means tighter spacing (high liquidity).
        """
        try:
            if self.binance_client:
                # Get 24hr ticker statistics
                ticker = self.binance_client.get_ticker(symbol=symbol)
                volume_24h = float(ticker['volume'])
                
                # Get order book for depth analysis
                order_book = self.binance_client.get_order_book(symbol=symbol, limit=20)
                
                # Calculate bid-ask spread
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                spread_pct = (best_ask - best_bid) / best_bid
                
                # Calculate depth score
                bid_depth = sum(float(level[1]) for level in order_book['bids'][:5])
                ask_depth = sum(float(level[1]) for level in order_book['asks'][:5])
                avg_depth = (bid_depth + ask_depth) / 2
                
                # Liquidity score based on volume and spread
                # High volume and low spread = high liquidity = factor < 1.0
                volume_score = min(1.0, volume_24h / 1000000)  # Normalize by 1M volume
                spread_score = max(0.1, min(1.0, spread_pct * 1000))  # Spread penalty
                
                # Combine factors
                liquidity_factor = spread_score / volume_score if volume_score > 0 else 1.0
                
                return max(0.5, min(2.0, liquidity_factor))  # Bound between 0.5 and 2.0
            
            return 1.0  # Default neutral factor
            
        except Exception:
            return 1.0

    async def _calculate_correlation(self, symbol1: str, symbol2: str, period: int = 30) -> Optional[float]:
        """Calculate correlation between two symbols"""
        try:
            # Get price data for both symbols
            data1 = await self._get_price_data(symbol1, interval='1h', limit=period * 24)
            data2 = await self._get_price_data(symbol2, interval='1h', limit=period * 24)
            
            if data1 is None or data2 is None:
                return None
            
            # Calculate returns
            returns1 = np.log(data1['close'].values[1:] / data1['close'].values[:-1])
            returns2 = np.log(data2['close'].values[1:] / data2['close'].values[:-1])
            
            # Ensure same length
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]
            
            if min_len < 10:
                return None
            
            # Calculate correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            return correlation if not np.isnan(correlation) else None
            
        except Exception:
            return None

    def adjust_grid_density(self, price_level: float, support_resistance_levels: List[float], 
                          volatility_params: GridParameters) -> float:
        """
        Adjust grid density based on technical levels and volatility.
        Returns adjusted density multiplier.
        """
        try:
            density_multiplier = volatility_params.density_multiplier
            
            # Increase density near support/resistance levels
            for level in support_resistance_levels:
                distance = abs(price_level - level) / level
                if distance < 0.01:  # Within 1% of key level
                    density_multiplier *= 1.5
                elif distance < 0.02:  # Within 2% of key level
                    density_multiplier *= 1.2
            
            # Adjust for round number psychology
            if self._is_round_number(price_level):
                density_multiplier *= 1.3
            
            # Adjust based on volatility confidence
            if volatility_params.composite_volatility > 0.5:  # High volatility
                density_multiplier *= 0.8  # Reduce density in high volatility
            elif volatility_params.composite_volatility < 0.2:  # Low volatility
                density_multiplier *= 1.3  # Increase density in low volatility
            
            return max(0.5, min(3.0, density_multiplier))  # Bound the multiplier
            
        except Exception as e:
            self.logger.error(f"Grid density adjustment failed: {e}")
            return volatility_params.density_multiplier

    def _is_round_number(self, price: float) -> bool:
        """Check if price is a round number (psychological level)"""
        try:
            # Check for round numbers at different scales
            price_str = f"{price:.8f}".rstrip('0').rstrip('.')
            
            # Major round numbers (e.g., 50000, 60000)
            if price >= 1000 and price % 1000 == 0:
                return True
            
            # Medium round numbers (e.g., 50500, 60200)
            if price >= 100 and price % 100 == 0:
                return True
            
            # Small round numbers (e.g., 50.50, 60.25)
            if price >= 10 and (price * 4) % 1 == 0:  # Quarter increments
                return True
            
            # Very small round numbers (e.g., 0.50, 1.25)
            if price < 10 and (price * 20) % 1 == 0:  # 0.05 increments
                return True
            
            return False
            
        except Exception:
            return False

    async def _get_price_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get price data from Binance API"""
        try:
            if self.binance_client:
                klines = self.binance_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades_count',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert to proper data types
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get price data for {symbol}: {e}")
            return None

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            if self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            return 0.0
        except Exception:
            return 0.0

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._vol_cache:
            return False
        
        cache_time = self._vol_cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self._cache_ttl

    def _update_cache(self, key: str, value: float):
        """Update cache with new value"""
        self._vol_cache[key] = {
            'value': value,
            'timestamp': datetime.now()
        }

    async def get_volatility_surface_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive volatility surface analysis for advanced grid management.
        Returns detailed volatility metrics and market regime analysis.
        """
        try:
            vol_metrics = await self.calculate_composite_volatility(symbol)
            grid_params = await self.calculate_dynamic_spacing(symbol, await self._get_current_price(symbol))
            
            # Additional surface analysis
            term_structure = await self._analyze_term_structure(symbol)
            skew_analysis = await self._analyze_volatility_skew(symbol)
            
            return {
                'volatility_metrics': {
                    'bitvol': vol_metrics.bitvol,
                    'lxvx': vol_metrics.lxvx,
                    'garch_vol': vol_metrics.garch_vol,
                    'realized_vol': vol_metrics.realized_vol,
                    'parkinson_vol': vol_metrics.parkinson_vol,
                    'composite_vol': vol_metrics.composite_vol,
                    'confidence': vol_metrics.confidence
                },
                'grid_parameters': {
                    'spacing': grid_params.spacing,
                    'density_multiplier': grid_params.density_multiplier,
                    'max_levels': grid_params.max_levels,
                    'regime': grid_params.regime,
                    'stress_factor': grid_params.stress_factor,
                    'liquidity_factor': grid_params.liquidity_factor
                },
                'market_analysis': {
                    'regime': vol_metrics.regime,
                    'term_structure': term_structure,
                    'skew_analysis': skew_analysis
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Volatility surface analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _analyze_term_structure(self, symbol: str) -> Dict[str, Any]:
        """Analyze volatility term structure"""
        try:
            # Calculate volatilities for different time horizons
            vol_7d = await self._calculate_realized_volatility(symbol, period=7)
            vol_30d = await self._calculate_realized_volatility(symbol, period=30)
            vol_90d = await self._calculate_realized_volatility(symbol, period=90)
            
            # Term structure analysis
            contango = vol_90d > vol_30d > vol_7d
            backwardation = vol_7d > vol_30d > vol_90d
            term_slope = (vol_90d - vol_7d) / vol_7d if vol_7d > 0 else 0
            
            return {
                'vol_7d': vol_7d,
                'vol_30d': vol_30d,
                'vol_90d': vol_90d,
                'contango': contango,
                'backwardation': backwardation,
                'term_slope': term_slope,
                'stress_signal': backwardation and term_slope < -0.2
            }
            
        except Exception:
            return {
                'vol_7d': 0.3, 'vol_30d': 0.3, 'vol_90d': 0.3,
                'contango': True, 'backwardation': False,
                'term_slope': 0, 'stress_signal': False
            }

    async def _analyze_volatility_skew(self, symbol: str) -> Dict[str, Any]:
        """Analyze volatility skew for directional bias"""
        try:
            # Get price data for skew analysis
            price_data = await self._get_price_data(symbol, interval='1h', limit=168)
            if price_data is None:
                return {'skew': 0, 'bearish_skew': False, 'bullish_skew': False}
            
            returns = np.log(price_data['close'].values[1:] / price_data['close'].values[:-1])
            
            # Calculate skewness of returns distribution
            skewness = stats.skew(returns) if len(returns) > 10 else 0
            
            # Interpret skewness
            bearish_skew = skewness < -0.5  # Negative skew (fat left tail)
            bullish_skew = skewness > 0.5   # Positive skew (fat right tail)
            
            return {
                'skew': float(skewness),
                'bearish_skew': bearish_skew,
                'bullish_skew': bullish_skew,
                'interpretation': 'bearish' if bearish_skew else 'bullish' if bullish_skew else 'neutral'
            }
            
        except Exception:
            return {
                'skew': 0,
                'bearish_skew': False,
                'bullish_skew': False,
                'interpretation': 'neutral'
            }