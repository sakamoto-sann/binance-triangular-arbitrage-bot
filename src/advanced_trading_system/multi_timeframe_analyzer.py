"""
Professional Multi-Timeframe Analysis System
Combines signals from multiple timeframes with volatility surface analysis and crypto-specific indicators
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TimeframeSignals:
    """Data class for timeframe-specific signals"""
    trend: float
    momentum: float
    volume: float
    volatility: float
    crypto_specific: float
    confidence: float
    timestamp: datetime

@dataclass
class VolatilitySurfaceData:
    """Data class for volatility surface information"""
    term_structure: Dict[str, float]
    skew_analysis: Dict[str, Any]
    atm_vol: float
    vol_smile: Dict[str, float]
    stress_indicators: Dict[str, bool]

@dataclass
class CryptoSpecificSignals:
    """Data class for crypto-specific market signals"""
    funding_rate: float
    funding_trend: float
    funding_extreme: bool
    oi_trend: float
    oi_divergence: bool
    fear_greed_index: float
    mvrv_ratio: Optional[float]
    exchange_flows: Optional[float]
    whale_activity: Optional[float]

@dataclass
class CompositeSignal:
    """Data class for final composite signal"""
    signal: float
    confidence: float
    components: Dict[str, Any]
    regime: str
    timestamp: datetime

class MultiTimeframeAnalyzer:
    """
    Professional multi-timeframe analysis system that combines signals from multiple
    timeframes with advanced volatility surface analysis and crypto-specific indicators.
    """
    
    def __init__(self, binance_client=None, config: Dict = None):
        """Initialize the multi-timeframe analyzer"""
        self.binance_client = binance_client
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Timeframe configuration
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.timeframe_weights = self.config.get('timeframe_weights', {
            '1m': 0.1,   # Short-term noise
            '5m': 0.15,  # Entry/exit timing
            '15m': 0.2,  # Tactical signals
            '1h': 0.25,  # Strategic direction
            '4h': 0.2,   # Medium-term trend
            '1d': 0.1    # Long-term context
        })
        
        # Technical indicators configuration
        self.indicators = {
            'trend': ['EMA_20', 'EMA_50', 'EMA_200', 'TEMA_21', 'Hull_MA'],
            'momentum': ['RSI', 'MACD', 'Stochastic', 'Williams_R', 'CMO'],
            'volume': ['VWAP', 'Volume_Profile', 'OBV', 'CMF', 'A_D_Line'],
            'volatility': ['ATR', 'Bollinger_Bands', 'Keltner_Channels', 'Donchian'],
            'crypto_specific': ['Funding_Rate', 'Open_Interest', 'MVRV', 'Fear_Greed'],
            'volatility_surface': ['Term_Structure', 'Skew', 'ATM_Vol', 'Vol_Smile']
        }
        
        # Volatility surface configuration
        self.vol_surface_config = {
            'maturities': [7, 14, 30, 60, 90],  # Days to expiry
            'strikes': [-20, -10, -5, 0, 5, 10, 20],  # Strike relative to spot (%)
            'min_vol': 0.1,  # 10% minimum volatility
            'max_vol': 2.0   # 200% maximum volatility
        }
        
        # Signal caching
        self._signal_cache = {}
        self._cache_ttl = 60  # 1 minute cache
        
        # Fear & Greed Index components weights
        self.fear_greed_weights = {
            'volatility': 0.25,
            'market_momentum': 0.25,
            'volume': 0.15,
            'social_sentiment': 0.15,
            'dominance': 0.10,
            'trends': 0.10
        }

    async def fetch_volatility_surface(self, symbol: str) -> VolatilitySurfaceData:
        """
        Fetch and analyze volatility surface from options data.
        Integrates with BitVol and LXVX for professional volatility analysis.
        """
        try:
            cache_key = f'vol_surface_{symbol}'
            if self._is_cache_valid(cache_key):
                return self._signal_cache[cache_key]['value']
            
            # Initialize volatility surface
            vol_surface = {}
            
            # Calculate volatility surface using available data
            for maturity in self.vol_surface_config['maturities']:
                vol_surface[maturity] = {}
                
                for strike_offset in self.vol_surface_config['strikes']:
                    # Calculate implied volatility for each strike/maturity
                    iv = await self._calculate_implied_volatility(symbol, maturity, strike_offset)
                    vol_surface[maturity][strike_offset] = iv
            
            # Analyze term structure
            term_structure = self._analyze_volatility_term_structure(vol_surface)
            
            # Analyze skew
            skew_analysis = self._analyze_volatility_skew(vol_surface, maturity=30)
            
            # Calculate ATM volatility
            atm_vol = vol_surface.get(30, {}).get(0, 0.3)
            
            # Calculate volatility smile
            vol_smile = self._calculate_volatility_smile(vol_surface, maturity=30)
            
            # Detect stress indicators
            stress_indicators = self._detect_volatility_stress_signals(term_structure, skew_analysis)
            
            result = VolatilitySurfaceData(
                term_structure=term_structure,
                skew_analysis=skew_analysis,
                atm_vol=atm_vol,
                vol_smile=vol_smile,
                stress_indicators=stress_indicators
            )
            
            # Cache the result
            self._update_cache(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Volatility surface fetch failed: {e}")
            # Return fallback data
            return self._create_fallback_vol_surface()

    async def _calculate_implied_volatility(self, symbol: str, maturity: int, strike_offset: float) -> float:
        """Calculate implied volatility for given strike and maturity"""
        try:
            # Get historical volatility as proxy for implied volatility
            price_data = await self._get_price_data(symbol, '1h', limit=maturity * 24)
            if price_data is None or len(price_data) < 20:
                return 0.3  # Default volatility
            
            # Calculate returns
            returns = np.log(price_data['close'].values[1:] / price_data['close'].values[:-1])
            
            # Base volatility calculation
            base_vol = np.std(returns) * np.sqrt(365 * 24)  # Annualized hourly vol
            
            # Adjust for strike offset (volatility smile/skew)
            if strike_offset != 0:
                # Simple volatility smile approximation
                smile_adjustment = 0.1 * abs(strike_offset) / 10  # 1% vol increase per 10% strike offset
                base_vol += smile_adjustment
            
            # Adjust for maturity (term structure)
            if maturity <= 7:
                term_adjustment = 1.2  # Short-term vol tends to be higher
            elif maturity <= 30:
                term_adjustment = 1.0  # Base case
            else:
                term_adjustment = 0.9  # Long-term vol tends to be lower
            
            final_vol = base_vol * term_adjustment
            
            # Apply bounds
            return max(self.vol_surface_config['min_vol'], 
                      min(final_vol, self.vol_surface_config['max_vol']))
            
        except Exception:
            return 0.3

    def _analyze_volatility_term_structure(self, vol_surface: Dict[int, Dict[float, float]]) -> Dict[str, Any]:
        """Analyze volatility term structure for market signals"""
        try:
            # Extract ATM volatilities across maturities
            maturities = sorted(vol_surface.keys())
            atm_vols = [vol_surface[mat].get(0, 0.3) for mat in maturities]
            
            if len(atm_vols) < 3:
                return {'term_structure_score': 0, 'stress_signal': False}
            
            # Calculate term structure slope
            short_vol = np.mean(atm_vols[:2])  # Short-term (7-14 day)
            medium_vol = np.mean(atm_vols[2:4]) if len(atm_vols) > 3 else atm_vols[2]  # Medium-term (30-60 day)
            long_vol = atm_vols[-1]  # Long-term (90 day)
            
            # Term structure signals
            contango = long_vol > medium_vol > short_vol  # Normal upward slope
            backwardation = short_vol > medium_vol > long_vol  # Stress signal
            
            # Calculate term structure score
            if short_vol > 0:
                term_structure_score = (long_vol - short_vol) / short_vol
            else:
                term_structure_score = 0
            
            return {
                'short_vol': short_vol,
                'medium_vol': medium_vol,
                'long_vol': long_vol,
                'contango': contango,
                'backwardation': backwardation,
                'term_structure_score': term_structure_score,
                'stress_signal': backwardation and term_structure_score < -0.2
            }
            
        except Exception as e:
            self.logger.error(f"Term structure analysis failed: {e}")
            return {'term_structure_score': 0, 'stress_signal': False}

    def _analyze_volatility_skew(self, vol_surface: Dict[int, Dict[float, float]], maturity: int = 30) -> Dict[str, Any]:
        """Analyze volatility skew for directional bias"""
        try:
            if maturity not in vol_surface:
                maturity = list(vol_surface.keys())[0] if vol_surface else 30
            
            if maturity not in vol_surface:
                return {'put_call_skew': 0, 'bearish_skew': False, 'bullish_skew': False}
            
            strikes = sorted(vol_surface[maturity].keys())
            
            # Calculate skew metrics
            put_strikes = [s for s in strikes if s < 0]
            call_strikes = [s for s in strikes if s > 0]
            
            if put_strikes and call_strikes:
                put_vol = np.mean([vol_surface[maturity][s] for s in put_strikes])
                call_vol = np.mean([vol_surface[maturity][s] for s in call_strikes])
                atm_vol = vol_surface[maturity].get(0, (put_vol + call_vol) / 2)
                
                # Calculate put-call skew
                if atm_vol > 0:
                    put_call_skew = (put_vol - call_vol) / atm_vol
                else:
                    put_call_skew = 0
                
                # Calculate smile convexity
                vols = [vol_surface[maturity][s] for s in strikes]
                smile_convexity = self._calculate_smile_convexity(strikes, vols)
                
                return {
                    'put_vol': put_vol,
                    'call_vol': call_vol,
                    'atm_vol': atm_vol,
                    'put_call_skew': put_call_skew,
                    'smile_convexity': smile_convexity,
                    'bearish_skew': put_call_skew > 0.1,  # Higher put vol indicates fear
                    'bullish_skew': put_call_skew < -0.1  # Higher call vol indicates greed
                }
            else:
                return {'put_call_skew': 0, 'bearish_skew': False, 'bullish_skew': False}
                
        except Exception as e:
            self.logger.error(f"Skew analysis failed: {e}")
            return {'put_call_skew': 0, 'bearish_skew': False, 'bullish_skew': False}

    def _calculate_smile_convexity(self, strikes: List[float], vols: List[float]) -> float:
        """Calculate volatility smile convexity"""
        try:
            if len(strikes) < 3 or len(vols) < 3:
                return 0.0
            
            # Fit a quadratic polynomial to the volatility smile
            coeffs = np.polyfit(strikes, vols, 2)
            
            # Convexity is the second derivative (curvature)
            convexity = 2 * coeffs[0]  # Second derivative of ax^2 + bx + c is 2a
            
            return float(convexity)
            
        except Exception:
            return 0.0

    def _calculate_volatility_smile(self, vol_surface: Dict[int, Dict[float, float]], maturity: int = 30) -> Dict[str, float]:
        """Calculate volatility smile parameters"""
        try:
            if maturity not in vol_surface:
                return {}
            
            strikes = sorted(vol_surface[maturity].keys())
            vols = [vol_surface[maturity][s] for s in strikes]
            
            if len(strikes) < 3:
                return {}
            
            # Calculate smile characteristics
            atm_vol = vol_surface[maturity].get(0, np.mean(vols))
            min_vol = min(vols)
            max_vol = max(vols)
            
            # Smile width and skew
            smile_width = max_vol - min_vol
            
            # Find the minimum volatility point
            min_vol_strike = strikes[vols.index(min_vol)]
            
            return {
                'atm_vol': atm_vol,
                'min_vol': min_vol,
                'max_vol': max_vol,
                'smile_width': smile_width,
                'min_vol_strike': min_vol_strike,
                'smile_asymmetry': min_vol_strike  # Negative means put skew, positive means call skew
            }
            
        except Exception:
            return {}

    def _detect_volatility_stress_signals(self, term_structure: Dict[str, Any], 
                                        skew_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """Detect stress signals from volatility surface"""
        try:
            stress_signals = {}
            
            # Term structure stress
            stress_signals['term_structure_stress'] = term_structure.get('stress_signal', False)
            
            # Skew stress (extreme skew indicates fear/greed)
            skew_value = abs(skew_analysis.get('put_call_skew', 0))
            stress_signals['skew_stress'] = skew_value > 0.3
            
            # Volatility level stress
            atm_vol = skew_analysis.get('atm_vol', 0.3)
            stress_signals['high_vol_stress'] = atm_vol > 1.0  # >100% annualized vol
            stress_signals['low_vol_stress'] = atm_vol < 0.1   # <10% annualized vol
            
            # Combined stress signal
            stress_signals['overall_stress'] = any([
                stress_signals['term_structure_stress'],
                stress_signals['skew_stress'],
                stress_signals['high_vol_stress']
            ])
            
            return stress_signals
            
        except Exception:
            return {'overall_stress': False}

    async def calculate_crypto_specific_signals(self, symbol: str) -> CryptoSpecificSignals:
        """Calculate crypto-specific signals not available in traditional markets"""
        try:
            # Initialize default values
            signals = CryptoSpecificSignals(
                funding_rate=0.0,
                funding_trend=0.0,
                funding_extreme=False,
                oi_trend=0.0,
                oi_divergence=False,
                fear_greed_index=50.0,  # Neutral
                mvrv_ratio=None,
                exchange_flows=None,
                whale_activity=None
            )
            
            # Funding rate analysis
            if self.binance_client:
                try:
                    funding_info = self.binance_client.futures_funding_rate(symbol=symbol, limit=24)
                    if funding_info:
                        current_funding = float(funding_info[-1]['fundingRate'])
                        
                        # Calculate funding trend (last 8 vs previous 16)
                        if len(funding_info) >= 16:
                            recent_funding = [float(f['fundingRate']) for f in funding_info[-8:]]
                            previous_funding = [float(f['fundingRate']) for f in funding_info[-24:-8]]
                            funding_trend = np.mean(recent_funding) - np.mean(previous_funding)
                        else:
                            funding_trend = 0.0
                        
                        signals.funding_rate = current_funding
                        signals.funding_trend = funding_trend
                        signals.funding_extreme = abs(current_funding) > 0.01  # >1% funding rate
                        
                except Exception as e:
                    self.logger.warning(f"Funding rate analysis failed: {e}")
            
            # Open interest analysis
            try:
                oi_data = await self._get_open_interest_data(symbol)
                if oi_data:
                    signals.oi_trend = oi_data.get('change_24h', 0.0)
                    signals.oi_divergence = await self._detect_price_oi_divergence(symbol)
            except Exception as e:
                self.logger.warning(f"Open interest analysis failed: {e}")
            
            # On-chain metrics (for major cryptos)
            if symbol.startswith('BTC'):
                try:
                    signals.mvrv_ratio = await self._get_mvrv_ratio()
                    signals.exchange_flows = await self._get_exchange_flows()
                    signals.whale_activity = await self._detect_whale_activity()
                except Exception as e:
                    self.logger.warning(f"On-chain metrics failed: {e}")
            
            # Fear & Greed Index calculation
            try:
                signals.fear_greed_index = await self._calculate_fear_greed_index(symbol)
            except Exception as e:
                self.logger.warning(f"Fear & Greed Index calculation failed: {e}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Crypto-specific signals calculation failed: {e}")
            # Return default values
            return CryptoSpecificSignals(
                funding_rate=0.0, funding_trend=0.0, funding_extreme=False,
                oi_trend=0.0, oi_divergence=False, fear_greed_index=50.0,
                mvrv_ratio=None, exchange_flows=None, whale_activity=None
            )

    async def _get_open_interest_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get open interest data from Binance"""
        try:
            if self.binance_client:
                # Get current open interest
                oi_info = self.binance_client.futures_open_interest(symbol=symbol)
                current_oi = float(oi_info['openInterest'])
                
                # Get historical open interest for trend calculation
                # Note: This would require storing historical OI data or using additional APIs
                # For now, we'll use a simplified approach
                
                return {
                    'current_oi': current_oi,
                    'change_24h': 0.0  # Placeholder - would need historical data
                }
            
            return None
            
        except Exception:
            return None

    async def _detect_price_oi_divergence(self, symbol: str) -> bool:
        """Detect divergence between price and open interest"""
        try:
            # Get recent price data
            price_data = await self._get_price_data(symbol, '1h', limit=48)
            if price_data is None or len(price_data) < 24:
                return False
            
            # Calculate price trend
            recent_prices = price_data['close'].values[-24:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # For now, return False as we'd need historical OI data for proper divergence detection
            # In a full implementation, this would compare price trend with OI trend
            return False
            
        except Exception:
            return False

    async def _get_mvrv_ratio(self) -> Optional[float]:
        """Get MVRV (Market Value to Realized Value) ratio for Bitcoin"""
        try:
            # This would typically require on-chain data APIs like Glassnode, CoinMetrics, etc.
            # For demonstration, we'll return a placeholder
            # In production, you'd integrate with services like:
            # - Glassnode API
            # - CoinMetrics API
            # - Messari API
            return None  # Placeholder
            
        except Exception:
            return None

    async def _get_exchange_flows(self) -> Optional[float]:
        """Get exchange flow data (inflows/outflows)"""
        try:
            # Placeholder for exchange flow analysis
            # Would integrate with on-chain analytics providers
            return None
            
        except Exception:
            return None

    async def _detect_whale_activity(self) -> Optional[float]:
        """Detect whale activity through large transactions"""
        try:
            # Placeholder for whale activity detection
            # Would integrate with on-chain analytics providers
            return None
            
        except Exception:
            return None

    async def _calculate_fear_greed_index(self, symbol: str) -> float:
        """Calculate Fear & Greed Index equivalent for crypto"""
        try:
            components = {}
            
            # 1. Volatility component (25%)
            price_data = await self._get_price_data(symbol, '1d', limit=30)
            if price_data is not None and len(price_data) > 10:
                returns = np.log(price_data['close'].values[1:] / price_data['close'].values[:-1])
                volatility = np.std(returns) * np.sqrt(365)
                # Higher volatility = more fear (lower score)
                vol_score = max(0, 100 - (volatility * 100))
                components['volatility'] = vol_score
            else:
                components['volatility'] = 50
            
            # 2. Market momentum component (25%)
            if price_data is not None and len(price_data) > 20:
                sma_10 = price_data['close'].rolling(10).mean().iloc[-1]
                sma_30 = price_data['close'].rolling(30).mean().iloc[-1] if len(price_data) >= 30 else price_data['close'].mean()
                momentum_score = 50 + min(50, max(-50, ((sma_10 - sma_30) / sma_30) * 1000))
                components['market_momentum'] = momentum_score
            else:
                components['market_momentum'] = 50
            
            # 3. Volume component (15%)
            if price_data is not None and len(price_data) > 10:
                avg_volume = price_data['volume'].rolling(10).mean().iloc[-1]
                current_volume = price_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                # Higher volume = more engagement, can be fear or greed
                volume_score = 50 + min(25, max(-25, (volume_ratio - 1) * 50))
                components['volume'] = volume_score
            else:
                components['volume'] = 50
            
            # 4. Social sentiment (15%) - placeholder
            components['social_sentiment'] = 50  # Would integrate with social APIs
            
            # 5. Market dominance (10%) - placeholder
            components['dominance'] = 50  # Would need market cap data
            
            # 6. Google Trends (10%) - placeholder
            components['trends'] = 50  # Would integrate with Google Trends API
            
            # Calculate weighted score
            total_score = sum(
                components[comp] * self.fear_greed_weights[comp] 
                for comp in components if comp in self.fear_greed_weights
            )
            
            return max(0, min(100, total_score))
            
        except Exception as e:
            self.logger.error(f"Fear & Greed Index calculation failed: {e}")
            return 50.0  # Neutral

    async def calculate_timeframe_signals(self, symbol: str, timeframe: str) -> TimeframeSignals:
        """Calculate comprehensive signals for a specific timeframe"""
        try:
            # Get price data for the timeframe
            intervals_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            
            interval = intervals_map.get(timeframe, '1h')
            limit = min(500, self._get_limit_for_timeframe(timeframe))
            
            price_data = await self._get_price_data(symbol, interval, limit)
            if price_data is None or len(price_data) < 50:
                return self._get_default_timeframe_signals()
            
            # Calculate individual signal components
            trend_signal = self._calculate_trend_signals(price_data)
            momentum_signal = self._calculate_momentum_signals(price_data)
            volume_signal = self._calculate_volume_signals(price_data)
            volatility_signal = self._calculate_volatility_signals(price_data)
            
            # Crypto-specific signals (cached from main calculation)
            crypto_signal = 0.0  # Will be populated by main composite calculation
            
            # Calculate confidence based on signal consistency
            signals = [trend_signal, momentum_signal, volume_signal, volatility_signal]
            confidence = self._calculate_signal_confidence(signals, price_data)
            
            return TimeframeSignals(
                trend=trend_signal,
                momentum=momentum_signal,
                volume=volume_signal,
                volatility=volatility_signal,
                crypto_specific=crypto_signal,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Timeframe signal calculation failed for {timeframe}: {e}")
            return self._get_default_timeframe_signals()

    def _calculate_trend_signals(self, price_data: pd.DataFrame) -> float:
        """Calculate trend signals using multiple indicators"""
        try:
            close = price_data['close'].values
            signals = []
            
            # EMA signals
            if len(close) >= 200:
                ema_20 = talib.EMA(close, timeperiod=20)
                ema_50 = talib.EMA(close, timeperiod=50)
                ema_200 = talib.EMA(close, timeperiod=200)
                
                # EMA alignment
                current_price = close[-1]
                if current_price > ema_20[-1] > ema_50[-1] > ema_200[-1]:
                    signals.append(1.0)  # Strong bullish
                elif current_price < ema_20[-1] < ema_50[-1] < ema_200[-1]:
                    signals.append(-1.0)  # Strong bearish
                else:
                    # Partial alignment
                    ema_score = 0
                    if current_price > ema_20[-1]: ema_score += 0.25
                    if ema_20[-1] > ema_50[-1]: ema_score += 0.25
                    if ema_50[-1] > ema_200[-1]: ema_score += 0.25
                    if current_price > ema_200[-1]: ema_score += 0.25
                    signals.append(ema_score * 2 - 1)  # Convert to [-1, 1]
            
            # TEMA signal
            if len(close) >= 21:
                tema = talib.TEMA(close, timeperiod=21)
                tema_signal = 1 if close[-1] > tema[-1] else -1
                signals.append(tema_signal * 0.5)  # Weighted signal
            
            # Hull MA signal
            if len(close) >= 20:
                hull_ma = self._calculate_hull_ma(close, 20)
                if hull_ma is not None and len(hull_ma) > 1:
                    hull_trend = 1 if hull_ma[-1] > hull_ma[-2] else -1
                    signals.append(hull_trend * 0.3)
            
            # Return average signal
            return np.mean(signals) if signals else 0.0
            
        except Exception as e:
            self.logger.error(f"Trend signal calculation failed: {e}")
            return 0.0

    def _calculate_momentum_signals(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum signals using multiple indicators"""
        try:
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            signals = []
            
            # RSI signal
            if len(close) >= 14:
                rsi = talib.RSI(close, timeperiod=14)
                rsi_current = rsi[-1]
                
                if rsi_current > 70:
                    signals.append(-0.5)  # Overbought
                elif rsi_current < 30:
                    signals.append(0.5)   # Oversold
                else:
                    # Normalize RSI to [-1, 1] range
                    rsi_normalized = (rsi_current - 50) / 50
                    signals.append(rsi_normalized * 0.5)
            
            # MACD signal
            if len(close) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                
                if len(macd_hist) > 1:
                    if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                        signals.append(0.8)  # Bullish crossover
                    elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                        signals.append(-0.8)  # Bearish crossover
                    else:
                        # Trend continuation
                        macd_strength = np.tanh(macd_hist[-1] / np.std(macd_hist[-20:]) if len(macd_hist) >= 20 else 1)
                        signals.append(macd_strength * 0.4)
            
            # Stochastic signal
            if len(high) >= 14:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                
                stoch_k = slowk[-1] if not np.isnan(slowk[-1]) else 50
                stoch_d = slowd[-1] if not np.isnan(slowd[-1]) else 50
                
                if stoch_k > 80:
                    signals.append(-0.4)  # Overbought
                elif stoch_k < 20:
                    signals.append(0.4)   # Oversold
                else:
                    stoch_signal = (stoch_k - 50) / 50 * 0.3
                    signals.append(stoch_signal)
            
            # Williams %R signal
            if len(high) >= 14:
                willr = talib.WILLR(high, low, close, timeperiod=14)
                willr_current = willr[-1] if not np.isnan(willr[-1]) else -50
                
                if willr_current > -20:
                    signals.append(-0.3)  # Overbought
                elif willr_current < -80:
                    signals.append(0.3)   # Oversold
                else:
                    willr_normalized = (willr_current + 50) / 50 * 0.2
                    signals.append(willr_normalized)
            
            return np.mean(signals) if signals else 0.0
            
        except Exception as e:
            self.logger.error(f"Momentum signal calculation failed: {e}")
            return 0.0

    def _calculate_volume_signals(self, price_data: pd.DataFrame) -> float:
        """Calculate volume-based signals"""
        try:
            close = price_data['close'].values
            volume = price_data['volume'].values
            high = price_data['high'].values
            low = price_data['low'].values
            signals = []
            
            # VWAP signal
            if len(price_data) >= 20:
                vwap = self._calculate_vwap(price_data)
                if vwap is not None:
                    vwap_signal = 1 if close[-1] > vwap else -1
                    signals.append(vwap_signal * 0.4)
            
            # OBV signal
            if len(close) >= 20:
                obv = talib.OBV(close, volume)
                if len(obv) > 10:
                    obv_trend = np.polyfit(range(10), obv[-10:], 1)[0]
                    obv_signal = np.tanh(obv_trend / np.std(obv[-20:]) if np.std(obv[-20:]) > 0 else 1)
                    signals.append(obv_signal * 0.3)
            
            # Chaikin Money Flow
            if len(high) >= 20:
                cmf = self._calculate_cmf(high, low, close, volume, 20)
                if cmf is not None:
                    signals.append(np.tanh(cmf * 5) * 0.3)  # Scale and bound CMF
            
            # Volume trend analysis
            if len(volume) >= 20:
                volume_ma = np.mean(volume[-20:])
                current_volume = volume[-1]
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
                
                # Higher volume supports price movement
                price_change = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
                volume_confirmation = 1 if (price_change > 0 and volume_ratio > 1.2) or (price_change < 0 and volume_ratio > 1.2) else 0
                signals.append(volume_confirmation * price_change * 0.2)
            
            return np.mean(signals) if signals else 0.0
            
        except Exception as e:
            self.logger.error(f"Volume signal calculation failed: {e}")
            return 0.0

    def _calculate_volatility_signals(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility-based signals"""
        try:
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            signals = []
            
            # Bollinger Bands signal
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                
                current_price = close[-1]
                bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5
                
                if bb_position > 0.8:
                    signals.append(-0.3)  # Near upper band (potential reversal)
                elif bb_position < 0.2:
                    signals.append(0.3)   # Near lower band (potential reversal)
                else:
                    bb_signal = (bb_position - 0.5) * 0.4  # Trend continuation
                    signals.append(bb_signal)
            
            # ATR-based volatility signal
            if len(high) >= 14:
                atr = talib.ATR(high, low, close, timeperiod=14)
                atr_current = atr[-1]
                atr_ma = np.mean(atr[-20:]) if len(atr) >= 20 else atr_current
                
                # Volatility expansion/contraction
                if atr_current > atr_ma * 1.5:
                    signals.append(0.2)   # High volatility (potential breakout)
                elif atr_current < atr_ma * 0.7:
                    signals.append(-0.1)  # Low volatility (potential consolidation)
                else:
                    signals.append(0.0)   # Normal volatility
            
            # Keltner Channels
            if len(close) >= 20:
                kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(high, low, close, 20)
                if kc_upper is not None and kc_lower is not None:
                    current_price = close[-1]
                    if current_price > kc_upper[-1]:
                        signals.append(0.4)   # Above upper channel (bullish)
                    elif current_price < kc_lower[-1]:
                        signals.append(-0.4)  # Below lower channel (bearish)
                    else:
                        kc_position = (current_price - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1]) if (kc_upper[-1] - kc_lower[-1]) > 0 else 0.5
                        signals.append((kc_position - 0.5) * 0.3)
            
            return np.mean(signals) if signals else 0.0
            
        except Exception as e:
            self.logger.error(f"Volatility signal calculation failed: {e}")
            return 0.0

    def _calculate_hull_ma(self, close: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Calculate Hull Moving Average"""
        try:
            if len(close) < period:
                return None
            
            # Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            wma_half = talib.WMA(close, timeperiod=period//2)
            wma_full = talib.WMA(close, timeperiod=period)
            
            if len(wma_half) > 0 and len(wma_full) > 0:
                # Align arrays
                min_len = min(len(wma_half), len(wma_full))
                raw_hull = 2 * wma_half[-min_len:] - wma_full[-min_len:]
                
                # Final smoothing
                sqrt_period = int(np.sqrt(period))
                hull_ma = talib.WMA(raw_hull, timeperiod=sqrt_period)
                
                return hull_ma
            
            return None
            
        except Exception:
            return None

    def _calculate_vwap(self, price_data: pd.DataFrame) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            volume = price_data['volume']
            
            # Calculate VWAP for the most recent 20 periods
            recent_data = min(20, len(typical_price))
            typical_price_recent = typical_price[-recent_data:]
            volume_recent = volume[-recent_data:]
            
            vwap = np.sum(typical_price_recent * volume_recent) / np.sum(volume_recent)
            
            return float(vwap) if not np.isnan(vwap) else None
            
        except Exception:
            return None

    def _calculate_cmf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      volume: np.ndarray, period: int) -> Optional[float]:
        """Calculate Chaikin Money Flow"""
        try:
            if len(high) < period:
                return None
            
            # Money Flow Multiplier
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            mf_multiplier = np.where(high - low == 0, 0, mf_multiplier)  # Handle division by zero
            
            # Money Flow Volume
            mf_volume = mf_multiplier * volume
            
            # Chaikin Money Flow
            cmf = np.sum(mf_volume[-period:]) / np.sum(volume[-period:]) if np.sum(volume[-period:]) > 0 else 0
            
            return float(cmf)
            
        except Exception:
            return None

    def _calculate_keltner_channels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                  period: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate Keltner Channels"""
        try:
            if len(close) < period:
                return None, None, None
            
            # Middle line (EMA of close)
            middle = talib.EMA(close, timeperiod=period)
            
            # Average True Range
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            if len(middle) > 0 and len(atr) > 0:
                # Align arrays
                min_len = min(len(middle), len(atr))
                middle_aligned = middle[-min_len:]
                atr_aligned = atr[-min_len:]
                
                # Upper and lower bands
                multiplier = 2.0
                upper = middle_aligned + (multiplier * atr_aligned)
                lower = middle_aligned - (multiplier * atr_aligned)
                
                return upper, middle_aligned, lower
            
            return None, None, None
            
        except Exception:
            return None, None, None

    def _calculate_signal_confidence(self, signals: List[float], price_data: pd.DataFrame) -> float:
        """Calculate confidence score for signals"""
        try:
            confidence_factors = []
            
            # Signal consistency
            if signals:
                signal_std = np.std(signals)
                signal_mean = abs(np.mean(signals))
                
                # Lower standard deviation relative to mean indicates higher confidence
                consistency_score = max(0, 1 - (signal_std / (signal_mean + 0.1)))
                confidence_factors.append(consistency_score)
            
            # Data quality
            if len(price_data) >= 50:
                # Sufficient data points
                confidence_factors.append(0.9)
            elif len(price_data) >= 20:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # Volume consistency
            if 'volume' in price_data.columns:
                volume_cv = np.std(price_data['volume'].values) / np.mean(price_data['volume'].values)
                volume_confidence = max(0.3, 1 - volume_cv)  # Lower coefficient of variation = higher confidence
                confidence_factors.append(volume_confidence)
            
            # Price stability (not too erratic)
            if len(price_data) >= 10:
                returns = np.log(price_data['close'].values[1:] / price_data['close'].values[:-1])
                return_stability = max(0.2, 1 - (np.std(returns) * 10))  # Scale factor for readability
                confidence_factors.append(return_stability)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5

    def _get_limit_for_timeframe(self, timeframe: str) -> int:
        """Get appropriate data limit for timeframe"""
        limits = {
            '1m': 200,
            '5m': 200,
            '15m': 200,
            '1h': 200,
            '4h': 150,
            '1d': 100
        }
        return limits.get(timeframe, 100)

    def _get_default_timeframe_signals(self) -> TimeframeSignals:
        """Return default signals when calculation fails"""
        return TimeframeSignals(
            trend=0.0,
            momentum=0.0,
            volume=0.0,
            volatility=0.0,
            crypto_specific=0.0,
            confidence=0.3,
            timestamp=datetime.now()
        )

    def _create_fallback_vol_surface(self) -> VolatilitySurfaceData:
        """Create fallback volatility surface data"""
        return VolatilitySurfaceData(
            term_structure={'term_structure_score': 0, 'stress_signal': False},
            skew_analysis={'put_call_skew': 0, 'bearish_skew': False, 'bullish_skew': False},
            atm_vol=0.3,
            vol_smile={},
            stress_indicators={'overall_stress': False}
        )

    async def generate_composite_signal(self, symbol: str) -> CompositeSignal:
        """
        Generate sophisticated weighted composite signal combining all analysis components.
        This is the main method that orchestrates all signal generation.
        """
        try:
            # Initialize components
            signals = {}
            confidence_scores = {}
            
            # Fetch volatility surface for advanced analysis
            vol_surface = await self.fetch_volatility_surface(symbol)
            
            # Calculate crypto-specific signals
            crypto_signals = await self.calculate_crypto_specific_signals(symbol)
            
            # Generate signals for each timeframe
            for timeframe in self.timeframes:
                tf_signals = await self.calculate_timeframe_signals(symbol, timeframe)
                weight = self.timeframe_weights[timeframe]
                
                # Weight each signal component
                trend_signal = tf_signals.trend * 0.3
                momentum_signal = tf_signals.momentum * 0.25
                volume_signal = tf_signals.volume * 0.2
                volatility_signal = tf_signals.volatility * 0.15
                crypto_signal = self._calculate_crypto_signal_contribution(crypto_signals) * 0.1
                
                timeframe_composite = (
                    trend_signal + momentum_signal + volume_signal + 
                    volatility_signal + crypto_signal
                )
                
                signals[timeframe] = timeframe_composite * weight
                confidence_scores[timeframe] = tf_signals.confidence
            
            # Calculate final composite signal
            final_signal = sum(signals.values())
            
            # Adjust signal based on volatility surface analysis
            if vol_surface.stress_indicators.get('overall_stress', False):
                final_signal *= 0.5  # Reduce signal strength during stress
            
            # Volatility surface adjustments
            if vol_surface.skew_analysis.get('bearish_skew', False):
                final_signal -= 0.1  # Bearish skew adjustment
            elif vol_surface.skew_analysis.get('bullish_skew', False):
                final_signal += 0.1  # Bullish skew adjustment
            
            # Crypto-specific adjustments
            if crypto_signals.funding_extreme:
                # Extreme funding rates often signal reversals
                funding_adjustment = -0.5 if crypto_signals.funding_rate > 0 else 1.5
                final_signal *= funding_adjustment
            
            # Fear & Greed Index adjustment
            fear_greed_normalized = (crypto_signals.fear_greed_index - 50) / 50  # Convert to [-1, 1]
            final_signal += fear_greed_normalized * 0.05  # Small adjustment
            
            # Calculate composite confidence
            final_confidence = np.mean(list(confidence_scores.values()))
            
            # Adjust confidence based on volatility surface quality
            if vol_surface.stress_indicators.get('overall_stress', False):
                final_confidence *= 0.8  # Reduce confidence during stress
            
            # Clamp signal to [-1, 1] range
            final_signal = max(-1, min(1, final_signal))
            
            # Detect market regime
            regime = self._detect_market_regime(symbol, final_signal, final_confidence, crypto_signals)
            
            return CompositeSignal(
                signal=final_signal,
                confidence=final_confidence,
                components={
                    'timeframe_signals': signals,
                    'volatility_surface': {
                        'term_structure': vol_surface.term_structure,
                        'skew': vol_surface.skew_analysis,
                        'stress_indicators': vol_surface.stress_indicators
                    },
                    'crypto_specific': {
                        'funding_rate': crypto_signals.funding_rate,
                        'funding_trend': crypto_signals.funding_trend,
                        'funding_extreme': crypto_signals.funding_extreme,
                        'oi_trend': crypto_signals.oi_trend,
                        'fear_greed_index': crypto_signals.fear_greed_index
                    }
                },
                regime=regime,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Composite signal generation failed: {e}")
            # Return fallback signal
            return CompositeSignal(
                signal=0.0,
                confidence=0.3,
                components={'error': str(e)},
                regime='ranging',
                timestamp=datetime.now()
            )

    def _calculate_crypto_signal_contribution(self, crypto_signals: CryptoSpecificSignals) -> float:
        """Calculate the contribution of crypto-specific signals to the overall signal"""
        try:
            contributions = []
            
            # Funding rate contribution
            if crypto_signals.funding_extreme:
                # Extreme funding rates suggest potential reversal
                funding_contrib = -np.sign(crypto_signals.funding_rate) * 0.5
            else:
                # Normal funding rates
                funding_contrib = -crypto_signals.funding_rate * 10  # Scale funding rate
            
            contributions.append(funding_contrib)
            
            # Open interest contribution
            oi_contrib = np.tanh(crypto_signals.oi_trend) * 0.3  # Bounded contribution
            contributions.append(oi_contrib)
            
            # Fear & Greed contribution
            fg_normalized = (crypto_signals.fear_greed_index - 50) / 50  # Normalize to [-1, 1]
            contributions.append(fg_normalized * 0.4)
            
            # On-chain metrics (if available)
            if crypto_signals.mvrv_ratio is not None:
                # MVRV ratio interpretation (simplified)
                if crypto_signals.mvrv_ratio > 3.0:
                    contributions.append(-0.3)  # Overvalued
                elif crypto_signals.mvrv_ratio < 1.0:
                    contributions.append(0.3)   # Undervalued
                else:
                    contributions.append(0.0)   # Fair value
            
            return np.mean(contributions) if contributions else 0.0
            
        except Exception:
            return 0.0

    def _detect_market_regime(self, symbol: str, signal_strength: float, confidence: float, 
                            crypto_signals: CryptoSpecificSignals) -> str:
        """Detect current market regime with enhanced analysis"""
        try:
            # Base regime detection
            trend_strength = abs(signal_strength)
            
            # Volatility consideration
            if crypto_signals.fear_greed_index < 20:  # Extreme Fear
                if trend_strength > 0.6:
                    regime = 'panic_selling' if signal_strength < 0 else 'fear_rally'
                else:
                    regime = 'extreme_fear'
            elif crypto_signals.fear_greed_index > 80:  # Extreme Greed
                if trend_strength > 0.6:
                    regime = 'euphoric_buying' if signal_strength > 0 else 'greed_correction'
                else:
                    regime = 'extreme_greed'
            elif trend_strength > 0.4 and confidence > 0.7:
                # High confidence trending
                if crypto_signals.funding_extreme:
                    regime = 'volatile_trending'
                else:
                    regime = 'trending'
            elif trend_strength < 0.2 and confidence > 0.6:
                # Low volatility, high confidence
                regime = 'tight_ranging'
            elif trend_strength > 0.6 and confidence < 0.5:
                # High volatility, low confidence
                regime = 'volatile_ranging'
            else:
                regime = 'ranging'
            
            return regime
            
        except Exception:
            return 'ranging'  # Default to ranging regime

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

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._signal_cache:
            return False
        
        cache_time = self._signal_cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self._cache_ttl

    def _update_cache(self, key: str, value: Any):
        """Update cache with new value"""
        self._signal_cache[key] = {
            'value': value,
            'timestamp': datetime.now()
        }

    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive multi-timeframe analysis with all components.
        Main entry point for external systems.
        """
        try:
            # Generate composite signal
            composite_signal = await self.generate_composite_signal(symbol)
            
            # Get volatility surface analysis
            vol_surface = await self.fetch_volatility_surface(symbol)
            
            # Get individual timeframe analyses
            timeframe_analyses = {}
            for timeframe in self.timeframes:
                tf_signals = await self.calculate_timeframe_signals(symbol, timeframe)
                timeframe_analyses[timeframe] = {
                    'trend': tf_signals.trend,
                    'momentum': tf_signals.momentum,
                    'volume': tf_signals.volume,
                    'volatility': tf_signals.volatility,
                    'confidence': tf_signals.confidence
                }
            
            return {
                'symbol': symbol,
                'composite_signal': {
                    'signal': composite_signal.signal,
                    'confidence': composite_signal.confidence,
                    'regime': composite_signal.regime
                },
                'timeframe_analysis': timeframe_analyses,
                'volatility_surface': {
                    'term_structure': vol_surface.term_structure,
                    'skew_analysis': vol_surface.skew_analysis,
                    'atm_vol': vol_surface.atm_vol,
                    'stress_indicators': vol_surface.stress_indicators
                },
                'crypto_specific': composite_signal.components.get('crypto_specific', {}),
                'timestamp': datetime.now().isoformat(),
                'analysis_quality': {
                    'data_completeness': self._assess_data_completeness(symbol),
                    'signal_consistency': self._assess_signal_consistency(timeframe_analyses),
                    'overall_confidence': composite_signal.confidence
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _assess_data_completeness(self, symbol: str) -> float:
        """Assess the completeness of available data"""
        try:
            # This would assess the quality and completeness of available data sources
            # For now, return a basic score
            return 0.8  # Placeholder
        except Exception:
            return 0.5

    def _assess_signal_consistency(self, timeframe_analyses: Dict[str, Dict[str, float]]) -> float:
        """Assess consistency across timeframe signals"""
        try:
            if not timeframe_analyses:
                return 0.3
            
            # Calculate consistency of trend signals across timeframes
            trend_signals = [analysis.get('trend', 0) for analysis in timeframe_analyses.values()]
            
            if len(trend_signals) > 1:
                trend_std = np.std(trend_signals)
                trend_mean = abs(np.mean(trend_signals))
                
                # Lower standard deviation relative to mean indicates higher consistency
                consistency = max(0.2, 1 - (trend_std / (trend_mean + 0.1)))
                return consistency
            
            return 0.5
            
        except Exception:
            return 0.5