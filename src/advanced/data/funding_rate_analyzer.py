"""
Advanced Trading System - Funding Rate Analyzer
Cross-exchange funding rate analysis and arbitrage opportunity detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import aiohttp
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class FundingRateData:
    """Funding rate data point."""
    exchange: str
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    mark_price: float
    index_price: float
    predicted_rate: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class FundingArbitrageOpportunity:
    """Funding rate arbitrage opportunity."""
    long_exchange: str
    short_exchange: str
    symbol: str
    rate_differential: float
    estimated_profit_8h: float
    estimated_profit_annual: float
    risk_score: float
    confidence: float
    min_position_size: float
    max_position_size: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class FundingRateForecast:
    """Funding rate forecast."""
    symbol: str
    exchange: str
    current_rate: float
    predicted_next_rate: float
    predicted_trend: str  # "increasing", "decreasing", "stable"
    confidence: float
    forecast_horizon_hours: int
    factors: Dict[str, float]  # Contributing factors to prediction
    timestamp: datetime = field(default_factory=datetime.now)

class FundingRateAnalyzer:
    """
    Advanced funding rate analysis system.
    
    Provides comprehensive funding rate analysis across multiple exchanges,
    including arbitrage detection, trend analysis, and predictive modeling.
    """
    
    def __init__(self, market_data_aggregator):
        """
        Initialize funding rate analyzer.
        
        Args:
            market_data_aggregator: Market data aggregation system
        """
        self.market_data = market_data_aggregator
        
        # Data storage
        self.funding_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self.current_rates: Dict[str, FundingRateData] = {}
        self.arbitrage_opportunities: List[FundingArbitrageOpportunity] = []
        self.forecasts: Dict[str, FundingRateForecast] = {}
        
        # Analysis parameters
        self.exchanges = ["binance", "bybit", "okex", "deribit"]
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.min_arbitrage_threshold = 0.0001  # 0.01% minimum rate difference
        self.forecast_lookback_periods = 100
        
        # Processing tasks
        self.analysis_tasks: List[asyncio.Task] = []
        self.is_active = False
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("FundingRateAnalyzer initialized")
    
    async def start(self) -> bool:
        """Start funding rate analysis."""
        try:
            self.logger.info("Starting funding rate analysis...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Start analysis tasks
            self.analysis_tasks = [
                asyncio.create_task(self._funding_rate_collector()),
                asyncio.create_task(self._arbitrage_detector()),
                asyncio.create_task(self._trend_analyzer()),
                asyncio.create_task(self._rate_predictor())
            ]
            
            self.is_active = True
            self.logger.info("Funding rate analysis started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting funding rate analysis: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop funding rate analysis."""
        try:
            self.logger.info("Stopping funding rate analysis...")
            
            self.is_active = False
            
            # Cancel analysis tasks
            for task in self.analysis_tasks:
                task.cancel()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            self.logger.info("Funding rate analysis stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping funding rate analysis: {e}")
            return False
    
    async def _funding_rate_collector(self):
        """Collect funding rates from multiple exchanges."""
        while self.is_active:
            try:
                # Collect from all exchanges
                collection_tasks = [
                    self._collect_binance_funding_rates(),
                    self._collect_bybit_funding_rates(),
                    self._collect_okex_funding_rates()
                ]
                
                await asyncio.gather(*collection_tasks, return_exceptions=True)
                
                # Sleep for 5 minutes (funding rates don't change frequently)
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in funding rate collector: {e}")
                await asyncio.sleep(60)
    
    async def _collect_binance_funding_rates(self):
        """Collect funding rates from Binance."""
        try:
            if not self.session:
                return
            
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data:
                        symbol = item.get('symbol', '')
                        if symbol in self.symbols:
                            funding_data = FundingRateData(
                                exchange="binance",
                                symbol=symbol,
                                funding_rate=float(item.get('lastFundingRate', 0)),
                                next_funding_time=datetime.fromtimestamp(
                                    int(item.get('nextFundingTime', 0)) / 1000
                                ),
                                mark_price=float(item.get('markPrice', 0)),
                                index_price=float(item.get('indexPrice', 0))
                            )
                            
                            key = f"binance_{symbol}"
                            self.current_rates[key] = funding_data
                            self.funding_history[key].append(funding_data)
                            
        except Exception as e:
            self.logger.error(f"Error collecting Binance funding rates: {e}")
    
    async def _collect_bybit_funding_rates(self):
        """Collect funding rates from Bybit."""
        try:
            if not self.session:
                return
            
            # Bybit API endpoint
            url = "https://api.bybit.com/v2/public/tickers"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('ret_code') == 0:
                        for item in data.get('result', []):
                            symbol = item.get('symbol', '')
                            if symbol in self.symbols:
                                funding_data = FundingRateData(
                                    exchange="bybit",
                                    symbol=symbol,
                                    funding_rate=float(item.get('funding_rate', 0)),
                                    next_funding_time=datetime.fromtimestamp(
                                        int(item.get('next_funding_time', 0))
                                    ),
                                    mark_price=float(item.get('mark_price', 0)),
                                    index_price=float(item.get('index_price', 0))
                                )
                                
                                key = f"bybit_{symbol}"
                                self.current_rates[key] = funding_data
                                self.funding_history[key].append(funding_data)
                                
        except Exception as e:
            self.logger.error(f"Error collecting Bybit funding rates: {e}")
    
    async def _collect_okex_funding_rates(self):
        """Collect funding rates from OKEx."""
        try:
            if not self.session:
                return
            
            # OKEx API endpoint
            url = "https://www.okx.com/api/v5/public/funding-rate"
            
            for symbol in self.symbols:
                # Convert symbol format for OKEx
                okex_symbol = symbol.replace('USDT', '-USDT-SWAP')
                
                params = {'instId': okex_symbol}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('code') == '0' and data.get('data'):
                            item = data['data'][0]
                            
                            funding_data = FundingRateData(
                                exchange="okex",
                                symbol=symbol,
                                funding_rate=float(item.get('fundingRate', 0)),
                                next_funding_time=datetime.fromtimestamp(
                                    int(item.get('nextFundingTime', 0)) / 1000
                                ),
                                mark_price=0.0,  # Not provided directly
                                index_price=0.0  # Not provided directly
                            )
                            
                            key = f"okex_{symbol}"
                            self.current_rates[key] = funding_data
                            self.funding_history[key].append(funding_data)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error collecting OKEx funding rates: {e}")
    
    async def _arbitrage_detector(self):
        """Detect funding rate arbitrage opportunities."""
        while self.is_active:
            try:
                opportunities = []
                
                for symbol in self.symbols:
                    # Get rates for this symbol across exchanges
                    symbol_rates = {}
                    for exchange in self.exchanges:
                        key = f"{exchange}_{symbol}"
                        if key in self.current_rates:
                            symbol_rates[exchange] = self.current_rates[key]
                    
                    if len(symbol_rates) >= 2:
                        # Find arbitrage opportunities
                        symbol_opportunities = await self._find_arbitrage_opportunities(
                            symbol, symbol_rates
                        )
                        opportunities.extend(symbol_opportunities)
                
                # Update opportunities list
                self.arbitrage_opportunities = opportunities
                
                # Log significant opportunities
                for opp in opportunities:
                    if opp.estimated_profit_annual > 0.1:  # 10% annual profit
                        self.logger.info(
                            f"Funding arbitrage opportunity: {opp.symbol} "
                            f"{opp.rate_differential:.4f} "
                            f"({opp.estimated_profit_annual:.1%} annual)"
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in arbitrage detector: {e}")
                await asyncio.sleep(60)
    
    async def _find_arbitrage_opportunities(self, symbol: str, 
                                         rates: Dict[str, FundingRateData]) -> List[FundingArbitrageOpportunity]:
        """Find arbitrage opportunities for a specific symbol."""
        opportunities = []
        
        try:
            exchanges = list(rates.keys())
            
            # Compare all pairs of exchanges
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange1, exchange2 = exchanges[i], exchanges[j]
                    rate1 = rates[exchange1].funding_rate
                    rate2 = rates[exchange2].funding_rate
                    
                    rate_diff = abs(rate1 - rate2)
                    
                    if rate_diff > self.min_arbitrage_threshold:
                        # Determine long and short sides
                        if rate1 > rate2:
                            long_exchange = exchange2  # Pay lower funding
                            short_exchange = exchange1  # Receive higher funding
                            net_rate = rate1 - rate2
                        else:
                            long_exchange = exchange1
                            short_exchange = exchange2
                            net_rate = rate2 - rate1
                        
                        # Calculate profit estimates
                        profit_8h = net_rate  # One funding period
                        profit_annual = net_rate * 3 * 365  # 3 periods per day
                        
                        # Calculate risk score
                        risk_score = await self._calculate_arbitrage_risk(
                            symbol, long_exchange, short_exchange, rates
                        )
                        
                        # Calculate confidence
                        confidence = await self._calculate_arbitrage_confidence(
                            symbol, long_exchange, short_exchange
                        )
                        
                        # Estimate position sizes
                        min_size, max_size = await self._estimate_position_sizes(
                            symbol, long_exchange, short_exchange
                        )
                        
                        opportunity = FundingArbitrageOpportunity(
                            long_exchange=long_exchange,
                            short_exchange=short_exchange,
                            symbol=symbol,
                            rate_differential=net_rate,
                            estimated_profit_8h=profit_8h,
                            estimated_profit_annual=profit_annual,
                            risk_score=risk_score,
                            confidence=confidence,
                            min_position_size=min_size,
                            max_position_size=max_size
                        )
                        
                        opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities for {symbol}: {e}")
        
        return opportunities
    
    async def _calculate_arbitrage_risk(self, symbol: str, long_exchange: str, 
                                      short_exchange: str, rates: Dict[str, FundingRateData]) -> float:
        """Calculate risk score for arbitrage opportunity."""
        try:
            risk_factors = []
            
            # Historical volatility of rate differential
            diff_history = []
            for i in range(min(50, len(self.funding_history[f"{long_exchange}_{symbol}"]))):
                if (len(self.funding_history[f"{long_exchange}_{symbol}"]) > i and
                    len(self.funding_history[f"{short_exchange}_{symbol}"]) > i):
                    
                    rate1 = list(self.funding_history[f"{long_exchange}_{symbol}"])[-i-1].funding_rate
                    rate2 = list(self.funding_history[f"{short_exchange}_{symbol}"])[-i-1].funding_rate
                    diff_history.append(abs(rate1 - rate2))
            
            if diff_history:
                diff_volatility = np.std(diff_history)
                risk_factors.append(min(diff_volatility * 10, 1.0))
            
            # Exchange reliability (simplified)
            exchange_risk = {
                "binance": 0.1,
                "bybit": 0.2,
                "okex": 0.3,
                "deribit": 0.25
            }
            
            avg_exchange_risk = (
                exchange_risk.get(long_exchange, 0.5) + 
                exchange_risk.get(short_exchange, 0.5)
            ) / 2
            risk_factors.append(avg_exchange_risk)
            
            # Mark price divergence
            if (f"{long_exchange}_{symbol}" in rates and 
                f"{short_exchange}_{symbol}" in rates):
                
                mark1 = rates[f"{long_exchange}_{symbol}"].mark_price
                mark2 = rates[f"{short_exchange}_{symbol}"].mark_price
                
                if mark1 > 0 and mark2 > 0:
                    mark_divergence = abs(mark1 - mark2) / max(mark1, mark2)
                    risk_factors.append(min(mark_divergence * 100, 1.0))
            
            # Overall risk score (0 = low risk, 1 = high risk)
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage risk: {e}")
            return 0.5
    
    async def _calculate_arbitrage_confidence(self, symbol: str, long_exchange: str, 
                                           short_exchange: str) -> float:
        """Calculate confidence in arbitrage opportunity."""
        try:
            confidence_factors = []
            
            # Data availability
            long_key = f"{long_exchange}_{symbol}"
            short_key = f"{short_exchange}_{symbol}"
            
            if long_key in self.funding_history and short_key in self.funding_history:
                data_quality = min(
                    len(self.funding_history[long_key]) / 100,
                    len(self.funding_history[short_key]) / 100,
                    1.0
                )
                confidence_factors.append(data_quality)
            
            # Rate stability (lower volatility = higher confidence)
            for key in [long_key, short_key]:
                if key in self.funding_history and len(self.funding_history[key]) > 5:
                    recent_rates = [r.funding_rate for r in list(self.funding_history[key])[-10:]]
                    rate_stability = 1.0 - min(np.std(recent_rates) * 1000, 1.0)
                    confidence_factors.append(rate_stability)
            
            # Time to next funding (closer = higher confidence)
            if (long_key in self.current_rates and short_key in self.current_rates):
                now = datetime.now()
                next_funding1 = self.current_rates[long_key].next_funding_time
                next_funding2 = self.current_rates[short_key].next_funding_time
                
                time_to_funding = min(
                    (next_funding1 - now).total_seconds() / 3600,
                    (next_funding2 - now).total_seconds() / 3600
                )
                
                # Higher confidence closer to funding time
                time_confidence = max(0, 1 - time_to_funding / 8)
                confidence_factors.append(time_confidence)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage confidence: {e}")
            return 0.5
    
    async def _estimate_position_sizes(self, symbol: str, long_exchange: str, 
                                     short_exchange: str) -> Tuple[float, float]:
        """Estimate optimal position sizes for arbitrage."""
        try:
            # Base position sizes (in USD)
            base_min = 1000   # $1K minimum
            base_max = 50000  # $50K maximum
            
            # Adjust based on symbol liquidity
            liquidity_multipliers = {
                "BTCUSDT": 5.0,
                "ETHUSDT": 3.0,
                "ADAUSDT": 1.5,
                "SOLUSDT": 2.0
            }
            
            multiplier = liquidity_multipliers.get(symbol, 1.0)
            
            min_size = base_min * multiplier
            max_size = base_max * multiplier
            
            return min_size, max_size
            
        except Exception as e:
            self.logger.error(f"Error estimating position sizes: {e}")
            return 1000.0, 10000.0
    
    async def _trend_analyzer(self):
        """Analyze funding rate trends."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        key = f"{exchange}_{symbol}"
                        
                        if (key in self.funding_history and 
                            len(self.funding_history[key]) >= 10):
                            
                            await self._analyze_funding_trend(symbol, exchange)
                
                await asyncio.sleep(600)  # Analyze every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in trend analyzer: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_funding_trend(self, symbol: str, exchange: str):
        """Analyze funding rate trend for specific symbol/exchange."""
        try:
            key = f"{exchange}_{symbol}"
            history = list(self.funding_history[key])
            
            if len(history) < 10:
                return
            
            # Get recent rates
            recent_rates = [r.funding_rate for r in history[-20:]]
            
            # Calculate trend
            x = np.arange(len(recent_rates))
            coeffs = np.polyfit(x, recent_rates, 1)
            trend_slope = coeffs[0]
            
            # Determine trend direction
            if trend_slope > 0.00001:  # 0.001% threshold
                trend_direction = "increasing"
            elif trend_slope < -0.00001:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            # Store trend analysis
            trend_key = f"{exchange}_{symbol}_trend"
            trend_data = {
                "direction": trend_direction,
                "slope": trend_slope,
                "strength": abs(trend_slope) * 10000,  # Strength measure
                "timestamp": datetime.now()
            }
            
            # You could store this in a trends dictionary or use it for predictions
            
        except Exception as e:
            self.logger.error(f"Error analyzing funding trend for {symbol}/{exchange}: {e}")
    
    async def _rate_predictor(self):
        """Predict future funding rates."""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        key = f"{exchange}_{symbol}"
                        
                        if (key in self.funding_history and 
                            len(self.funding_history[key]) >= self.forecast_lookback_periods):
                            
                            forecast = await self._generate_rate_forecast(symbol, exchange)
                            if forecast:
                                self.forecasts[key] = forecast
                
                await asyncio.sleep(1800)  # Generate forecasts every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in rate predictor: {e}")
                await asyncio.sleep(900)
    
    async def _generate_rate_forecast(self, symbol: str, exchange: str) -> Optional[FundingRateForecast]:
        """Generate funding rate forecast."""
        try:
            key = f"{exchange}_{symbol}"
            history = list(self.funding_history[key])
            
            if len(history) < 50:
                return None
            
            # Get historical rates
            rates = [r.funding_rate for r in history[-self.forecast_lookback_periods:]]
            current_rate = rates[-1]
            
            # Simple moving average prediction
            short_ma = np.mean(rates[-10:])
            long_ma = np.mean(rates[-30:])
            
            # Trend-based prediction
            x = np.arange(len(rates[-20:]))
            recent_rates = rates[-20:]
            coeffs = np.polyfit(x, recent_rates, 1)
            trend_prediction = current_rate + coeffs[0] * 3  # 3 periods ahead
            
            # Combine predictions
            predicted_rate = (short_ma * 0.4 + long_ma * 0.3 + trend_prediction * 0.3)
            
            # Determine trend
            if predicted_rate > current_rate * 1.1:
                trend = "increasing"
            elif predicted_rate < current_rate * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Calculate confidence based on prediction stability
            predictions = [short_ma, long_ma, trend_prediction]
            prediction_std = np.std(predictions)
            confidence = max(0, 1 - prediction_std / abs(current_rate)) if current_rate != 0 else 0.5
            
            # Contributing factors
            factors = {
                "short_term_momentum": short_ma - current_rate,
                "long_term_trend": long_ma - current_rate,
                "recent_volatility": np.std(rates[-10:]),
                "price_correlation": 0.0  # Would calculate correlation with price movements
            }
            
            forecast = FundingRateForecast(
                symbol=symbol,
                exchange=exchange,
                current_rate=current_rate,
                predicted_next_rate=predicted_rate,
                predicted_trend=trend,
                confidence=confidence,
                forecast_horizon_hours=8,  # Next funding period
                factors=factors
            )
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating rate forecast for {symbol}/{exchange}: {e}")
            return None
    
    async def get_current_rates(self, symbol: Optional[str] = None) -> Dict[str, FundingRateData]:
        """Get current funding rates."""
        if symbol:
            return {k: v for k, v in self.current_rates.items() if symbol in k}
        return self.current_rates.copy()
    
    async def get_arbitrage_opportunities(self, min_profit_annual: float = 0.05) -> List[FundingArbitrageOpportunity]:
        """Get funding arbitrage opportunities above threshold."""
        return [
            opp for opp in self.arbitrage_opportunities 
            if opp.estimated_profit_annual >= min_profit_annual
        ]
    
    async def get_rate_forecasts(self, symbol: Optional[str] = None) -> Dict[str, FundingRateForecast]:
        """Get funding rate forecasts."""
        if symbol:
            return {k: v for k, v in self.forecasts.items() if symbol in k}
        return self.forecasts.copy()
    
    async def get_historical_rates(self, symbol: str, exchange: str, 
                                 periods: int = 100) -> List[FundingRateData]:
        """Get historical funding rates."""
        key = f"{exchange}_{symbol}"
        if key in self.funding_history:
            return list(self.funding_history[key])[-periods:]
        return []
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "is_active": self.is_active,
            "current_rates_count": len(self.current_rates),
            "arbitrage_opportunities": len(self.arbitrage_opportunities),
            "forecasts_count": len(self.forecasts),
            "symbols": self.symbols,
            "exchanges": self.exchanges,
            "best_opportunity": max(
                self.arbitrage_opportunities,
                key=lambda x: x.estimated_profit_annual,
                default=None
            )
        }