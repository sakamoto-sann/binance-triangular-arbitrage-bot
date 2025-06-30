"""
Cross-Exchange Funding Rate Arbitrage System
Monitors and capitalizes on funding rate differences across multiple exchanges
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class FundingRateData:
    """Data class for funding rate information"""
    exchange: str
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    mark_price: float
    index_price: float
    timestamp: datetime

@dataclass
class ArbitrageOpportunity:
    """Data class for arbitrage opportunity"""
    symbol: str
    long_exchange: str
    short_exchange: str
    long_funding_rate: float
    short_funding_rate: float
    rate_difference: float
    annualized_return: float
    risk_score: float
    max_position_size: float
    estimated_profit: float
    confidence: float

@dataclass
class ArbitragePosition:
    """Data class for active arbitrage position"""
    opportunity_id: str
    symbol: str
    long_exchange: str
    short_exchange: str
    long_quantity: float
    short_quantity: float
    long_entry_price: float
    short_entry_price: float
    entry_time: datetime
    expected_funding_payments: List[float]
    actual_funding_payments: List[float]
    unrealized_pnl: float
    realized_pnl: float
    status: str  # 'active', 'closing', 'closed'

class FundingRateArbitrage:
    """
    Professional funding rate arbitrage system that monitors and capitalizes
    on funding rate differences across multiple exchanges.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the funding rate arbitrage system"""
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Exchange configuration
        self.exchanges = self.config.get('exchanges', ['binance', 'okx', 'bybit'])
        self.funding_threshold = self.config.get('funding_threshold', 0.01)  # 1% annualized difference
        
        # Position limits (USDT)
        self.position_limits = self.config.get('position_limits', {
            'binance': 1000000,
            'okx': 500000,
            'bybit': 500000
        })
        
        # Risk management
        self.max_total_exposure = self.config.get('max_total_exposure', 2000000)
        self.max_single_position = self.config.get('max_single_position', 100000)
        self.min_rate_difference = self.config.get('min_rate_difference', 0.005)  # 0.5% annualized
        
        # Monitoring configuration
        self.symbols_to_monitor = self.config.get('symbols_to_monitor', [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT'
        ])
        
        # Funding rate data cache
        self.funding_data_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Active positions tracking
        self.active_positions = {}  # opportunity_id -> ArbitragePosition
        self.position_history = []
        
        # Exchange API configurations (placeholders for actual API clients)
        self.exchange_clients = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_opportunities_found': 0,
            'total_positions_opened': 0,
            'total_funding_collected': 0.0,
            'success_rate': 0.0,
            'average_holding_time': 0.0,
            'sharpe_ratio': 0.0
        }

    async def scan_funding_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan for profitable funding rate arbitrage opportunities across all exchanges.
        Returns a ranked list of opportunities.
        """
        try:
            self.logger.info("Scanning for funding rate arbitrage opportunities")
            
            # Fetch funding rates from all exchanges
            funding_data = await self._fetch_all_funding_rates()
            
            if not funding_data:
                self.logger.warning("No funding rate data available")
                return []
            
            # Find arbitrage opportunities
            opportunities = []
            
            for symbol in self.symbols_to_monitor:
                symbol_opportunities = self._find_symbol_opportunities(symbol, funding_data)
                opportunities.extend(symbol_opportunities)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)
            ranked_opportunities = self._rank_opportunities(filtered_opportunities)
            
            # Update performance metrics
            self.performance_metrics['total_opportunities_found'] += len(ranked_opportunities)
            
            self.logger.info(f"Found {len(ranked_opportunities)} profitable funding arbitrage opportunities")
            
            return ranked_opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to scan funding opportunities: {e}")
            return []

    async def _fetch_all_funding_rates(self) -> Dict[str, Dict[str, FundingRateData]]:
        """Fetch funding rates from all configured exchanges"""
        try:
            funding_data = {}
            
            # Create tasks for each exchange
            tasks = []
            for exchange in self.exchanges:
                task = self._fetch_exchange_funding_rates(exchange)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                exchange = self.exchanges[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to fetch funding rates from {exchange}: {result}")
                    continue
                
                if result:
                    funding_data[exchange] = result
            
            return funding_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch all funding rates: {e}")
            return {}

    async def _fetch_exchange_funding_rates(self, exchange: str) -> Dict[str, FundingRateData]:
        """Fetch funding rates from a specific exchange"""
        try:
            cache_key = f'funding_rates_{exchange}'
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                return self.funding_data_cache[cache_key]['data']
            
            funding_rates = {}
            
            if exchange == 'binance':
                funding_rates = await self._fetch_binance_funding_rates()
            elif exchange == 'okx':
                funding_rates = await self._fetch_okx_funding_rates()
            elif exchange == 'bybit':
                funding_rates = await self._fetch_bybit_funding_rates()
            else:
                self.logger.warning(f"Unsupported exchange: {exchange}")
                return {}
            
            # Cache the results
            self._update_cache(cache_key, funding_rates)
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to fetch funding rates from {exchange}: {e}")
            return {}

    async def _fetch_binance_funding_rates(self) -> Dict[str, FundingRateData]:
        """Fetch funding rates from Binance"""
        try:
            funding_rates = {}
            
            # Use the existing Binance client if available
            if hasattr(self, 'binance_client') and self.binance_client:
                for symbol in self.symbols_to_monitor:
                    try:
                        # Get current funding rate
                        funding_info = self.binance_client.futures_funding_rate(symbol=symbol, limit=1)
                        if funding_info:
                            funding_rate = float(funding_info[0]['fundingRate'])
                            funding_time = datetime.fromtimestamp(int(funding_info[0]['fundingTime']) / 1000)
                            
                            # Get mark price
                            mark_price_info = self.binance_client.futures_mark_price(symbol=symbol)
                            mark_price = float(mark_price_info['markPrice'])
                            index_price = float(mark_price_info.get('indexPrice', mark_price))
                            
                            funding_rates[symbol] = FundingRateData(
                                exchange='binance',
                                symbol=symbol,
                                funding_rate=funding_rate,
                                next_funding_time=funding_time + timedelta(hours=8),  # Binance funding every 8 hours
                                mark_price=mark_price,
                                index_price=index_price,
                                timestamp=datetime.now()
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to get Binance funding rate for {symbol}: {e}")
                        continue
            else:
                # Fallback to REST API calls
                funding_rates = await self._fetch_binance_funding_rates_api()
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Binance funding rates: {e}")
            return {}

    async def _fetch_binance_funding_rates_api(self) -> Dict[str, FundingRateData]:
        """Fetch Binance funding rates via REST API"""
        try:
            funding_rates = {}
            base_url = "https://fapi.binance.com"
            
            async with aiohttp.ClientSession() as session:
                # Get all funding rates
                url = f"{base_url}/fapi/v1/premiumIndex"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data:
                            symbol = item['symbol']
                            if symbol in self.symbols_to_monitor:
                                funding_rate = float(item['lastFundingRate'])
                                mark_price = float(item['markPrice'])
                                index_price = float(item['indexPrice'])
                                next_funding_time = datetime.fromtimestamp(int(item['nextFundingTime']) / 1000)
                                
                                funding_rates[symbol] = FundingRateData(
                                    exchange='binance',
                                    symbol=symbol,
                                    funding_rate=funding_rate,
                                    next_funding_time=next_funding_time,
                                    mark_price=mark_price,
                                    index_price=index_price,
                                    timestamp=datetime.now()
                                )
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Binance funding rates via API: {e}")
            return {}

    async def _fetch_okx_funding_rates(self) -> Dict[str, FundingRateData]:
        """Fetch funding rates from OKX"""
        try:
            funding_rates = {}
            base_url = "https://www.okx.com"
            
            async with aiohttp.ClientSession() as session:
                for symbol in self.symbols_to_monitor:
                    try:
                        # Convert symbol format (BTCUSDT -> BTC-USDT-SWAP)
                        okx_symbol = self._convert_to_okx_symbol(symbol)
                        
                        # Get funding rate
                        url = f"{base_url}/api/v5/public/funding-rate"
                        params = {'instId': okx_symbol}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data['code'] == '0' and data['data']:
                                    item = data['data'][0]
                                    funding_rate = float(item['fundingRate'])
                                    next_funding_time = datetime.fromtimestamp(int(item['nextFundingTime']) / 1000)
                                    
                                    # Get mark price
                                    mark_url = f"{base_url}/api/v5/public/mark-price"
                                    mark_params = {'instType': 'SWAP', 'instId': okx_symbol}
                                    
                                    async with session.get(mark_url, params=mark_params) as mark_response:
                                        mark_price = 0.0
                                        index_price = 0.0
                                        
                                        if mark_response.status == 200:
                                            mark_data = await mark_response.json()
                                            if mark_data['code'] == '0' and mark_data['data']:
                                                mark_price = float(mark_data['data'][0]['markPx'])
                                                index_price = float(mark_data['data'][0].get('idxPx', mark_price))
                                    
                                    funding_rates[symbol] = FundingRateData(
                                        exchange='okx',
                                        symbol=symbol,
                                        funding_rate=funding_rate,
                                        next_funding_time=next_funding_time,
                                        mark_price=mark_price,
                                        index_price=index_price,
                                        timestamp=datetime.now()
                                    )
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to get OKX funding rate for {symbol}: {e}")
                        continue
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OKX funding rates: {e}")
            return {}

    async def _fetch_bybit_funding_rates(self) -> Dict[str, FundingRateData]:
        """Fetch funding rates from Bybit"""
        try:
            funding_rates = {}
            base_url = "https://api.bybit.com"
            
            async with aiohttp.ClientSession() as session:
                for symbol in self.symbols_to_monitor:
                    try:
                        # Convert symbol format (BTCUSDT -> BTCUSDT for Bybit)
                        bybit_symbol = symbol
                        
                        # Get funding rate
                        url = f"{base_url}/v2/public/tickers"
                        params = {'symbol': bybit_symbol}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data['ret_code'] == 0 and data['result']:
                                    result = data['result'][0] if isinstance(data['result'], list) else data['result']
                                    
                                    funding_rate = float(result.get('funding_rate', 0))
                                    mark_price = float(result.get('mark_price', 0))
                                    index_price = float(result.get('index_price', mark_price))
                                    
                                    # Bybit funding every 8 hours
                                    next_funding_time = datetime.now().replace(minute=0, second=0, microsecond=0)
                                    next_funding_time += timedelta(hours=8 - (next_funding_time.hour % 8))
                                    
                                    funding_rates[symbol] = FundingRateData(
                                        exchange='bybit',
                                        symbol=symbol,
                                        funding_rate=funding_rate,
                                        next_funding_time=next_funding_time,
                                        mark_price=mark_price,
                                        index_price=index_price,
                                        timestamp=datetime.now()
                                    )
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to get Bybit funding rate for {symbol}: {e}")
                        continue
            
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Bybit funding rates: {e}")
            return {}

    def _convert_to_okx_symbol(self, symbol: str) -> str:
        """Convert Binance symbol format to OKX format"""
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}-USDT-SWAP"
        return symbol

    def _find_symbol_opportunities(self, symbol: str, 
                                 funding_data: Dict[str, Dict[str, FundingRateData]]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a specific symbol"""
        try:
            opportunities = []
            
            # Get funding rates for this symbol across exchanges
            symbol_rates = {}
            for exchange, exchange_data in funding_data.items():
                if symbol in exchange_data:
                    symbol_rates[exchange] = exchange_data[symbol]
            
            if len(symbol_rates) < 2:
                return opportunities  # Need at least 2 exchanges
            
            # Find all pairs with significant rate differences
            exchanges = list(symbol_rates.keys())
            
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange1 = exchanges[i]
                    exchange2 = exchanges[j]
                    
                    rate1 = symbol_rates[exchange1].funding_rate
                    rate2 = symbol_rates[exchange2].funding_rate
                    
                    rate_difference = abs(rate1 - rate2)
                    
                    if rate_difference >= self.min_rate_difference:
                        # Determine which exchange to go long/short
                        if rate1 > rate2:
                            # Go long on exchange2 (pay less funding), short on exchange1 (receive more funding)
                            long_exchange = exchange2
                            short_exchange = exchange1
                            long_funding_rate = rate2
                            short_funding_rate = rate1
                        else:
                            # Go long on exchange1, short on exchange2
                            long_exchange = exchange1
                            short_exchange = exchange2
                            long_funding_rate = rate1
                            short_funding_rate = rate2
                        
                        # Calculate opportunity metrics
                        opportunity = self._calculate_opportunity_metrics(
                            symbol, long_exchange, short_exchange,
                            long_funding_rate, short_funding_rate,
                            symbol_rates[long_exchange], symbol_rates[short_exchange]
                        )
                        
                        if opportunity and opportunity.confidence > 0.5:
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to find opportunities for {symbol}: {e}")
            return []

    def _calculate_opportunity_metrics(self, symbol: str, long_exchange: str, short_exchange: str,
                                     long_funding_rate: float, short_funding_rate: float,
                                     long_data: FundingRateData, short_data: FundingRateData) -> Optional[ArbitrageOpportunity]:
        """Calculate detailed metrics for an arbitrage opportunity"""
        try:
            # Calculate rate difference and annualized return
            rate_difference = short_funding_rate - long_funding_rate
            
            # Funding payments typically occur every 8 hours (3 times per day)
            annualized_return = rate_difference * 3 * 365
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(symbol, long_exchange, short_exchange, long_data, short_data)
            
            # Calculate maximum position size
            max_position_size = self._calculate_max_position_size(symbol, long_exchange, short_exchange)
            
            # Estimate profit for a reasonable position size
            position_size = min(max_position_size, self.max_single_position)
            estimated_profit = position_size * rate_difference
            
            # Calculate confidence score
            confidence = self._calculate_opportunity_confidence(
                rate_difference, risk_score, long_data, short_data
            )
            
            return ArbitrageOpportunity(
                symbol=symbol,
                long_exchange=long_exchange,
                short_exchange=short_exchange,
                long_funding_rate=long_funding_rate,
                short_funding_rate=short_funding_rate,
                rate_difference=rate_difference,
                annualized_return=annualized_return,
                risk_score=risk_score,
                max_position_size=max_position_size,
                estimated_profit=estimated_profit,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate opportunity metrics: {e}")
            return None

    def _calculate_risk_score(self, symbol: str, long_exchange: str, short_exchange: str,
                            long_data: FundingRateData, short_data: FundingRateData) -> float:
        """Calculate risk score for the arbitrage opportunity"""
        try:
            risk_factors = []
            
            # Price divergence risk
            price_divergence = abs(long_data.mark_price - short_data.mark_price) / np.mean([long_data.mark_price, short_data.mark_price])
            risk_factors.append(min(1.0, price_divergence * 100))  # Scale and cap
            
            # Exchange risk (subjective scoring)
            exchange_risk_scores = {
                'binance': 0.1,  # Low risk
                'okx': 0.2,      # Medium risk
                'bybit': 0.3     # Higher risk
            }
            
            long_exchange_risk = exchange_risk_scores.get(long_exchange, 0.5)
            short_exchange_risk = exchange_risk_scores.get(short_exchange, 0.5)
            risk_factors.append((long_exchange_risk + short_exchange_risk) / 2)
            
            # Funding rate volatility risk (higher absolute rates = higher risk)
            avg_funding_rate = (abs(long_data.funding_rate) + abs(short_data.funding_rate)) / 2
            volatility_risk = min(1.0, avg_funding_rate * 100)  # Scale
            risk_factors.append(volatility_risk)
            
            # Symbol-specific risk
            symbol_risk_scores = {
                'BTCUSDT': 0.1,  # Most stable
                'ETHUSDT': 0.2,
                'BNBUSDT': 0.3,
                'ADAUSDT': 0.4,
                'SOLUSDT': 0.4
            }
            symbol_risk = symbol_risk_scores.get(symbol, 0.5)
            risk_factors.append(symbol_risk)
            
            # Return average risk score
            return np.mean(risk_factors)
            
        except Exception:
            return 0.5  # Default medium risk

    def _calculate_max_position_size(self, symbol: str, long_exchange: str, short_exchange: str) -> float:
        """Calculate maximum position size for the arbitrage"""
        try:
            # Get position limits for both exchanges
            long_limit = self.position_limits.get(long_exchange, 100000)
            short_limit = self.position_limits.get(short_exchange, 100000)
            
            # Use the smaller limit
            exchange_limit = min(long_limit, short_limit)
            
            # Apply symbol-specific limits
            symbol_multipliers = {
                'BTCUSDT': 1.0,   # No reduction
                'ETHUSDT': 0.8,   # 20% reduction
                'BNBUSDT': 0.6,   # 40% reduction
                'ADAUSDT': 0.4,   # 60% reduction
                'SOLUSDT': 0.4    # 60% reduction
            }
            
            symbol_multiplier = symbol_multipliers.get(symbol, 0.3)
            symbol_limit = exchange_limit * symbol_multiplier
            
            # Apply global limits
            max_position = min(symbol_limit, self.max_single_position)
            
            return max_position
            
        except Exception:
            return self.max_single_position * 0.1  # Conservative fallback

    def _calculate_opportunity_confidence(self, rate_difference: float, risk_score: float,
                                        long_data: FundingRateData, short_data: FundingRateData) -> float:
        """Calculate confidence score for the opportunity"""
        try:
            confidence_factors = []
            
            # Rate difference factor (higher difference = higher confidence)
            rate_confidence = min(1.0, rate_difference / 0.02)  # Normalize to 2% max
            confidence_factors.append(rate_confidence)
            
            # Risk factor (lower risk = higher confidence)
            risk_confidence = 1.0 - risk_score
            confidence_factors.append(risk_confidence)
            
            # Data freshness factor
            data_age = max(
                (datetime.now() - long_data.timestamp).total_seconds(),
                (datetime.now() - short_data.timestamp).total_seconds()
            )
            freshness_confidence = max(0.1, 1.0 - (data_age / 600))  # 10-minute decay
            confidence_factors.append(freshness_confidence)
            
            # Price consistency factor
            price_consistency = 1.0 - abs(long_data.mark_price - short_data.mark_price) / np.mean([long_data.mark_price, short_data.mark_price])
            confidence_factors.append(max(0.1, price_consistency))
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.3  # Low confidence fallback

    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on various criteria"""
        try:
            filtered = []
            
            for opp in opportunities:
                # Minimum rate difference filter
                if opp.rate_difference < self.min_rate_difference:
                    continue
                
                # Minimum annualized return filter
                if opp.annualized_return < 0.05:  # 5% minimum annualized return
                    continue
                
                # Maximum risk filter
                if opp.risk_score > 0.8:  # Reject very high risk opportunities
                    continue
                
                # Minimum confidence filter
                if opp.confidence < 0.4:
                    continue
                
                # Check if we already have a position in this symbol pair
                existing_position = self._check_existing_position(opp.symbol, opp.long_exchange, opp.short_exchange)
                if existing_position:
                    continue
                
                filtered.append(opp)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Failed to filter opportunities: {e}")
            return opportunities

    def _rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by attractiveness"""
        try:
            if not opportunities:
                return opportunities
            
            # Calculate ranking score for each opportunity
            for opp in opportunities:
                # Score components
                return_score = opp.annualized_return * 10  # Scale annualized return
                risk_score = (1.0 - opp.risk_score) * 5    # Invert risk (lower risk = higher score)
                confidence_score = opp.confidence * 3
                size_score = min(2.0, opp.max_position_size / 50000)  # Position size benefit
                
                # Combined score
                opp.ranking_score = return_score + risk_score + confidence_score + size_score
            
            # Sort by ranking score (descending)
            opportunities.sort(key=lambda x: x.ranking_score, reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to rank opportunities: {e}")
            return opportunities

    def _check_existing_position(self, symbol: str, long_exchange: str, short_exchange: str) -> bool:
        """Check if we already have a position for this symbol/exchange pair"""
        try:
            for position in self.active_positions.values():
                if (position.symbol == symbol and 
                    position.long_exchange == long_exchange and 
                    position.short_exchange == short_exchange and
                    position.status == 'active'):
                    return True
            return False
        except Exception:
            return False

    async def execute_funding_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """
        Execute cross-exchange funding arbitrage.
        Goes long on the exchange with negative/lower funding and short on the exchange with positive/higher funding.
        """
        try:
            self.logger.info(f"Executing funding arbitrage for {opportunity.symbol}: "
                           f"Long {opportunity.long_exchange}, Short {opportunity.short_exchange}")
            
            # Validate opportunity
            if not self._validate_opportunity(opportunity):
                return {'status': 'failed', 'error': 'Opportunity validation failed'}
            
            # Calculate position size
            position_size = self._calculate_execution_position_size(opportunity)
            
            if position_size < 1000:  # Minimum position size
                return {'status': 'failed', 'error': 'Position size too small'}
            
            # Execute positions simultaneously
            execution_result = await self._execute_simultaneous_positions(opportunity, position_size)
            
            if execution_result['status'] == 'success':
                # Create and track position
                position = self._create_arbitrage_position(opportunity, execution_result, position_size)
                opportunity_id = f"{opportunity.symbol}_{opportunity.long_exchange}_{opportunity.short_exchange}_{int(time.time())}"
                self.active_positions[opportunity_id] = position
                
                # Update performance metrics
                self.performance_metrics['total_positions_opened'] += 1
                
                self.logger.info(f"Successfully executed funding arbitrage position: {opportunity_id}")
                
                return {
                    'status': 'success',
                    'opportunity_id': opportunity_id,
                    'position': {
                        'symbol': position.symbol,
                        'long_exchange': position.long_exchange,
                        'short_exchange': position.short_exchange,
                        'position_size': position_size,
                        'long_entry_price': position.long_entry_price,
                        'short_entry_price': position.short_entry_price,
                        'expected_funding_rate': opportunity.rate_difference,
                        'estimated_profit': opportunity.estimated_profit
                    },
                    'execution_details': execution_result
                }
            else:
                return {
                    'status': 'failed',
                    'error': execution_result.get('error', 'Execution failed'),
                    'details': execution_result
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute funding arbitrage: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate the arbitrage opportunity before execution"""
        try:
            # Check minimum requirements
            if opportunity.rate_difference < self.min_rate_difference:
                self.logger.warning("Rate difference below minimum threshold")
                return False
            
            if opportunity.confidence < 0.5:
                self.logger.warning("Opportunity confidence too low")
                return False
            
            if opportunity.risk_score > 0.7:
                self.logger.warning("Risk score too high")
                return False
            
            # Check position limits
            current_exposure = self._calculate_current_exposure()
            if current_exposure + opportunity.max_position_size > self.max_total_exposure:
                self.logger.warning("Total exposure limit would be exceeded")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Opportunity validation failed: {e}")
            return False

    def _calculate_execution_position_size(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate optimal position size for execution"""
        try:
            # Start with maximum position size
            base_size = opportunity.max_position_size
            
            # Adjust for confidence
            confidence_adjusted_size = base_size * opportunity.confidence
            
            # Adjust for risk
            risk_adjusted_size = confidence_adjusted_size * (1.0 - opportunity.risk_score * 0.5)
            
            # Check available exposure
            current_exposure = self._calculate_current_exposure()
            available_exposure = self.max_total_exposure - current_exposure
            
            # Final position size
            final_size = min(risk_adjusted_size, available_exposure, self.max_single_position)
            
            # Round to reasonable precision
            return round(final_size, 2)
            
        except Exception:
            return min(10000, opportunity.max_position_size * 0.1)  # Conservative fallback

    def _calculate_current_exposure(self) -> float:
        """Calculate current total exposure across all positions"""
        try:
            total_exposure = 0.0
            
            for position in self.active_positions.values():
                if position.status == 'active':
                    position_exposure = abs(position.long_quantity * position.long_entry_price)
                    total_exposure += position_exposure
            
            return total_exposure
            
        except Exception:
            return 0.0

    async def _execute_simultaneous_positions(self, opportunity: ArbitrageOpportunity, 
                                            position_size: float) -> Dict[str, Any]:
        """Execute long and short positions simultaneously"""
        try:
            # Calculate quantities for each exchange
            long_quantity = position_size / 2  # Half position size for long
            short_quantity = position_size / 2  # Half position size for short
            
            # Execute both positions concurrently
            long_task = self._execute_exchange_position(
                opportunity.long_exchange, opportunity.symbol, 'long', long_quantity
            )
            short_task = self._execute_exchange_position(
                opportunity.short_exchange, opportunity.symbol, 'short', short_quantity
            )
            
            long_result, short_result = await asyncio.gather(long_task, short_task, return_exceptions=True)
            
            # Check results
            if isinstance(long_result, Exception) or isinstance(short_result, Exception):
                # Handle partial execution
                await self._handle_partial_execution(long_result, short_result, opportunity)
                return {
                    'status': 'failed',
                    'error': 'Partial execution - positions closed',
                    'long_result': str(long_result) if isinstance(long_result, Exception) else long_result,
                    'short_result': str(short_result) if isinstance(short_result, Exception) else short_result
                }
            
            if long_result['status'] == 'success' and short_result['status'] == 'success':
                return {
                    'status': 'success',
                    'long_execution': long_result,
                    'short_execution': short_result,
                    'total_quantity': long_quantity + short_quantity
                }
            else:
                # Handle failed executions
                await self._handle_failed_execution(long_result, short_result, opportunity)
                return {
                    'status': 'failed',
                    'error': 'One or both executions failed',
                    'long_result': long_result,
                    'short_result': short_result
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute simultaneous positions: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def _execute_exchange_position(self, exchange: str, symbol: str, 
                                       side: str, quantity: float) -> Dict[str, Any]:
        """Execute a position on a specific exchange (simulation)"""
        try:
            # This is a simulation - in production, this would place actual orders
            
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Simulate price (would get real price from exchange)
            base_price = 50000.0 if symbol == 'BTCUSDT' else 3000.0  # Simplified
            
            # Simulate slippage
            slippage = np.random.uniform(-0.001, 0.001)  # ±0.1% slippage
            execution_price = base_price * (1 + slippage)
            
            # Simulate execution fee
            fee_rate = 0.0004  # 0.04% fee
            execution_fee = quantity * execution_price * fee_rate
            
            return {
                'status': 'success',
                'exchange': exchange,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'execution_price': execution_price,
                'execution_fee': execution_fee,
                'slippage': slippage,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'exchange': exchange,
                'symbol': symbol
            }

    async def _handle_partial_execution(self, long_result: Any, short_result: Any, 
                                      opportunity: ArbitrageOpportunity):
        """Handle partial execution scenario"""
        try:
            # Close any successful positions
            if not isinstance(long_result, Exception) and long_result.get('status') == 'success':
                await self._close_position(opportunity.long_exchange, opportunity.symbol, 'long')
            
            if not isinstance(short_result, Exception) and short_result.get('status') == 'success':
                await self._close_position(opportunity.short_exchange, opportunity.symbol, 'short')
            
            self.logger.warning(f"Handled partial execution for {opportunity.symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle partial execution: {e}")

    async def _handle_failed_execution(self, long_result: Dict, short_result: Dict, 
                                     opportunity: ArbitrageOpportunity):
        """Handle failed execution scenario"""
        try:
            # Close any successful positions
            if long_result.get('status') == 'success':
                await self._close_position(opportunity.long_exchange, opportunity.symbol, 'long')
            
            if short_result.get('status') == 'success':
                await self._close_position(opportunity.short_exchange, opportunity.symbol, 'short')
            
            self.logger.warning(f"Handled failed execution for {opportunity.symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle failed execution: {e}")

    async def _close_position(self, exchange: str, symbol: str, side: str):
        """Close a position on a specific exchange (simulation)"""
        try:
            # Simulation of position closing
            self.logger.info(f"Closing {side} position for {symbol} on {exchange}")
            await asyncio.sleep(0.1)  # Simulate execution time
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")

    def _create_arbitrage_position(self, opportunity: ArbitrageOpportunity, 
                                 execution_result: Dict, position_size: float) -> ArbitragePosition:
        """Create arbitrage position tracking object"""
        try:
            long_execution = execution_result['long_execution']
            short_execution = execution_result['short_execution']
            
            return ArbitragePosition(
                opportunity_id=f"{opportunity.symbol}_{int(time.time())}",
                symbol=opportunity.symbol,
                long_exchange=opportunity.long_exchange,
                short_exchange=opportunity.short_exchange,
                long_quantity=long_execution['quantity'],
                short_quantity=short_execution['quantity'],
                long_entry_price=long_execution['execution_price'],
                short_entry_price=short_execution['execution_price'],
                entry_time=datetime.now(),
                expected_funding_payments=[opportunity.rate_difference],
                actual_funding_payments=[],
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                status='active'
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create arbitrage position: {e}")
            raise

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.funding_data_cache:
            return False
        
        cache_time = self.funding_data_cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl

    def _update_cache(self, cache_key: str, data: Any):
        """Update cache with new data"""
        self.funding_data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    async def monitor_positions(self) -> Dict[str, Any]:
        """Monitor and manage active arbitrage positions"""
        try:
            monitoring_results = {}
            
            for opportunity_id, position in self.active_positions.items():
                if position.status != 'active':
                    continue
                
                # Update position metrics
                position_update = await self._update_position_metrics(position)
                
                # Check for closing conditions
                should_close = self._should_close_position(position, position_update)
                
                if should_close:
                    close_result = await self._close_arbitrage_position(position)
                    monitoring_results[opportunity_id] = {
                        'action': 'closed',
                        'reason': should_close,
                        'result': close_result
                    }
                else:
                    monitoring_results[opportunity_id] = {
                        'action': 'monitoring',
                        'metrics': position_update
                    }
            
            return {
                'status': 'completed',
                'active_positions': len([p for p in self.active_positions.values() if p.status == 'active']),
                'monitoring_results': monitoring_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _update_position_metrics(self, position: ArbitragePosition) -> Dict[str, Any]:
        """Update metrics for an active position"""
        try:
            # Get current prices
            long_price = await self._get_current_price(position.long_exchange, position.symbol)
            short_price = await self._get_current_price(position.short_exchange, position.symbol)
            
            if long_price is None or short_price is None:
                return {'error': 'Unable to get current prices'}
            
            # Calculate unrealized PnL
            long_pnl = (long_price - position.long_entry_price) * position.long_quantity
            short_pnl = (position.short_entry_price - short_price) * position.short_quantity
            total_unrealized_pnl = long_pnl + short_pnl
            
            # Update position
            position.unrealized_pnl = total_unrealized_pnl
            
            # Calculate holding time
            holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600  # hours
            
            return {
                'long_price': long_price,
                'short_price': short_price,
                'long_pnl': long_pnl,
                'short_pnl': short_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'holding_time_hours': holding_time,
                'funding_payments_collected': len(position.actual_funding_payments)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update position metrics: {e}")
            return {'error': str(e)}

    async def _get_current_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get current price from exchange"""
        try:
            # Simulation - would get real price from exchange API
            base_prices = {
                'BTCUSDT': 50000.0,
                'ETHUSDT': 3000.0,
                'BNBUSDT': 400.0,
                'ADAUSDT': 1.0,
                'SOLUSDT': 100.0
            }
            
            base_price = base_prices.get(symbol, 1000.0)
            # Add some random variation
            variation = np.random.uniform(-0.002, 0.002)  # ±0.2%
            return base_price * (1 + variation)
            
        except Exception:
            return None

    def _should_close_position(self, position: ArbitragePosition, metrics: Dict[str, Any]) -> Optional[str]:
        """Determine if a position should be closed"""
        try:
            # Check for profit target
            if position.unrealized_pnl > 1000:  # $1000 profit target
                return 'profit_target_reached'
            
            # Check for stop loss
            if position.unrealized_pnl < -500:  # $500 stop loss
                return 'stop_loss_triggered'
            
            # Check for maximum holding time
            holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
            if holding_time > 168:  # 1 week maximum holding time
                return 'max_holding_time_reached'
            
            # Check for funding rate convergence
            # Would need to fetch current funding rates and compare
            
            return None  # Keep position open
            
        except Exception:
            return 'error_in_monitoring'

    async def _close_arbitrage_position(self, position: ArbitragePosition) -> Dict[str, Any]:
        """Close an arbitrage position"""
        try:
            self.logger.info(f"Closing arbitrage position for {position.symbol}")
            
            # Close both legs simultaneously
            close_long_task = self._close_position(position.long_exchange, position.symbol, 'long')
            close_short_task = self._close_position(position.short_exchange, position.symbol, 'short')
            
            await asyncio.gather(close_long_task, close_short_task)
            
            # Update position status
            position.status = 'closed'
            position.realized_pnl = position.unrealized_pnl
            
            # Update performance metrics
            self.performance_metrics['total_funding_collected'] += position.realized_pnl
            
            # Add to history
            self.position_history.append(position)
            
            return {
                'status': 'success',
                'realized_pnl': position.realized_pnl,
                'holding_time': (datetime.now() - position.entry_time).total_seconds() / 3600
            }
            
        except Exception as e:
            self.logger.error(f"Failed to close arbitrage position: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            active_positions = len([p for p in self.active_positions.values() if p.status == 'active'])
            closed_positions = len(self.position_history)
            
            # Calculate success rate
            if closed_positions > 0:
                profitable_positions = len([p for p in self.position_history if p.realized_pnl > 0])
                success_rate = profitable_positions / closed_positions
            else:
                success_rate = 0.0
            
            # Calculate average holding time
            if self.position_history:
                holding_times = [(datetime.now() - p.entry_time).total_seconds() / 3600 for p in self.position_history]
                avg_holding_time = np.mean(holding_times)
            else:
                avg_holding_time = 0.0
            
            # Update performance metrics
            self.performance_metrics['success_rate'] = success_rate
            self.performance_metrics['average_holding_time'] = avg_holding_time
            
            return {
                'active_positions': active_positions,
                'closed_positions': closed_positions,
                'total_opportunities_found': self.performance_metrics['total_opportunities_found'],
                'total_positions_opened': self.performance_metrics['total_positions_opened'],
                'total_funding_collected': self.performance_metrics['total_funding_collected'],
                'success_rate': success_rate,
                'average_holding_time_hours': avg_holding_time,
                'current_exposure': self._calculate_current_exposure(),
                'position_history': [
                    {
                        'symbol': p.symbol,
                        'realized_pnl': p.realized_pnl,
                        'holding_time_hours': (datetime.now() - p.entry_time).total_seconds() / 3600
                    }
                    for p in self.position_history[-10:]  # Last 10 positions
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }