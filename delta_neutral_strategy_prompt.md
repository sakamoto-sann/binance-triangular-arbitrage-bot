# Delta-Neutral Grid Trading Strategy Implementation Prompt

## Overview
Create a sophisticated trading bot that combines spot grid trading with futures positions to maintain delta neutrality while profiting from funding fees. The system should compound profits and maximize trading volume to achieve VIP 1 status on Binance while strictly adhering to ALL Binance API rules, regulations, and terms of service.

## CRITICAL: Binance API Compliance Framework

### 1. API Rate Limits (MANDATORY COMPLIANCE)

**Spot API Rate Limits (MUST NOT EXCEED)**:
- **Weight-based limits**: 1200 request weight per minute per IP
- **Raw requests**: No more than 6000 requests per 5 minutes per IP
- **Order rate limits**: 
  - 10 orders per second per account
  - 100 orders per 10 seconds per account
  - 200 orders per day per account (can be higher based on account performance)
- **WebSocket connections**: Maximum 5 streams per connection, 1024 connections per IP

**Futures API Rate Limits (MUST NOT EXCEED)**:
- **Weight-based limits**: 2400 request weight per minute per IP
- **Order rate limits**:
  - 300 orders per 10 seconds per account
  - Account-specific limits based on VIP level and trading history
- **Position limits**: Based on risk management and margin requirements

**Implementation Requirements**:
```python
class BinanceAPIRateLimiter:
    def __init__(self):
        # Spot API limits (use 70% of max for safety)
        self.spot_weight_limit = 840  # 70% of 1200
        self.spot_raw_requests_limit = 4200  # 70% of 6000 per 5min
        self.spot_orders_per_second = 7  # 70% of 10
        self.spot_orders_per_10s = 70  # 70% of 100
        
        # Futures API limits (use 70% of max for safety)
        self.futures_weight_limit = 1680  # 70% of 2400
        self.futures_orders_per_10s = 210  # 70% of 300
        
        # Tracking queues
        self.spot_weight_queue = deque()
        self.futures_weight_queue = deque()
        self.spot_order_queue = deque()
        self.futures_order_queue = deque()
    
    async def wait_for_spot_request(self, weight=1):
        # CRITICAL: Must implement proper weight tracking
        await self._wait_for_weight_limit(self.spot_weight_queue, weight, self.spot_weight_limit, 60)
    
    async def wait_for_futures_request(self, weight=1):
        # CRITICAL: Must implement proper weight tracking
        await self._wait_for_weight_limit(self.futures_weight_queue, weight, self.futures_weight_limit, 60)
    
    async def wait_for_spot_order(self):
        # CRITICAL: Must respect order rate limits
        await self._wait_for_order_limit(self.spot_order_queue, self.spot_orders_per_10s, 10)
    
    async def wait_for_futures_order(self):
        # CRITICAL: Must respect order rate limits
        await self._wait_for_order_limit(self.futures_order_queue, self.futures_orders_per_10s, 10)
```

### 2. Binance API Error Handling (MANDATORY)

**HTTP Error Codes (MUST HANDLE PROPERLY)**:
- **HTTP 418**: IP banned (CRITICAL - must stop all operations)
- **HTTP 429**: Rate limit exceeded (CRITICAL - must implement backoff)
- **HTTP 5XX**: Server errors (must implement retry with exponential backoff)
- **-1003**: WAF (Web Application Firewall) triggered
- **-1021**: Timestamp outside recv window
- **-2010**: Insufficient funds
- **-2011**: Unknown order
- **-1013**: Filter failure (price/quantity precision)

**Error Handling Implementation**:
```python
class BinanceErrorHandler:
    def __init__(self):
        self.ban_status = False
        self.rate_limited_until = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
    
    async def handle_api_error(self, error_code, error_msg):
        if error_code == -1003:  # WAF triggered
            # CRITICAL: Must reduce request frequency
            await self.implement_waf_backoff()
        elif error_code == 418:  # IP banned
            # CRITICAL: Must stop all operations immediately
            self.ban_status = True
            await self.emergency_shutdown()
        elif error_code == 429:  # Rate limit exceeded
            # CRITICAL: Must implement proper backoff
            await self.implement_rate_limit_backoff()
        elif error_code == -1021:  # Timestamp issues
            # CRITICAL: Must sync server time
            await self.sync_server_time()
        elif error_code in [-2010, -2011]:  # Order/balance issues
            # CRITICAL: Must validate before orders
            await self.validate_account_status()
```

### 3. Order Validation and Filters (MANDATORY COMPLIANCE)

**Symbol Filters (MUST IMPLEMENT ALL)**:
```python
class BinanceOrderValidator:
    def __init__(self):
        self.exchange_info = {}  # Must load from /api/v3/exchangeInfo
        self.symbol_filters = {}
    
    def validate_order(self, symbol, side, order_type, quantity, price=None):
        """
        CRITICAL: Must validate ALL filters before placing order
        """
        filters = self.symbol_filters[symbol]
        
        # PRICE_FILTER validation
        if 'PRICE_FILTER' in filters:
            price_filter = filters['PRICE_FILTER']
            min_price = float(price_filter['minPrice'])
            max_price = float(price_filter['maxPrice'])
            tick_size = float(price_filter['tickSize'])
            
            if price and (price < min_price or price > max_price):
                raise ValueError(f"Price {price} outside range [{min_price}, {max_price}]")
            
            # Price must be multiple of tick_size
            if price and (price % tick_size) != 0:
                price = self.round_to_tick_size(price, tick_size)
        
        # LOT_SIZE validation
        if 'LOT_SIZE' in filters:
            lot_filter = filters['LOT_SIZE']
            min_qty = float(lot_filter['minQty'])
            max_qty = float(lot_filter['maxQty'])
            step_size = float(lot_filter['stepSize'])
            
            if quantity < min_qty or quantity > max_qty:
                raise ValueError(f"Quantity {quantity} outside range [{min_qty}, {max_qty}]")
            
            # Quantity must be multiple of step_size
            if (quantity % step_size) != 0:
                quantity = self.round_to_step_size(quantity, step_size)
        
        # MIN_NOTIONAL validation
        if 'MIN_NOTIONAL' in filters:
            min_notional = float(filters['MIN_NOTIONAL']['minNotional'])
            notional_value = quantity * (price or self.get_current_price(symbol))
            
            if notional_value < min_notional:
                raise ValueError(f"Notional value {notional_value} below minimum {min_notional}")
        
        # MARKET_LOT_SIZE validation for market orders
        if order_type == 'MARKET' and 'MARKET_LOT_SIZE' in filters:
            market_filter = filters['MARKET_LOT_SIZE']
            market_min_qty = float(market_filter['minQty'])
            market_max_qty = float(market_filter['maxQty'])
            
            if quantity < market_min_qty or quantity > market_max_qty:
                raise ValueError(f"Market order quantity {quantity} outside range")
        
        return quantity, price
```

### 4. WebSocket Connection Management (MANDATORY)

**WebSocket Rules (MUST COMPLY)**:
- Maximum 5 streams per connection
- Maximum 1024 connections per IP
- Proper connection lifecycle management
- Heartbeat/ping-pong handling
- Reconnection with exponential backoff

```python
class BinanceWebSocketManager:
    def __init__(self):
        self.max_streams_per_connection = 5
        self.connections = []
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
    
    async def create_connection(self, streams):
        """
        CRITICAL: Must respect stream limits per connection
        """
        if len(streams) > self.max_streams_per_connection:
            # Split into multiple connections
            return await self.create_multiple_connections(streams)
        
        # Create single connection with proper error handling
        return await self.create_single_connection(streams)
    
    async def handle_websocket_error(self, error):
        """
        CRITICAL: Must implement proper reconnection logic
        """
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            raise Exception("Max reconnection attempts exceeded")
        
        # Exponential backoff: 2^attempt seconds
        backoff_time = min(2 ** self.reconnect_attempts, 60)
        await asyncio.sleep(backoff_time)
        
        return await self.reconnect()
```

### 5. Account and Position Management (MANDATORY)

**Account Requirements**:
- Must validate account status before trading
- Must check trading permissions
- Must respect account-specific limits
- Must handle margin requirements properly

```python
class BinanceAccountManager:
    def __init__(self):
        self.account_info = {}
        self.trading_status = {}
        self.position_limits = {}
    
    async def validate_account_status(self):
        """
        CRITICAL: Must check account status before trading
        """
        # Check if account can trade
        account_info = await self.get_account_info()
        
        if not account_info['canTrade']:
            raise Exception("Account trading disabled")
        
        if not account_info['canWithdraw']:
            # Log warning but continue (affects only withdrawals)
            logger.warning("Account withdrawal disabled")
        
        # Check for any restrictions
        if account_info.get('accountType') == 'SPOT':
            # Spot-only account restrictions
            pass
    
    async def check_balance_requirements(self, symbol, quantity, price):
        """
        CRITICAL: Must validate sufficient balance before orders
        """
        base_asset = self.get_base_asset(symbol)
        quote_asset = self.get_quote_asset(symbol)
        
        balances = await self.get_account_balances()
        
        # Check balance requirements
        required_balance = quantity * price  # For buy orders
        available_balance = float(balances[quote_asset]['free'])
        
        if required_balance > available_balance:
            raise ValueError(f"Insufficient {quote_asset} balance")
        
        return True
```

### 6. Futures-Specific Compliance (MANDATORY)

**Futures API Rules**:
- Must handle margin requirements
- Must respect leverage limits
- Must implement proper risk management
- Must handle position mode (hedge/one-way)

```python
class BinanceFuturesCompliance:
    def __init__(self):
        self.position_mode = 'HEDGE'  # or 'ONE_WAY'
        self.max_leverage = {}  # Per symbol
        self.margin_requirements = {}
    
    async def validate_futures_order(self, symbol, side, quantity, leverage=1):
        """
        CRITICAL: Must validate futures-specific requirements
        """
        # Check leverage limits
        max_leverage = self.max_leverage.get(symbol, 1)
        if leverage > max_leverage:
            raise ValueError(f"Leverage {leverage} exceeds maximum {max_leverage}")
        
        # Check margin requirements
        required_margin = self.calculate_required_margin(symbol, quantity, leverage)
        available_margin = await self.get_available_margin()
        
        if required_margin > available_margin:
            raise ValueError("Insufficient margin for futures position")
        
        # Check position limits
        current_position = await self.get_position_info(symbol)
        max_position = self.get_max_position_limit(symbol)
        
        if abs(current_position + quantity) > max_position:
            raise ValueError("Position would exceed maximum limit")
        
        return True
    
    async def manage_margin_requirements(self):
        """
        CRITICAL: Must maintain adequate margin at all times
        """
        margin_ratio = await self.get_margin_ratio()
        
        if margin_ratio < 0.1:  # 10% margin ratio threshold
            # CRITICAL: Must reduce positions or add margin
            await self.emergency_margin_management()
```

## Core Strategy Implementation

### 1. Delta-Neutral Position Management (Compliant)
```python
class CompliantDeltaNeutralManager:
    def __init__(self):
        self.rate_limiter = BinanceAPIRateLimiter()
        self.order_validator = BinanceOrderValidator()
        self.account_manager = BinanceAccountManager()
        self.error_handler = BinanceErrorHandler()
    
    async def execute_delta_neutral_trade(self, spot_symbol, futures_symbol, quantity):
        """
        CRITICAL: Must comply with all API rules for both spot and futures
        """
        try:
            # Pre-trade validation
            await self.account_manager.validate_account_status()
            await self.rate_limiter.wait_for_spot_order()
            
            # Validate spot order
            spot_quantity, spot_price = self.order_validator.validate_order(
                spot_symbol, 'BUY', 'LIMIT', quantity, self.get_current_price(spot_symbol)
            )
            
            # Execute spot order
            spot_order = await self.execute_spot_order(spot_symbol, 'BUY', spot_quantity, spot_price)
            
            if spot_order['status'] == 'FILLED':
                # Execute corresponding futures hedge
                await self.rate_limiter.wait_for_futures_order()
                futures_order = await self.execute_futures_hedge(futures_symbol, spot_quantity)
                
                return {
                    'spot_order': spot_order,
                    'futures_order': futures_order,
                    'delta_neutral': True
                }
        
        except Exception as e:
            await self.error_handler.handle_api_error(e.code if hasattr(e, 'code') else -1, str(e))
            raise
```

### 2. Grid Trading with Full Compliance
```python
class CompliantGridTrader:
    def __init__(self, symbol):
        self.symbol = symbol
        self.rate_limiter = BinanceAPIRateLimiter()
        self.order_validator = BinanceOrderValidator()
        self.active_orders = {}
        self.grid_levels = []
    
    async def create_compliant_grid(self, center_price, grid_spacing, grid_levels):
        """
        CRITICAL: Must validate all grid orders before placement
        """
        validated_orders = []
        
        for i in range(-grid_levels//2, grid_levels//2 + 1):
            if i == 0:
                continue  # Skip center price
            
            price = center_price * (1 + (i * grid_spacing))
            side = 'BUY' if i < 0 else 'SELL'
            quantity = self.calculate_grid_quantity(abs(i))
            
            try:
                # Validate each order
                validated_qty, validated_price = self.order_validator.validate_order(
                    self.symbol, side, 'LIMIT', quantity, price
                )
                
                validated_orders.append({
                    'side': side,
                    'quantity': validated_qty,
                    'price': validated_price,
                    'level': i
                })
                
            except ValueError as e:
                logger.warning(f"Grid level {i} validation failed: {e}")
                continue
        
        return await self.place_grid_orders(validated_orders)
    
    async def place_grid_orders(self, orders):
        """
        CRITICAL: Must respect rate limits when placing multiple orders
        """
        placed_orders = []
        
        for order in orders:
            try:
                # Wait for rate limit
                await self.rate_limiter.wait_for_spot_order()
                
                # Place order
                result = await self.place_single_order(
                    self.symbol,
                    order['side'],
                    'LIMIT',
                    order['quantity'],
                    order['price']
                )
                
                placed_orders.append(result)
                self.active_orders[result['orderId']] = order
                
                # Small delay between orders to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to place grid order: {e}")
                await self.error_handler.handle_api_error(
                    e.code if hasattr(e, 'code') else -1, str(e)
                )
        
        return placed_orders
```

### 3. Funding Fee Collection (Compliant)
```python
class CompliantFundingFeeCollector:
    def __init__(self):
        self.rate_limiter = BinanceAPIRateLimiter()
        self.futures_compliance = BinanceFuturesCompliance()
    
    async def collect_funding_fees(self):
        """
        CRITICAL: Must comply with futures API rules for funding fee collection
        """
        try:
            await self.rate_limiter.wait_for_futures_request(weight=20)  # Income endpoint weight
            
            # Get funding fee history
            funding_history = await self.get_futures_income_history(
                incomeType='FUNDING_FEE',
                limit=100
            )
            
            total_funding_fees = sum(float(record['income']) for record in funding_history)
            
            return {
                'total_funding_fees': total_funding_fees,
                'fee_records': funding_history
            }
            
        except Exception as e:
            await self.error_handler.handle_api_error(
                e.code if hasattr(e, 'code') else -1, str(e)
            )
            raise
```

### 4. Position Compounding (Compliant)
```python
class CompliantPositionCompounder:
    def __init__(self):
        self.rate_limiter = BinanceAPIRateLimiter()
        self.account_manager = BinanceAccountManager()
        self.max_position_increase = 0.1  # 10% max increase per compounding
    
    async def compound_positions(self, profit_amount):
        """
        CRITICAL: Must validate account status and limits before compounding
        """
        try:
            # Validate account status
            await self.account_manager.validate_account_status()
            
            # Calculate safe position increase
            current_portfolio_value = await self.get_portfolio_value()
            max_increase = current_portfolio_value * self.max_position_increase
            position_increase = min(profit_amount, max_increase)
            
            # Validate against account limits
            if position_increase < 10:  # Minimum viable increase
                logger.info("Profit too small for compounding")
                return
            
            # Execute position increase with proper validation
            return await self.increase_positions(position_increase)
            
        except Exception as e:
            logger.error(f"Compounding failed: {e}")
            raise
```

## CRITICAL Implementation Checklist

### Pre-Deployment Validation (MANDATORY):
- [ ] Rate limiting implemented for all API endpoints
- [ ] All order filters properly validated
- [ ] Error handling for all Binance error codes
- [ ] WebSocket connection management compliant
- [ ] Account status validation implemented
- [ ] Margin requirements properly handled
- [ ] Position limits respected
- [ ] Audit trail and logging complete
- [ ] Emergency shutdown procedures ready
- [ ] Compliance monitoring active

### Runtime Monitoring (MANDATORY):
- [ ] Real-time rate limit usage tracking
- [ ] API error monitoring and alerting
- [ ] Order validation success rate
- [ ] Position limit compliance
- [ ] Margin ratio monitoring
- [ ] WebSocket connection health
- [ ] Account status changes
- [ ] Regulatory compliance metrics

## Risk Management and Disclaimers

**CRITICAL WARNINGS**:
- Any violation of Binance API rules can result in account suspension or permanent ban
- Rate limit violations can trigger IP bans affecting all operations
- Improper order validation can result in failed trades and losses
- Margin requirements must be continuously monitored to avoid liquidation
- This system requires constant monitoring and maintenance

**Legal Compliance**:
- Must comply with local financial regulations
- Tax implications must be properly managed
- Professional advice strongly recommended
- Users assume full responsibility for compliance violations

---

**IMPLEMENTATION PRIORITY**:
1. **CRITICAL**: Full Binance API compliance framework
2. **HIGH**: Core trading logic with validation
3. **MEDIUM**: Advanced features and optimization
4. **LOW**: Performance tuning and monitoring enhancements

This implementation framework ensures full compliance with Binance API rules and regulations while implementing a sophisticated delta-neutral trading strategy.