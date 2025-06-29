#!/usr/bin/env python3
"""
Triangular Arbitrage Bot for Binance
Real-time WebSocket-based arbitrage detection and execution
"""

import asyncio
import json
import logging
import logging.handlers
import time
from collections import defaultdict, deque
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
import websockets
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException
from . import config
from .telegram_notifier import TelegramNotifier
from .__version__ import __version__, print_version

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            config.LOG_FILE, 
            maxBytes=config.LOG_MAX_BYTES, 
            backupCount=config.LOG_BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Advanced rate limiter for Binance API compliance"""
    
    def __init__(self):
        self.request_timestamps = deque()
        self.order_timestamps_10s = deque()
        self.order_timestamps_1m = deque()
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, is_order: bool = False):
        """Wait if necessary to comply with rate limits"""
        async with self.lock:
            current_time = time.time()
            
            # Clean old timestamps
            self._clean_old_timestamps(current_time)
            
            # Check request rate limit (1000 per minute)
            if len(self.request_timestamps) >= config.MAX_REQUESTS_PER_MINUTE:
                sleep_time = 60 - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    self._clean_old_timestamps(time.time())
            
            # Check order rate limits if this is an order
            if is_order:
                if len(self.order_timestamps_10s) >= config.MAX_ORDERS_PER_10_SECONDS:
                    sleep_time = 10 - (current_time - self.order_timestamps_10s[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                
                if len(self.order_timestamps_1m) >= config.MAX_ORDERS_PER_MINUTE:
                    sleep_time = 60 - (current_time - self.order_timestamps_1m[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Record this request/order
            current_time = time.time()
            self.request_timestamps.append(current_time)
            if is_order:
                self.order_timestamps_10s.append(current_time)
                self.order_timestamps_1m.append(current_time)
    
    def _clean_old_timestamps(self, current_time: float):
        """Remove timestamps older than their respective windows"""
        # Clean 1-minute request window
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Clean 10-second order window
        while self.order_timestamps_10s and current_time - self.order_timestamps_10s[0] > 10:
            self.order_timestamps_10s.popleft()
        
        # Clean 1-minute order window
        while self.order_timestamps_1m and current_time - self.order_timestamps_1m[0] > 60:
            self.order_timestamps_1m.popleft()

class DataStore:
    """Manages real-time price data and exchange information"""
    
    def __init__(self):
        self.prices: Dict[str, Dict] = {}
        self.symbol_info: Dict[str, Dict] = {}
        self.exchange_info_loaded = False
        self.price_update_callbacks = []
    
    async def load_exchange_info(self, client: AsyncClient):
        """Load and cache exchange information"""
        try:
            exchange_info = await client.get_exchange_info()
            for symbol_data in exchange_info['symbols']:
                if symbol_data['status'] == 'TRADING':
                    symbol = symbol_data['symbol']
                    self.symbol_info[symbol] = {
                        'base': symbol_data['baseAsset'],
                        'quote': symbol_data['quoteAsset'],
                        'filters': {f['filterType']: f for f in symbol_data['filters']}
                    }
            self.exchange_info_loaded = True
            logger.info(f"Loaded {len(self.symbol_info)} trading symbols")
        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")
            raise
    
    def get_symbol_name(self, base: str, quote: str) -> Optional[str]:
        """Get symbol name from base and quote assets"""
        symbol = f"{base}{quote}"
        return symbol if symbol in self.symbol_info else None
    
    def get_price_data(self, symbol: str) -> Optional[Dict]:
        """Get current price data for symbol"""
        return self.prices.get(symbol)
    
    def update_price(self, symbol: str, bid: float, ask: float):
        """Update price data and notify callbacks"""
        self.prices[symbol] = {
            'bid': bid,
            'ask': ask,
            'timestamp': time.time()
        }
        
        # Notify callbacks
        for callback in self.price_update_callbacks:
            try:
                callback(symbol, bid, ask)
            except Exception as e:
                logger.error(f"Error in price update callback: {e}")
    
    def add_price_callback(self, callback):
        """Add callback for price updates"""
        self.price_update_callbacks.append(callback)

class TradeExecutor:
    """Handles triangular arbitrage trade execution"""
    
    def __init__(self, client: AsyncClient, data_store: DataStore, rate_limiter: RateLimiter):
        self.client = client
        self.data_store = data_store
        self.rate_limiter = rate_limiter
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = time.time()
    
    async def execute_triangle(self, triangle: List[str], initial_amount: float) -> Dict:
        """Execute a triangular arbitrage trade"""
        try:
            # Check daily loss limit
            if self._check_daily_loss_limit():
                return {"success": False, "error": "Daily loss limit exceeded"}
            
            # Get symbol names for the triangle
            symbols = self._get_triangle_symbols(triangle)
            if not symbols:
                return {"success": False, "error": "Invalid triangle symbols"}
            
            # Check if we have sufficient balance
            balance_check = await self._check_balance(triangle[0], initial_amount)
            if not balance_check["success"]:
                return balance_check
            
            # Execute the three trades
            trades = []
            current_amount = initial_amount
            current_asset = triangle[0]
            
            for i in range(3):
                symbol = symbols[i]
                target_asset = triangle[(i + 1) % 3]
                
                trade_result = await self._execute_single_trade(
                    symbol, current_asset, target_asset, current_amount
                )
                
                if not trade_result["success"]:
                    # Cancel any pending orders and return
                    await self._cancel_pending_orders(trades)
                    return trade_result
                
                trades.append(trade_result)
                current_amount = trade_result["received_amount"]
                current_asset = target_asset
            
            # Calculate final profit/loss
            final_amount = current_amount
            profit = final_amount - initial_amount
            profit_percentage = (profit / initial_amount) * 100
            
            # Update daily P&L
            self.daily_pnl += profit
            
            logger.info(f"Triangle completed: {triangle[0]} {initial_amount:.6f} -> {final_amount:.6f}, "
                       f"Profit: {profit:.6f} ({profit_percentage:.4f}%)")
            
            return {
                "success": True,
                "initial_amount": initial_amount,
                "final_amount": final_amount,
                "profit": profit,
                "profit_percentage": profit_percentage,
                "trades": trades
            }
            
        except Exception as e:
            logger.error(f"Error executing triangle: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_triangle_symbols(self, triangle: List[str]) -> Optional[List[str]]:
        """Get symbol names for triangle path"""
        symbols = []
        for i in range(3):
            base = triangle[i]
            quote = triangle[(i + 1) % 3]
            symbol = self.data_store.get_symbol_name(base, quote)
            if not symbol:
                # Try reverse
                symbol = self.data_store.get_symbol_name(quote, base)
                if not symbol:
                    return None
            symbols.append(symbol)
        return symbols
    
    async def _check_balance(self, asset: str, required_amount: float) -> Dict:
        """Check if we have sufficient balance for the trade"""
        try:
            await self.rate_limiter.wait_if_needed()
            balance = await self.client.get_asset_balance(asset=asset)
            available = float(balance['free'])
            
            if available < required_amount:
                return {
                    "success": False,
                    "error": f"Insufficient {asset} balance: {available} < {required_amount}"
                }
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return {"success": False, "error": f"Balance check failed: {str(e)}"}
    
    async def _execute_single_trade(self, symbol: str, from_asset: str, to_asset: str, amount: float) -> Dict:
        """Execute a single trade in the triangle"""
        try:
            symbol_info = self.data_store.symbol_info.get(symbol)
            if not symbol_info:
                return {"success": False, "error": f"Symbol info not found for {symbol}"}
            
            # Determine if we're buying or selling
            is_buy = symbol_info['base'] == to_asset
            side = 'BUY' if is_buy else 'SELL'
            
            # Get current price
            price_data = self.data_store.get_price_data(symbol)
            if not price_data:
                return {"success": False, "error": f"No price data for {symbol}"}
            
            # Calculate order parameters
            if is_buy:
                price = price_data['ask']  # We pay the ask price when buying
                quantity = amount / price
            else:
                price = price_data['bid']  # We receive the bid price when selling
                quantity = amount
            
            # Apply symbol filters
            order_params = self._apply_symbol_filters(symbol, quantity, price)
            if not order_params["success"]:
                return order_params
            
            # Place limit order with IOC (Immediate or Cancel)
            await self.rate_limiter.wait_if_needed(is_order=True)
            
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='IOC',
                quantity=order_params["quantity"],
                price=order_params["price"]
            )
            
            # Check if order was filled
            if order['status'] in ['FILLED', 'PARTIALLY_FILLED']:
                filled_qty = float(order['executedQty'])
                if is_buy:
                    received_amount = filled_qty
                else:
                    received_amount = filled_qty * float(order['price'])
                
                # Account for trading fees
                received_amount *= (1 - config.TRADING_FEE)
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": side,
                    "order_id": order['orderId'],
                    "received_amount": received_amount,
                    "order": order
                }
            else:
                return {"success": False, "error": f"Order not filled: {order['status']}"}
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error in trade: {e}")
            return {"success": False, "error": f"API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error in single trade: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_symbol_filters(self, symbol: str, quantity: float, price: float) -> Dict:
        """Apply Binance symbol filters to order parameters"""
        try:
            symbol_info = self.data_store.symbol_info[symbol]
            filters = symbol_info['filters']
            
            # Apply LOT_SIZE filter
            if 'LOT_SIZE' in filters:
                lot_filter = filters['LOT_SIZE']
                step_size = float(lot_filter['stepSize'])
                min_qty = float(lot_filter['minQty'])
                max_qty = float(lot_filter['maxQty'])
                
                # Round quantity to step size
                quantity = float(Decimal(str(quantity)).quantize(
                    Decimal(str(step_size)), rounding=ROUND_DOWN
                ))
                
                if quantity < min_qty or quantity > max_qty:
                    return {"success": False, "error": f"Quantity {quantity} outside LOT_SIZE limits"}
            
            # Apply PRICE_FILTER
            if 'PRICE_FILTER' in filters:
                price_filter = filters['PRICE_FILTER']
                tick_size = float(price_filter['tickSize'])
                min_price = float(price_filter['minPrice'])
                max_price = float(price_filter['maxPrice'])
                
                # Round price to tick size
                price = float(Decimal(str(price)).quantize(
                    Decimal(str(tick_size)), rounding=ROUND_DOWN
                ))
                
                if price < min_price or price > max_price:
                    return {"success": False, "error": f"Price {price} outside PRICE_FILTER limits"}
            
            # Apply MIN_NOTIONAL filter
            if 'MIN_NOTIONAL' in filters:
                min_notional = float(filters['MIN_NOTIONAL']['minNotional'])
                notional = quantity * price
                
                if notional < min_notional:
                    return {"success": False, "error": f"Notional {notional} below minimum {min_notional}"}
            
            return {"success": True, "quantity": quantity, "price": price}
            
        except Exception as e:
            logger.error(f"Error applying symbol filters: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cancel_pending_orders(self, trades: List[Dict]):
        """Cancel any pending orders from failed triangle execution"""
        for trade in trades:
            if trade.get("success") and trade.get("order_id"):
                try:
                    await self.client.cancel_order(
                        symbol=trade["symbol"],
                        orderId=trade["order_id"]
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel order {trade['order_id']}: {e}")
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        current_time = time.time()
        
        # Reset daily P&L if it's a new day
        if current_time - self.daily_pnl_reset_time > 86400:  # 24 hours
            self.daily_pnl = 0.0
            self.daily_pnl_reset_time = current_time
        
        return self.daily_pnl < -config.MAX_DAILY_LOSS_USDT

class ArbitrageBot:
    """Main triangular arbitrage bot"""
    
    def __init__(self):
        self.client = None
        self.data_store = DataStore()
        self.rate_limiter = RateLimiter()
        self.trade_executor = None
        self.websocket = None
        self.running = False
        self.telegram_notifier = TelegramNotifier()
        self.trade_semaphore = asyncio.Semaphore(1)  # Only one trade at a time
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
    
    async def start(self):
        """Start the arbitrage bot"""
        try:
            logger.info(f"Starting Triangular Arbitrage Bot v{__version__}")
            print_version()
            
            # Send startup notification
            await self.telegram_notifier.send_bot_status("starting", f"v{__version__} - Testnet: {config.TESTNET}")
            
            # Initialize Binance client
            self.client = await AsyncClient.create(
                config.API_KEY, 
                config.API_SECRET,
                testnet=config.TESTNET
            )
            logger.info(f"Binance client initialized (testnet: {config.TESTNET})")
            
            # Load exchange information
            await self.data_store.load_exchange_info(self.client)
            
            # Initialize trade executor
            self.trade_executor = TradeExecutor(self.client, self.data_store, self.rate_limiter)
            
            # Add price update callback for opportunity detection
            self.data_store.add_price_callback(self._on_price_update)
            
            # Start WebSocket connection
            await self._start_websocket()
            
            # Send running notification
            await self.telegram_notifier.send_bot_status("running", "All systems operational")
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            await self.telegram_notifier.send_error_alert("Bot Startup", str(e), "critical")
            await self.stop()
            raise
    
    async def _start_websocket(self):
        """Start WebSocket connection for real-time price data"""
        try:
            # Get all symbols we need for our triangles
            symbols_needed = set()
            for triangle in config.TRADING_TRIANGLES:
                for i in range(3):
                    base = triangle[i]
                    quote = triangle[(i + 1) % 3]
                    symbol = self.data_store.get_symbol_name(base, quote)
                    if symbol:
                        symbols_needed.add(symbol.lower())
            
            if not symbols_needed:
                raise ValueError("No valid symbols found for configured triangles")
            
            # Create WebSocket stream URL
            streams = [f"{symbol}@bookTicker" for symbol in symbols_needed]
            if config.TESTNET:
                base_url = "wss://testnet.binance.vision/ws"
            else:
                base_url = "wss://stream.binance.com:9443/ws"
            stream_url = f"{base_url}/{'/'.join(streams)}"
            
            logger.info(f"Connecting to WebSocket with {len(streams)} streams")
            
            self.running = True
            while self.running:
                try:
                    async with websockets.connect(stream_url) as websocket:
                        self.websocket = websocket
                        self.reconnect_count = 0  # Reset on successful connection
                        logger.info("WebSocket connected")
                        await self.telegram_notifier.send_bot_status("connected", "WebSocket stream active")
                        
                        async for message in websocket:
                            if not self.running:
                                break
                            
                            try:
                                data = json.loads(message)
                                
                                # Handle single stream data
                                if 'stream' in data:
                                    stream_data = data['data']
                                    symbol = stream_data['s']
                                    bid = float(stream_data['b'])
                                    ask = float(stream_data['a'])
                                    self.data_store.update_price(symbol, bid, ask)
                                
                                # Handle direct stream data
                                elif 's' in data:
                                    symbol = data['s']
                                    bid = float(data['b'])
                                    ask = float(data['a'])
                                    self.data_store.update_price(symbol, bid, ask)
                                    
                            except Exception as e:
                                logger.error(f"Error processing WebSocket message: {e}")
                
                except websockets.exceptions.ConnectionClosed:
                    if self.running:
                        self.reconnect_count += 1
                        if self.reconnect_count > self.max_reconnect_attempts:
                            logger.error("Max reconnection attempts exceeded, stopping bot")
                            await self.telegram_notifier.send_error_alert("WebSocket", "Max reconnections exceeded", "high")
                            self.running = False
                            break
                        
                        # Exponential backoff for reconnection
                        delay = min(config.WEBSOCKET_RECONNECT_DELAY * (2 ** (self.reconnect_count - 1)), 60)
                        logger.warning(f"WebSocket connection closed, reconnecting in {delay}s (attempt {self.reconnect_count})")
                        await self.telegram_notifier.send_bot_status("reconnecting", f"Attempt {self.reconnect_count}/{self.max_reconnect_attempts}")
                        await asyncio.sleep(delay)
                except Exception as e:
                    if self.running:
                        self.reconnect_count += 1
                        if self.reconnect_count > self.max_reconnect_attempts:
                            logger.error("Max reconnection attempts exceeded due to errors, stopping bot")
                            await self.telegram_notifier.send_error_alert("WebSocket", "Max reconnections exceeded", "high")
                            self.running = False
                            break
                        
                        # Exponential backoff for errors
                        delay = min(config.WEBSOCKET_RECONNECT_DELAY * (2 ** (self.reconnect_count - 1)), 60)
                        logger.error(f"WebSocket error: {e}, reconnecting in {delay}s (attempt {self.reconnect_count})")
                        await self.telegram_notifier.send_error_alert("WebSocket", str(e), "medium")
                        await asyncio.sleep(delay)
                        
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            self.running = False
            raise
    
    def _on_price_update(self, symbol: str, bid: float, ask: float):
        """Handle price updates and check for arbitrage opportunities"""
        try:
            # Check all configured triangles for opportunities
            for triangle in config.TRADING_TRIANGLES:
                opportunity = self._calculate_triangle_profit(triangle)
                if opportunity and opportunity["profit_percentage"] > config.MIN_PROFIT_THRESHOLD:
                    logger.info(f"Arbitrage opportunity found: {triangle} - "
                               f"{opportunity['profit_percentage']:.4f}% profit")
                    
                    # Send opportunity notification and execute trade asynchronously
                    asyncio.create_task(self._execute_opportunity(triangle, opportunity))
                    
        except Exception as e:
            logger.error(f"Error in price update handler: {e}")
    
    def _calculate_triangle_profit(self, triangle: List[str]) -> Optional[Dict]:
        """Calculate potential profit for a triangular arbitrage"""
        try:
            symbols = self.trade_executor._get_triangle_symbols(triangle)
            if not symbols:
                return None
            
            # Start with 1 unit of the base currency
            amount = 1.0
            
            for i in range(3):
                symbol = symbols[i]
                price_data = self.data_store.get_price_data(symbol)
                if not price_data:
                    return None
                
                symbol_info = self.data_store.symbol_info.get(symbol)
                if not symbol_info:
                    return None
                
                target_asset = triangle[(i + 1) % 3]
                is_buy = symbol_info['base'] == target_asset
                
                if is_buy:
                    # We're buying, so we pay the ask price
                    amount = (amount / price_data['ask']) * (1 - config.TRADING_FEE)
                else:
                    # We're selling, so we receive the bid price
                    amount = (amount * price_data['bid']) * (1 - config.TRADING_FEE)
            
            # Calculate actual profit correctly
            profit_absolute = amount - 1.0
            profit_percentage = profit_absolute * 100  # Convert to percentage
            
            if profit_absolute > 0:
                return {
                    "triangle": triangle,
                    "symbols": symbols,
                    "profit_percentage": profit_percentage,
                    "profit_absolute": profit_absolute,
                    "final_amount": amount
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating triangle profit: {e}")
            return None
    
    async def _execute_opportunity(self, triangle: List[str], opportunity: Dict):
        """Execute an arbitrage opportunity"""
        # Use semaphore to prevent concurrent trades
        async with self.trade_semaphore:
            try:
                # Send opportunity notification
                await self.telegram_notifier.send_opportunity_alert(triangle, opportunity["profit_percentage"])
                
                # Calculate trade amount based on actual balance
                try:
                    balance = await self.client.get_asset_balance(asset=triangle[0])
                    available_balance = float(balance['free'])
                    
                    # Calculate maximum trade amount based on actual balance
                    max_trade_amount = available_balance * config.MAX_TRADE_AMOUNT_PERCENTAGE
                    trade_amount = min(config.TRADE_AMOUNT_USDT, max_trade_amount)
                    
                    # Ensure minimum viable trade amount
                    if trade_amount < 10.0:  # Minimum 10 USDT equivalent
                        logger.warning(f"Trade amount too small: {trade_amount:.2f}, skipping opportunity")
                        return
                        
                except Exception as e:
                    logger.error(f"Failed to get balance for {triangle[0]}: {e}")
                    # Fallback to configured amount
                    trade_amount = config.TRADE_AMOUNT_USDT
                
                # Execute the triangular arbitrage
                result = await self.trade_executor.execute_triangle(triangle, trade_amount)
                
                # Send execution notification
                await self.telegram_notifier.send_trade_execution(result)
                
                if result["success"]:
                    logger.info(f"Successfully executed arbitrage: {result}")
                else:
                    logger.warning(f"Failed to execute arbitrage: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error executing opportunity: {e}")
                await self.telegram_notifier.send_error_alert("Trade Execution", str(e), "high")
    
    async def stop(self):
        """Stop the arbitrage bot"""
        logger.info("Stopping Triangular Arbitrage Bot...")
        self.running = False
        
        # Send stopping notification
        await self.telegram_notifier.send_bot_status("stopping", "Shutting down all connections")
        
        if self.websocket:
            await self.websocket.close()
        
        if self.client:
            await self.client.close_connection()
        
        # Send stopped notification
        await self.telegram_notifier.send_bot_status("stopped", "Bot shutdown complete")
        logger.info("Bot stopped")

async def main():
    """Main entry point"""
    bot = ArbitrageBot()
    
    try:
        await bot.start()
        
        # Keep the bot running
        while bot.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())