import asyncio
import logging
from datetime import datetime
from src.delta_neutral_grid_bot.binance_client import BinanceClient
from src.delta_neutral_grid_bot.config import SYMBOL, GRID_SIZE, GRID_INTERVAL, ORDER_SIZE, FUTURES_SYMBOL
from src.delta_neutral_grid_bot.state_manager import StateManager
from src.compliance.binance_api_rate_limiter import BinanceAPIRateLimiter
from src.compliance.binance_error_handler import BinanceErrorHandler
from src.compliance.binance_order_validator import BinanceOrderValidator
from src.compliance.binance_account_manager import BinanceAccountManager
from src.compliance.binance_futures_compliance import BinanceFuturesCompliance
from src.compliance.compliant_grid_trader import CompliantGridTrader
from src.compliance.compliant_delta_neutral_manager import CompliantDeltaNeutralManager
from src.compliance.compliant_funding_fee_collector import CompliantFundingFeeCollector
from src.volume_tracker import VIPVolumeTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeltaNeutralGridBot:
    def __init__(self):
        self.client = BinanceClient()
        self.state_manager = StateManager()
        self.rate_limiter = BinanceAPIRateLimiter()
        self.error_handler = BinanceErrorHandler()
        self.order_validator = None
        self.account_manager = None
        self.futures_compliance = None
        self.grid_trader = None
        self.delta_neutral_manager = None
        self.funding_fee_collector = None
        self.volume_tracker = VIPVolumeTracker()
        self.shutdown_event = asyncio.Event()
        self.grid_orders = self.state_manager.get('grid_orders', [])
        self.total_funding_fees = 0.0
        self.total_profits = 0.0

    async def run(self):
        logging.info("Starting delta-neutral grid bot...")
        async with self.client as client:
            exchange_info = await client.get_exchange_info()
            if not exchange_info:
                logging.critical("Could not get exchange info. Shutting down.")
                return

            self.order_validator = BinanceOrderValidator(exchange_info)
            self.account_manager = BinanceAccountManager(client)
            self.futures_compliance = BinanceFuturesCompliance(client)
            self.grid_trader = CompliantGridTrader(client, self.rate_limiter, self.order_validator)
            self.delta_neutral_manager = CompliantDeltaNeutralManager(client, self.rate_limiter, self.order_validator)
            self.funding_fee_collector = CompliantFundingFeeCollector(client, self.rate_limiter)

            if not await self.account_manager.initialize():
                logging.critical("Account initialization failed. Shutting down.")
                return

            if not self.grid_orders:
                await self.setup_grid(client)

            while not self.shutdown_event.is_set():
                try:
                    await self.check_grid(client)
                    await self.hedge_position(client)
                    await self.collect_funding_fees(client)
                    self.state_manager.set('grid_orders', self.grid_orders)
                    
                    # Log VIP status every 10 minutes
                    if hasattr(self, '_last_vip_log'):
                        if (datetime.now() - self._last_vip_log).seconds >= 600:
                            self.volume_tracker.log_vip_status()
                            self._last_vip_log = datetime.now()
                    else:
                        self._last_vip_log = datetime.now()
                        
                    await asyncio.sleep(60)  # Check every minute
                except KeyboardInterrupt:
                    logging.info("Stopping bot...")
                    await self.cleanup(client)
                    break
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    await self.error_handler.handle_error(e)

    async def setup_grid(self, client):
        logging.info("Setting up grid...")
        try:
            ticker = await client.get_symbol_ticker(symbol=SYMBOL)
            if not ticker:
                logging.error("Failed to get ticker price for grid setup")
                return

            current_price = float(ticker['price'])
            buy_prices, sell_prices = self.calculate_grid_lines(current_price)
        except Exception as e:
            logging.error(f"Error setting up grid: {e}")
            await self.error_handler.handle_error(e)
            return

        # Create BUY orders below current price
        for price in buy_prices:
            try:
                buy_order = await self.grid_trader.place_limit_order(SYMBOL, 'BUY', ORDER_SIZE, price)
                if buy_order:
                    self.grid_orders.append(buy_order)
            except Exception as e:
                logging.error(f"Failed to place BUY order at {price}: {e}")
                await self.error_handler.handle_error(e)
        
        # Create SELL orders above current price  
        for price in sell_prices:
            try:
                sell_order = await self.grid_trader.place_limit_order(SYMBOL, 'SELL', ORDER_SIZE, price)
                if sell_order:
                    self.grid_orders.append(sell_order)
            except Exception as e:
                logging.error(f"Failed to place SELL order at {price}: {e}")
                await self.error_handler.handle_error(e)
        
        self.state_manager.set('grid_orders', self.grid_orders)

    def calculate_grid_lines(self, current_price):
        buy_prices = []
        sell_prices = []
        for i in range(GRID_SIZE // 2):
            buy_prices.append(current_price - (i + 1) * GRID_INTERVAL)
            sell_prices.append(current_price + (i + 1) * GRID_INTERVAL)
        return sorted(buy_prices, reverse=True), sorted(sell_prices)

    async def check_grid(self, client):
        logging.info("Checking grid...")
        for order in self.grid_orders[:]: # Iterate over a copy
            try:
                order_status = await client.get_order(symbol=SYMBOL, orderId=order['orderId'])
                if order_status['status'] == 'FILLED':
                    logging.info(f"Order {order['orderId']} filled!")
                    self.grid_orders.remove(order)
                    
                    # Replace the filled order with a new one on the opposite side
                    if order_status['side'] == 'BUY':
                        # BUY order filled, place SELL order above
                        new_price = float(order_status['price']) + GRID_INTERVAL
                        new_order = await self.grid_trader.place_limit_order(SYMBOL, 'SELL', ORDER_SIZE, new_price)
                        logging.info(f"BUY filled at {order_status['price']}, placing SELL at {new_price}")
                    else: # SELL order filled
                        # SELL order filled, place BUY order below  
                        new_price = float(order_status['price']) - GRID_INTERVAL
                        new_order = await self.grid_trader.place_limit_order(SYMBOL, 'BUY', ORDER_SIZE, new_price)
                        logging.info(f"SELL filled at {order_status['price']}, placing BUY at {new_price}")
                        
                    if new_order:
                        self.grid_orders.append(new_order)
                        
                        # Track volume for VIP status
                        trade_volume = float(order_status['price']) * float(order_status['executedQty'])
                        self.volume_tracker.track_trade_volume(trade_volume, SYMBOL)

            except Exception as e:
                logging.error(f"Error checking order {order.get('orderId', 'unknown')}: {e}")
                await self.error_handler.handle_error(e)

    async def hedge_position(self, client):
        logging.info("Hedging position...")
        try:
            spot_balance = await client.get_asset_balance(asset=SYMBOL.replace("USDT", ""))
            if not spot_balance:
                logging.warning("Could not get spot balance for hedging")
                return
            
            spot_position = float(spot_balance['free'])
            
            futures_position_risk = await client.get_position_risk(symbol=FUTURES_SYMBOL)
            if not futures_position_risk:
                logging.warning("Could not get futures position for hedging")
                return
            
            futures_position = float(futures_position_risk[0]['positionAmt'])
            
            # CORRECT calculation for delta-neutral: spot MINUS futures
            delta = spot_position - futures_position

            if abs(delta) > ORDER_SIZE: # Only hedge if the delta is significant
                if delta > 0: # Too much spot exposure, need SHORT futures
                    logging.info(f"Delta {delta} > 0, placing SHORT futures order for {abs(delta)}")
                    await self.place_futures_sell_order(client, FUTURES_SYMBOL, abs(delta))
                else: # Too much short exposure, need LONG futures  
                    logging.info(f"Delta {delta} < 0, placing LONG futures order for {abs(delta)}")
                    await self.place_futures_buy_order(client, FUTURES_SYMBOL, abs(delta))

        except Exception as e:
            logging.error(f"Error in hedge_position: {e}")
            await self.error_handler.handle_error(e)

    async def place_futures_buy_order(self, client, symbol, quantity):
        """Place a LONG futures order to reduce short exposure"""
        try:
            if not await self.rate_limiter.wait_for_order():
                logging.error("Rate limit exceeded, cannot place futures buy order")
                return None
            
            order = await client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            logging.info(f"Placed futures BUY order: {order}")
            return order
        except Exception as e:
            logging.error(f"Failed to place futures BUY order: {e}")
            await self.error_handler.handle_error(e)
            return None

    async def place_futures_sell_order(self, client, symbol, quantity):
        """Place a SHORT futures order to reduce long exposure"""
        try:
            if not await self.rate_limiter.wait_for_order():
                logging.error("Rate limit exceeded, cannot place futures sell order")
                return None
            
            order = await client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            logging.info(f"Placed futures SELL order: {order}")
            return order
        except Exception as e:
            logging.error(f"Failed to place futures SELL order: {e}")
            await self.error_handler.handle_error(e)
            return None

    async def collect_funding_fees(self, client):
        """Actively monitor and collect funding fees"""
        try:
            if not self.funding_fee_collector:
                return
                
            funding_income = await self.funding_fee_collector.get_funding_fee_income()
            if funding_income and funding_income > 0:
                logging.info(f"Collected funding fees: {funding_income} USDT")
                self.total_funding_fees += funding_income
                self.total_profits += funding_income
                
                # Compound funding fees into larger positions
                await self.compound_profits(funding_income)
                
        except Exception as e:
            logging.error(f"Error collecting funding fees: {e}")
            await self.error_handler.handle_error(e)

    async def compound_profits(self, profit_amount: float):
        """Compound profits by increasing position sizes"""
        try:
            if profit_amount < 50:  # Minimum threshold
                return
            
            # Increase grid order sizes by profit percentage (max 10% increase)
            size_increase = min(profit_amount / 1000, 0.1)
            global ORDER_SIZE
            ORDER_SIZE *= (1 + size_increase)
            
            logging.info(f"Compounded {profit_amount} USDT profit, new order size: {ORDER_SIZE:.6f}")
            
            # Update state
            self.state_manager.set('order_size', ORDER_SIZE)
            self.state_manager.set('total_profits', self.total_profits)
            
        except Exception as e:
            logging.error(f"Error compounding profits: {e}")
            await self.error_handler.handle_error(e)

    async def cleanup(self, client):
        logging.info("Cleaning up open orders...")
        try:
            open_orders = await client.get_open_orders(symbol=SYMBOL)
            for order in open_orders:
                try:
                    if await self.rate_limiter.wait_for_order():
                        await client.cancel_order(symbol=SYMBOL, orderId=order['orderId'])
                        logging.info(f"Cancelled order {order['orderId']}")
                except Exception as e:
                    logging.error(f"Error cancelling order {order.get('orderId', 'unknown')}: {e}")
                    await self.error_handler.handle_error(e)
        except Exception as e:
            logging.error(f"Error getting open orders for cleanup: {e}")
            await self.error_handler.handle_error(e)
        
        try:
            await self.rate_limiter.shutdown()
        except Exception as e:
            logging.error(f"Error shutting down rate limiter: {e}")

if __name__ == "__main__":
    bot = DeltaNeutralGridBot()
    asyncio.run(bot.run())