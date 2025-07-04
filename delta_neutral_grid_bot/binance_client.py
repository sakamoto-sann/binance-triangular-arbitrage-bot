from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException
from src.delta_neutral_grid_bot.config import API_KEY, API_SECRET
from src.delta_neutral_grid_bot.logger import get_logger

logger = get_logger(__name__)

class BinanceClient:
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = await AsyncClient.create(API_KEY, API_SECRET)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close_connection()

    async def get_account(self):
        try:
            account_info = await self.client.get_account()
            return account_info
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def create_order(self, symbol, side, type, quantity, price=None, timeInForce='GTC'):
        try:
            if price:
                order = await self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=type,
                    timeInForce=timeInForce,
                    quantity=quantity,
                    price=price
                )
            else:
                order = await self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=type,
                    quantity=quantity
                )
            logger.info(f"Placed order: {order}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def get_open_orders(self, symbol):
        try:
            open_orders = await self.client.get_open_orders(symbol=symbol)
            return open_orders
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def cancel_order(self, symbol, orderId):
        try:
            result = await self.client.cancel_order(symbol=symbol, orderId=orderId)
            logger.info(f"Canceled order: {result}")
            return result
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error canceling order: {e}")
            return None

    async def get_symbol_ticker(self, symbol):
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return ticker
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting symbol ticker: {e}")
            return None

    async def get_exchange_info(self):
        try:
            info = await self.client.get_exchange_info()
            return info
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting exchange info: {e}")
            return None
