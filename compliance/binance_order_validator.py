
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceOrderValidator:
    def __init__(self, exchange_info):
        self.exchange_info = exchange_info

    def validate_order(self, symbol, quantity, price):
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Could not find exchange info for symbol: {symbol}")
            return False

        if not self._validate_price_filter(symbol_info, price):
            return False
        if not self._validate_lot_size(symbol_info, quantity):
            return False
        if not self._validate_min_notional(symbol_info, quantity, price):
            return False
        
        logging.info(f"Order for {symbol} with quantity {quantity} and price {price} is valid.")
        return True

    def _get_symbol_info(self, symbol):
        for s in self.exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None

    def _get_filter(self, symbol_info, filter_type):
        for f in symbol_info['filters']:
            if f['filterType'] == filter_type:
                return f
        return None

    def _validate_price_filter(self, symbol_info, price):
        price_filter = self._get_filter(symbol_info, 'PRICE_FILTER')
        if not price_filter:
            return True # No filter to validate against

        min_price = float(price_filter['minPrice'])
        max_price = float(price_filter['maxPrice'])
        tick_size = float(price_filter['tickSize'])

        if price < min_price:
            logging.error(f"Price {price} is below the minimum price of {min_price}.")
            return False
        if max_price > 0 and price > max_price:
            logging.error(f"Price {price} is above the maximum price of {max_price}.")
            return False
        if (price - min_price) % tick_size != 0:
            logging.error(f"Price {price} does not meet the tick size of {tick_size}.")
            return False
        return True

    def _validate_lot_size(self, symbol_info, quantity):
        lot_size_filter = self._get_filter(symbol_info, 'LOT_SIZE')
        if not lot_size_filter:
            return True

        min_qty = float(lot_size_filter['minQty'])
        max_qty = float(lot_size_filter['maxQty'])
        step_size = float(lot_size_filter['stepSize'])

        if quantity < min_qty:
            logging.error(f"Quantity {quantity} is below the minimum quantity of {min_qty}.")
            return False
        if quantity > max_qty:
            logging.error(f"Quantity {quantity} is above the maximum quantity of {max_qty}.")
            return False
        if (quantity - min_qty) % step_size != 0:
            logging.error(f"Quantity {quantity} does not meet the step size of {step_size}.")
            return False
        return True

    def _validate_min_notional(self, symbol_info, quantity, price):
        min_notional_filter = self._get_filter(symbol_info, 'MIN_NOTIONAL')
        if not min_notional_filter:
            return True

        min_notional = float(min_notional_filter['minNotional'])
        notional_value = quantity * price

        if notional_value < min_notional:
            logging.error(f"Notional value {notional_value} is below the minimum notional of {min_notional}.")
            return False
        return True

if __name__ == '__main__':
    # Mock exchange info for testing
    mock_exchange_info = {
        'symbols': [
            {
                'symbol': 'BTCUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000.00', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.00001', 'maxQty': '9000.0', 'stepSize': '0.00001'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.0'}
                ]
            }
        ]
    }

    validator = BinanceOrderValidator(mock_exchange_info)

    # Valid order
    validator.validate_order('BTCUSDT', 0.001, 60000)

    # Invalid price
    validator.validate_order('BTCUSDT', 0.001, 0.001)

    # Invalid quantity
    validator.validate_order('BTCUSDT', 0.000001, 60000)

    # Invalid notional
    validator.validate_order('BTCUSDT', 0.0001, 60000)
