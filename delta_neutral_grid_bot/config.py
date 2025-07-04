import os

# Binance API credentials
API_KEY = os.environ.get("BINANCE_API_KEY", "YOUR_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_API_SECRET")

# Trading parameters
SYMBOL = "BTCUSDT"
FUTURES_SYMBOL = "BTCUSDT"  # Binance USDT-M Perpetual Contract
GRID_SIZE = 10
GRID_INTERVAL = 100  # In USDT
ORDER_SIZE = 0.001  # In BTC

# Strategy parameters
HEDGE_RATIO = 1.0
MIN_PROFIT_PERCENT = 0.1

# Other settings
LOG_LEVEL = "INFO"
LOG_FILE = "delta_neutral_bot.log"
