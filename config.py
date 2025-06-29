import os

# Binance API Configuration - NEVER hardcode these in production
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

if not API_KEY or not API_SECRET:
    raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET environment variables must be set")

# Trading Configuration
TRADING_TRIANGLES = [
    ["USDT", "BTC", "ETH"],  # USDT -> BTC -> ETH -> USDT
    ["USDT", "BTC", "BNB"],  # USDT -> BTC -> BNB -> USDT  
    ["USDT", "ETH", "BNB"],  # USDT -> ETH -> BNB -> USDT
    ["BTC", "ETH", "BNB"],   # BTC -> ETH -> BNB -> BTC
]

# Risk Management
TRADE_AMOUNT_USDT = 50.0  # Amount to trade per opportunity (in USDT equivalent)
MIN_PROFIT_THRESHOLD = 0.0015  # Minimum 0.15% profit after fees to execute trade
MAX_TRADE_AMOUNT_PERCENTAGE = 0.05  # Maximum 5% of available balance per trade
MAX_DAILY_LOSS_USDT = 100.0  # Maximum daily loss limit
SLIPPAGE_TOLERANCE = 0.001  # 0.1% slippage tolerance

# Trading Fees (Binance standard rates)
TRADING_FEE = 0.001  # 0.1% trading fee per trade

# Rate Limiting (Conservative settings to avoid API bans)
MAX_REQUESTS_PER_MINUTE = 1000  # Binance allows 1200, we use 1000 for safety
MAX_ORDERS_PER_10_SECONDS = 40   # Binance allows 50, we use 40 for safety
MAX_ORDERS_PER_MINUTE = 80       # Binance allows 100, we use 80 for safety

# WebSocket Configuration
WEBSOCKET_RECONNECT_DELAY = 5  # Seconds to wait before reconnecting
MAX_RECONNECT_ATTEMPTS = 10

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "arbitrage_bot.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Order Configuration
ORDER_TIMEOUT_SECONDS = 3  # Time to wait for order fills
PRICE_PRECISION = 8  # Decimal places for price calculations
QUANTITY_PRECISION = 8  # Decimal places for quantity calculations

# Performance Tuning
OPPORTUNITY_SCAN_INTERVAL = 0.1  # Seconds between opportunity scans
MIN_LIQUIDITY_THRESHOLD = 1000  # Minimum USD value in order book