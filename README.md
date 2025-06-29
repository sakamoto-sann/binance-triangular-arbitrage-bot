# Triangular Arbitrage Bot v0.3

A production-ready triangular arbitrage bot for Binance with real-time WebSocket data processing and comprehensive risk management.

## Features

- ⚡ **Real-time WebSocket data** - No polling delays
- 🛡️ **LIMIT orders with IOC** - No slippage from MARKET orders  
- 🔄 **Triangular arbitrage** - USDT->BTC->ETH->USDT patterns
- 📊 **Proper asset tracking** - Correct amount calculations through triangle legs
- 🚦 **Rate limiting** - Binance API compliant
- 💰 **Risk management** - Balance checks, profit thresholds, daily loss limits
- 📝 **Comprehensive logging** - Monitor trades and performance
- ⚙️ **Symbol filters** - LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL compliance

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
```

### 3. Configure Trading Parameters
Edit `config.py` to set:
- Trading triangles (default: USDT/BTC/ETH combinations)
- Trade amount per opportunity
- Minimum profit threshold
- Risk management limits

### 4. Run the Bot
```bash
python arbitrage_bot.py
```

## Configuration

### Trading Triangles
```python
TRADING_TRIANGLES = [
    ["USDT", "BTC", "ETH"],  # USDT -> BTC -> ETH -> USDT
    ["USDT", "BTC", "BNB"],  # USDT -> BTC -> BNB -> USDT  
    ["USDT", "ETH", "BNB"],  # USDT -> ETH -> BNB -> USDT
    ["BTC", "ETH", "BNB"],   # BTC -> ETH -> BNB -> BTC
]
```

### Risk Management
- `TRADE_AMOUNT_USDT`: Amount to trade per opportunity
- `MIN_PROFIT_THRESHOLD`: Minimum profit % to execute trade
- `MAX_DAILY_LOSS_USDT`: Daily loss limit
- `SLIPPAGE_TOLERANCE`: Maximum acceptable slippage

### Rate Limiting
Conservative settings to avoid API bans:
- 1000 requests per minute (Binance allows 1200)
- 40 orders per 10 seconds (Binance allows 50)
- 80 orders per minute (Binance allows 100)

## How It Works

1. **WebSocket Connection**: Establishes real-time price feeds for all triangle symbols
2. **Opportunity Detection**: Calculates profit potential for each triangle on price updates
3. **Trade Execution**: If profit > threshold, executes 3-leg triangle with LIMIT IOC orders
4. **Risk Management**: Checks balances, applies symbol filters, monitors daily P&L

## Safety Features

- **LIMIT IOC Orders**: Prevents slippage by canceling unfilled orders immediately
- **Balance Validation**: Ensures sufficient funds before trading
- **Daily Loss Limits**: Stops trading if daily loss exceeds threshold
- **Symbol Filters**: Complies with Binance quantity/price/notional requirements
- **Rate Limiting**: Prevents API bans with conservative request limits
- **Error Recovery**: Cancels pending orders on failed triangle execution

## Monitoring

The bot logs all activity to `arbitrage_bot.log` with rotation:
- Trade executions and results
- Profit/loss calculations  
- API errors and reconnections
- Rate limiting events

## Important Notes

⚠️ **TESTNET FIRST**: Always test on Binance Testnet before live trading
⚠️ **SMALL AMOUNTS**: Start with small trade amounts to verify functionality
⚠️ **MARKET CONDITIONS**: Arbitrage opportunities are rare and fleeting
⚠️ **FEES**: Consider VIP levels for lower trading fees

## Legal Disclaimer

This software is for educational purposes. Trading cryptocurrencies involves risk of loss. Use at your own risk and ensure compliance with local regulations.