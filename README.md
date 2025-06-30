# Professional Crypto Trading System v2.0.0 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Trading: Institutional](https://img.shields.io/badge/trading-institutional-gold.svg)](https://github.com/)
[![Volatility: BitVol](https://img.shields.io/badge/volatility-BitVol%2FLXVX-purple.svg)](https://github.com/)
[![Risk: Professional](https://img.shields.io/badge/risk-professional-green.svg)](https://github.com/)

**The world's first retail crypto trading system with institutional-grade volatility indicators (BitVol, LXVX) and professional risk management.**

## 🏆 Revolutionary v2.0.0 Features

**Institutional-Grade Components:**
- 📊 **BitVol & LXVX Integration** - Professional volatility indicators used by hedge funds
- 🧠 **GARCH(1,1) Volatility Forecasting** - Academic-grade volatility prediction
- ⚡ **8 Advanced Trading Modules** - 7,323+ lines of professional code
- 🎯 **Multi-Timeframe Analysis** - 6 timeframes with sophisticated signal weighting
- 🔄 **Cross-Exchange Arbitrage** - Multi-venue funding rate opportunities
- 🎲 **Kelly Criterion Sizing** - Mathematically optimal position sizing
- 🛡️ **Gamma-Aware Hedging** - Option-like exposure management
- 🚨 **Emergency Risk Protocols** - Multi-level circuit breakers and systemic risk monitoring

## 🚀 Features

- ⚡ **Real-time WebSocket data** - Zero polling delays with live price feeds
- 🛡️ **LIMIT orders with IOC** - No slippage from MARKET orders  
- 🔄 **Triangular arbitrage** - USDT→BTC→ETH→USDT patterns
- 📊 **Proper asset tracking** - Accurate amount calculations through triangle legs
- 🚦 **Rate limiting** - Binance API compliant (1000 req/min, 40 orders/10s)
- 💰 **Risk management** - Balance checks, profit thresholds, daily loss limits
- 📱 **Telegram notifications** - Real-time trade alerts and status updates
- 📝 **Comprehensive logging** - Monitor trades and performance with rotation
- ⚙️ **Symbol filters** - LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL compliance
- 🔐 **Security** - Message sanitization, API key protection
- 🐳 **Docker support** - Easy deployment with containers
- ✅ **Testnet support** - Safe testing environment

## 📋 Requirements

- Python 3.8+
- Binance API keys with spot trading permissions
- Minimum 100 USDT equivalent for meaningful arbitrage
- Optional: Telegram bot for notifications

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git
cd binance-triangular-arbitrage-bot

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Configuration

```bash
# Set environment variables
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
export BINANCE_TESTNET="true"  # Start with testnet!

# Optional: Telegram notifications
export TELEGRAM_ENABLED="true"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_IDS="your_chat_id"
```

### 3. Run the Bot

```bash
# Test run
python -m arbitrage_bot.arbitrage_bot

# Or using entry point
arbitrage-bot

# Production deployment
./deployment/deploy.sh
```

## 📊 Trading Configuration

### Default Trading Triangles
```python
TRADING_TRIANGLES = [
    ["USDT", "BTC", "ETH"],  # USDT → BTC → ETH → USDT
    ["USDT", "BTC", "BNB"],  # USDT → BTC → BNB → USDT  
    ["USDT", "ETH", "BNB"],  # USDT → ETH → BNB → USDT
    ["BTC", "ETH", "BNB"],   # BTC → ETH → BNB → BTC
]
```

### Risk Management Settings
- **Trade Amount**: 50 USDT per opportunity (configurable)
- **Min Profit**: 0.15% after fees (configurable) 
- **Max Daily Loss**: 100 USDT (configurable)
- **Balance Limit**: 5% of available balance per trade

## 🛡️ Safety Features

- **LIMIT IOC Orders**: Prevents slippage by canceling unfilled orders immediately
- **Balance Validation**: Ensures sufficient funds before trading
- **Daily Loss Limits**: Stops trading if daily loss exceeds threshold
- **Symbol Filters**: Complies with Binance quantity/price/notional requirements
- **Rate Limiting**: Prevents API bans with conservative request limits
- **Error Recovery**: Cancels pending orders on failed triangle execution
- **Message Sanitization**: Prevents API key leakage in notifications

## 📱 Telegram Notifications

Get real-time updates on:
- 💰 Trade executions (success/failure with P&L)
- 💡 Arbitrage opportunities detected
- 🚨 Error alerts with severity levels
- 📊 Daily trading summaries
- 🔄 Bot status updates (start/stop/reconnect)

## 📈 How It Works

1. **WebSocket Connection**: Establishes real-time price feeds for all triangle symbols
2. **Opportunity Detection**: Calculates profit potential for each triangle on price updates
3. **Trade Execution**: If profit > threshold, executes 3-leg triangle with LIMIT IOC orders
4. **Risk Management**: Checks balances, applies symbol filters, monitors daily P&L
5. **Notifications**: Sends real-time updates via Telegram

## 🚨 Important Safety Notes

⚠️ **ALWAYS TEST ON TESTNET FIRST** (`BINANCE_TESTNET="true"`)  
⚠️ **START WITH SMALL AMOUNTS** (`TRADE_AMOUNT_USDT = 10.0`)  
⚠️ **MONITOR CONTINUOUSLY** for the first 24 hours  
⚠️ **CONSIDER VIP LEVELS** for lower trading fees  
⚠️ **UNDERSTAND THE RISKS** - Arbitrage opportunities are rare and fleeting  

## 🏗️ Project Structure

```
├── src/arbitrage_bot/          # Main source code
│   ├── arbitrage_bot.py        # Core bot logic
│   ├── telegram_notifier.py    # Telegram integration
│   ├── config.py               # Configuration management
│   └── __version__.py          # Version information
├── deployment/                 # Deployment scripts
│   └── deploy.sh              # Automated deployment
├── scripts/                   # Utility scripts
│   └── monitor.sh             # Monitoring script
├── docs/                      # Documentation
│   ├── deploy_guide.md        # Deployment guide
│   ├── quick_start.md         # Quick start guide
│   └── API.md                 # API documentation
├── tests/                     # Test suite
├── examples/                  # Example configurations
└── .github/                   # GitHub workflows
```

## 🔧 Advanced Configuration

See [docs/deploy_guide.md](docs/deploy_guide.md) for:
- Production deployment on Ubuntu/Debian
- Docker containerization
- Systemd service configuration
- Log rotation and monitoring
- Security hardening

## 📚 Documentation

- [Quick Start Guide](docs/quick_start.md) - Get running in 5 minutes
- [Deployment Guide](docs/deploy_guide.md) - Production deployment
- [API Documentation](docs/API.md) - Code reference
- [Migration Guide](docs/MIGRATION.md) - Upgrade instructions

## 🧪 Testing

```bash
# Install test dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run with coverage
pytest --cov=arbitrage_bot tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Always understand the risks and trade responsibly.

## 🙏 Acknowledgments

- [python-binance](https://github.com/sammchardy/python-binance) - Binance API wrapper
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) - Telegram bot framework
- [websockets](https://github.com/aaugustin/websockets) - WebSocket implementation

## 📞 Support

- 🐛 Bug reports: [GitHub Issues](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/issues)
- 💬 Questions: [GitHub Discussions](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/discussions)
- 📧 Email: noreply@example.com

---

**⭐ Star this repository if you find it useful!**