# Migration Guide: v0.3.x → v1.0.0

This guide helps you upgrade from version 0.3.x to 1.0.0, which includes critical security fixes and breaking changes.

## ⚠️ Critical Security Notice

**Version 0.3.x contains critical vulnerabilities that cause guaranteed trading losses and potential API key exposure. Immediate upgrade to v1.0.0 is required.**

## 🚨 Pre-Migration Checklist

Before upgrading:

- [ ] **Stop the bot immediately** - v0.3.x has trading loss bugs
- [ ] **Backup your configuration** - Save your current settings
- [ ] **Document your setup** - Note your current environment variables
- [ ] **Test environment ready** - Ensure testnet access for validation

## 📦 Breaking Changes

### 1. Repository Structure

**Before (v0.3.x):**
```
├── arbitrage_bot.py
├── config.py
├── requirements.txt
├── deploy.sh
└── README.md
```

**After (v1.0.0):**
```
├── src/arbitrage_bot/
│   ├── arbitrage_bot.py
│   ├── config.py
│   ├── telegram_notifier.py
│   └── __version__.py
├── deployment/
│   └── deploy.sh
├── scripts/
│   └── monitor.sh
├── docs/
└── tests/
```

### 2. Import Changes

**Before:**
```python
import arbitrage_bot
import config
```

**After:**
```python
from arbitrage_bot import arbitrage_bot
from arbitrage_bot import config
```

### 3. Command Line Usage

**Before:**
```bash
python arbitrage_bot.py
```

**After:**
```bash
# Method 1: Module execution
python -m arbitrage_bot.arbitrage_bot

# Method 2: Entry point (after pip install)
arbitrage-bot

# Method 3: Direct execution
python src/arbitrage_bot/arbitrage_bot.py
```

## 🔧 Configuration Changes

### Environment Variables

**New variables in v1.0.0:**
```bash
# Telegram notifications (optional)
export TELEGRAM_ENABLED="true"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_IDS="chat_id1,chat_id2"
```

**Existing variables (unchanged):**
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export BINANCE_TESTNET="true"
```

### Configuration File Updates

The `config.py` structure remains largely the same, but with additional Telegram settings:

```python
# New in v1.0.0
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [id.strip() for id in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if id.strip()]
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true" and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS

NOTIFICATION_SETTINGS = {
    "trade_execution": {"enabled": True, "rate_limit_seconds": 1, "max_per_hour": 30},
    "opportunity": {"enabled": True, "rate_limit_seconds": 10, "max_per_hour": 20},
    "error": {"enabled": True, "rate_limit_seconds": 5, "max_per_hour": 15},
    "daily_summary": {"enabled": True, "rate_limit_seconds": 0, "max_per_hour": 1},
    "status": {"enabled": True, "rate_limit_seconds": 30, "max_per_hour": 10"}
}
```

## 📋 Step-by-Step Migration

### Step 1: Backup Current Setup

```bash
# Stop the current bot
sudo systemctl stop arbitrage-bot.service  # If using systemd

# Backup your environment
cp ~/.bot_env ~/.bot_env.backup

# Backup your configuration
cp config.py config.py.backup
```

### Step 2: Download v1.0.0

```bash
# Pull the latest version
git pull origin main

# Or clone fresh
git clone https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git v1.0.0
cd v1.0.0
```

### Step 3: Install Dependencies

```bash
# Create new virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Step 4: Migrate Configuration

```bash
# Copy your backed up environment settings
cp ~/.bot_env.backup ~/.bot_env

# Add new Telegram settings (optional)
echo 'export TELEGRAM_ENABLED="false"' >> ~/.bot_env
echo 'export TELEGRAM_BOT_TOKEN=""' >> ~/.bot_env
echo 'export TELEGRAM_CHAT_IDS=""' >> ~/.bot_env
```

### Step 5: Test the Migration

```bash
# Test import
python -c "from arbitrage_bot import config; print('Config loaded successfully')"

# Test bot startup (dry run)
python -m arbitrage_bot.arbitrage_bot --dry-run  # If implemented

# Test with testnet
export BINANCE_TESTNET="true"
python -m arbitrage_bot.arbitrage_bot
```

### Step 6: Update Deployment Scripts

**If using systemd:**

Update your service file to use the new structure:

```ini
[Unit]
Description=Triangular Arbitrage Bot v1.0.0
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/path/to/v1.0.0
Environment=PATH=/path/to/v1.0.0/venv/bin
EnvironmentFile=/home/botuser/.bot_env
ExecStart=/path/to/v1.0.0/venv/bin/python -m arbitrage_bot.arbitrage_bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**If using the deployment script:**

```bash
# Use the new deployment script
./deployment/deploy.sh
```

## 🔍 Validation Checklist

After migration, verify:

- [ ] **Bot starts successfully** without errors
- [ ] **API connection works** (test with small operation)
- [ ] **WebSocket connects** and receives price data
- [ ] **Logging works** and shows expected output
- [ ] **Telegram notifications work** (if enabled)
- [ ] **Configuration values** are loaded correctly
- [ ] **No critical errors** in logs

## 🚨 Critical Fixes in v1.0.0

### Trading Logic Fixes

1. **Profit Calculation**: Fixed formula that was causing guaranteed losses
   - **Impact**: Bot will now correctly identify profitable opportunities
   - **Before**: `profit_percentage = amount - 1.0` (WRONG)
   - **After**: `profit_percentage = (amount - 1.0) * 100` (CORRECT)

2. **Trade Amount Logic**: Fixed broken trade sizing
   - **Impact**: Bot will use proper trade amounts based on balance
   - **Before**: Always traded tiny amounts (2.5 USDT)
   - **After**: Uses actual balance with 5% limit

3. **Race Condition Prevention**: Added trade execution semaphore
   - **Impact**: Prevents multiple simultaneous trades
   - **Benefit**: No more insufficient balance errors

### Security Fixes

1. **Message Sanitization**: All outputs are now sanitized
   - **Impact**: API keys cannot leak through error messages
   - **Coverage**: Comprehensive regex patterns for all sensitive data

2. **Enhanced Error Handling**: Secure error messages
   - **Impact**: Internal state not exposed in errors
   - **Benefit**: Better security posture

## 🆘 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're using the new import paths
python -c "from arbitrage_bot import arbitrage_bot"
```

**Module not found:**
```bash
# Ensure you're in the right directory and environment
cd /path/to/v1.0.0
source venv/bin/activate
pip install -e .
```

**Configuration not loading:**
```bash
# Check environment variables
source ~/.bot_env
env | grep BINANCE
env | grep TELEGRAM
```

**Service won't start:**
```bash
# Check service configuration
sudo systemctl status arbitrage-bot.service
sudo journalctl -u arbitrage-bot.service -n 50
```

### Getting Help

If you encounter issues during migration:

1. **Check the logs** for specific error messages
2. **Verify environment** variables are set correctly
3. **Test each component** individually
4. **Use testnet** for safe testing
5. **Check GitHub Issues** for known problems
6. **Create an issue** if you need help

### Rollback Plan

If migration fails and you need to rollback:

```bash
# Stop new version
sudo systemctl stop arbitrage-bot.service

# Restore backup
cp ~/.bot_env.backup ~/.bot_env
cp config.py.backup config.py

# Return to previous version
git checkout v0.3.x  # Only as emergency fallback

# Note: v0.3.x has critical vulnerabilities - fix and upgrade ASAP
```

## 📞 Support

For migration assistance:

- 📖 **Documentation**: [README.md](../README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/discussions)
- 📧 **Email**: noreply@example.com

## ✅ Post-Migration

After successful migration:

1. **Monitor closely** for first 24 hours
2. **Verify profitability** calculations are correct
3. **Test Telegram notifications** if enabled
4. **Update documentation** and procedures
5. **Share feedback** about the migration process

**Congratulations on upgrading to v1.0.0! Your bot is now secure and reliable.** 🎉