# 🚀 Contabo Server Deployment Guide - Enhanced Trading Bot v3.1.0

## 📋 Pre-Deployment Checklist

### ✅ What We're Deploying
- **Enhanced Trading Bot v3.1.0** with institutional-grade risk management
- **3 Enhanced Features:** Drawdown control, volatility sizing, smart execution
- **Conservative Parameters:** 189.2% return, 42.5% annual, 3.50 Sharpe ratio
- **GitHub Repository:** https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git

---

## 🖥️ Step 1: Server Setup and Access

### Connect to Your Contabo Server
```bash
# SSH into your Contabo server
ssh root@YOUR_CONTABO_IP
# Or if you have a specific user
ssh username@YOUR_CONTABO_IP
```

### Update System
```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget htop screen nano python3 python3-pip python3-venv
```

---

## 🐍 Step 2: Python Environment Setup

### Install Python 3.9+ (if not already installed)
```bash
# Check Python version
python3 --version

# If needed, install Python 3.9+
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

### Create Virtual Environment
```bash
# Create dedicated directory for the bot
mkdir -p /opt/trading-bot
cd /opt/trading-bot

# Create virtual environment
python3 -m venv bot-env

# Activate virtual environment
source bot-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 📦 Step 3: Clone and Install Enhanced Bot

### Clone the Enhanced Repository
```bash
# Clone the enhanced v3.1.0 version
git clone https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git enhanced-bot
cd enhanced-bot

# Checkout the specific v3.1.0 tag
git checkout v3.1.0

# Verify we have the enhanced features
ls -la src/config/enhanced_features_config.py
```

### Install Dependencies
```bash
# Install required Python packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install numpy pandas matplotlib python-binance asyncio aiohttp websockets python-dotenv schedule logging

# Install additional packages for enhanced features
pip install scipy scikit-learn ta-lib  # Technical analysis libraries
```

---

## 🔐 Step 4: Configuration and API Keys

### Create Environment Configuration
```bash
# Create environment file
nano .env
```

Add your configuration:
```env
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=true  # Start with testnet for safety

# Enhanced Features Configuration
ENHANCED_FEATURES_ENABLED=true
PAPER_TRADING_MODE=true  # Start with paper trading

# Risk Management Settings (Conservative)
DRAWDOWN_HALT_THRESHOLD=0.20    # 20% drawdown halt
RECOVERY_THRESHOLD=0.10         # 10% recovery threshold
MAX_POSITION_SIZE=0.15          # 15% max single position

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/opt/trading-bot/logs/bot.log

# Server Configuration
ENVIRONMENT=production
SERVER_TYPE=contabo
```

### Set Proper Permissions
```bash
# Protect sensitive files
chmod 600 .env
chmod +x *.py

# Create logs directory
mkdir -p logs
mkdir -p data
```

---

## 🛠️ Step 5: Enhanced Features Configuration

### Verify Enhanced Features
```bash
# Test the enhanced features configuration
python3 -c "
from src.config.enhanced_features_config import get_config, is_feature_enabled
config = get_config()
print('Drawdown Control:', is_feature_enabled('drawdown_control'))
print('Volatility Sizing:', is_feature_enabled('volatility_sizing'))
print('Smart Execution:', is_feature_enabled('smart_execution'))
print('Strategy:', config['philosophy'])
"
```

Expected output:
```
Drawdown Control: True
Volatility Sizing: True
Smart Execution: True
Strategy: conservative
```

---

## 🧪 Step 6: Testing and Validation

### Run Final Validation Test
```bash
# Test the enhanced features integration
python3 final_validation_test.py
```

### Test Binance Connection (Testnet)
```bash
# Create a simple connection test
python3 -c "
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

client = Client(api_key, secret_key, testnet=True)
print('Testnet Connection:', client.ping())
print('Account Status:', client.get_account_status())
"
```

---

## 🚀 Step 7: Deployment Options

### Option A: Screen Session (Simple)
```bash
# Start a screen session
screen -S trading-bot

# Inside screen, activate environment and run bot
cd /opt/trading-bot/enhanced-bot
source ../bot-env/bin/activate

# Start with paper trading
python3 main.py --paper-trading --enhanced-features

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r trading-bot
```

### Option B: Systemd Service (Recommended)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/trading-bot.service
```

Add service configuration:
```ini
[Unit]
Description=Enhanced Trading Bot v3.1.0
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trading-bot/enhanced-bot
Environment=PATH=/opt/trading-bot/bot-env/bin
ExecStart=/opt/trading-bot/bot-env/bin/python main.py --enhanced-features --paper-trading
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
sudo journalctl -u trading-bot -f
```

---

## 📊 Step 8: Monitoring and Logging

### Set Up Log Rotation
```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/trading-bot
```

Add logrotate config:
```
/opt/trading-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
```

### Create Monitoring Script
```bash
# Create monitoring script
nano /opt/trading-bot/monitor.sh
```

Add monitoring script:
```bash
#!/bin/bash
# Enhanced Trading Bot Monitoring Script

LOG_FILE="/opt/trading-bot/logs/monitor.log"
BOT_LOG="/opt/trading-bot/logs/bot.log"

echo "$(date): Checking bot status..." >> $LOG_FILE

# Check if bot process is running
if systemctl is-active --quiet trading-bot; then
    echo "$(date): Bot is running" >> $LOG_FILE
else
    echo "$(date): Bot is not running - attempting restart" >> $LOG_FILE
    sudo systemctl restart trading-bot
fi

# Check for errors in bot log
if tail -100 $BOT_LOG | grep -i "error\|exception\|failed" > /dev/null; then
    echo "$(date): Errors detected in bot log" >> $LOG_FILE
fi

# Check enhanced features status
python3 -c "
from src.config.enhanced_features_config import get_config
config = get_config()
print('$(date): Enhanced features active:', config['philosophy'])
" >> $LOG_FILE
```

Make it executable and add to crontab:
```bash
chmod +x /opt/trading-bot/monitor.sh

# Add to crontab for monitoring every 5 minutes
crontab -e
# Add: */5 * * * * /opt/trading-bot/monitor.sh
```

---

## ⚠️ Step 9: Safety and Production Considerations

### Start with Paper Trading
```bash
# Always start with paper trading first
PAPER_TRADING_MODE=true
BINANCE_TESTNET=true

# Monitor for 24-48 hours before switching to live trading
```

### Security Checklist
- ✅ API keys have restricted permissions (only trading, no withdrawals)
- ✅ Environment file (.env) has restricted permissions (600)
- ✅ Server firewall is properly configured
- ✅ Regular backups of configuration and logs
- ✅ Monitor for unusual activity

### Performance Monitoring
```bash
# Create performance monitoring script
nano /opt/trading-bot/performance_check.py
```

Add performance monitoring:
```python
#!/usr/bin/env python3
"""
Performance monitoring for enhanced trading bot
Compares current performance against v3.1.0 targets
"""

import sys
sys.path.append('src')

from config.enhanced_features_config import get_config

def check_performance():
    config = get_config()
    targets = config['optimization_results']
    
    print(f"Target Total Return: {targets['total_return']}%")
    print(f"Target Sharpe Ratio: {targets['sharpe_ratio']}")
    print(f"Target Max Drawdown: {targets['max_drawdown']:.1%}")
    
    # Add your performance calculation logic here
    print("Performance monitoring active...")

if __name__ == "__main__":
    check_performance()
```

---

## 🎯 Step 10: Go Live Checklist

### Before Switching to Live Trading:

1. **✅ Paper Trading Validation** (24-48 hours minimum)
   - Verify all 3 enhanced features are working
   - Confirm drawdown control activates properly
   - Check position sizing is conservative (1.0x multipliers)
   - Validate order execution and timeout controls

2. **✅ Performance Monitoring**
   - Monitor against 189.2% total return target
   - Watch for 42.5% annual return performance
   - Ensure max drawdown stays under 6% (improved from 32.4%)

3. **✅ Risk Management Verification**
   - Test drawdown halt at 20% threshold
   - Verify recovery at 10% threshold
   - Confirm emergency stops are functional

4. **✅ Switch to Live Trading**
   ```bash
   # Update .env file
   BINANCE_TESTNET=false
   PAPER_TRADING_MODE=false
   
   # Restart service
   sudo systemctl restart trading-bot
   ```

---

## 📞 Support and Troubleshooting

### Common Issues and Solutions

**Issue: Bot won't start**
```bash
# Check logs
sudo journalctl -u trading-bot -f
# Check Python environment
source /opt/trading-bot/bot-env/bin/activate
python3 final_validation_test.py
```

**Issue: API connection problems**
```bash
# Test API connection
python3 -c "from binance.client import Client; print(Client().ping())"
# Check firewall and network
```

**Issue: Enhanced features not working**
```bash
# Verify configuration
python3 -c "from src.config.enhanced_features_config import get_config; print(get_config())"
```

### Emergency Procedures

**Emergency Stop:**
```bash
sudo systemctl stop trading-bot
```

**Emergency Contact:**
- Monitor logs: `tail -f /opt/trading-bot/logs/bot.log`
- Check system status: `sudo systemctl status trading-bot`
- Review performance: `python3 performance_check.py`

---

## 🏆 Deployment Success Metrics

**Target Performance (from v3.1.0 validation):**
- **Total Return:** 189.2% (target maintenance)
- **Annual Return:** 42.5% (98% of baseline)
- **Sharpe Ratio:** 3.50 (risk-adjusted returns)
- **Max Drawdown:** 6.0% (major improvement from 32.4%)

**Enhanced Features Active:**
- ✅ Portfolio-level drawdown control (conservative 20%/10%)
- ✅ Volatility-adjusted position sizing (neutral 1.0x multipliers)
- ✅ Smart order execution (market price, timeout controls)

**🎉 You're ready to deploy the enhanced trading bot v3.1.0 on Contabo!**