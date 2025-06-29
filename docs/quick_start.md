# 🚀 Quick Start Guide - Contabo Ubuntu 20.04

## Fast Deployment (5 Minutes)

### 1. Connect to Your Server
```bash
ssh root@your_contabo_ip
# or 
ssh username@your_contabo_ip
```

### 2. Download and Run Auto-Deploy Script
```bash
# Create user (if needed)
sudo useradd -m -s /bin/bash botuser
sudo usermod -aG sudo botuser
sudo passwd botuser
sudo su - botuser

# Download deployment files
mkdir -p ~/arbitrage_bot
cd ~/arbitrage_bot

# Copy the bot files here (use scp, git, or manual copy)
# Then run the auto-deploy script:
chmod +x deploy.sh
./deploy.sh
```

### 3. Configure API Keys
```bash
nano ~/.bot_env
```
Replace with your actual keys:
```bash
export BINANCE_API_KEY="your_actual_api_key"
export BINANCE_API_SECRET="your_actual_api_secret"
export BINANCE_TESTNET="true"  # Start with testnet!
```

### 4. Start the Bot
```bash
# Start the service
sudo systemctl start arbitrage-bot.service

# Check status
./monitor.sh status

# Watch logs
./monitor.sh logs
```

## Essential Commands

```bash
# Service Control
sudo systemctl start arbitrage-bot.service     # Start
sudo systemctl stop arbitrage-bot.service      # Stop  
sudo systemctl restart arbitrage-bot.service   # Restart
sudo systemctl status arbitrage-bot.service    # Status

# Monitoring
./monitor.sh dashboard    # Full dashboard
./monitor.sh live        # Live monitoring
./monitor.sh health      # Health check
./monitor.sh api         # Test API connection

# Maintenance  
./update_bot.sh          # Update bot
./backup_bot.sh          # Create backup
./bot_status.sh          # Quick status
```

## Safety Checklist ✅

Before going live:
- [ ] ✅ Test on TESTNET first (`BINANCE_TESTNET="true"`)
- [ ] ✅ Start with small amounts (`TRADE_AMOUNT_USDT = 10.0`)
- [ ] ✅ Monitor for 24 hours continuously
- [ ] ✅ Check API permissions (spot trading only)
- [ ] ✅ Verify balance regularly
- [ ] ✅ Set up alerts/notifications

## Troubleshooting

**Bot won't start:**
```bash
sudo journalctl -u arbitrage-bot.service -n 50
```

**API issues:**
```bash
./monitor.sh api
```

**Check health:**
```bash
./monitor.sh health
```

**Reset everything:**
```bash
sudo systemctl stop arbitrage-bot.service
cd ~/arbitrage_bot
source bot_env/bin/activate
python3 arbitrage_bot.py  # Test manually
```

## File Structure
```
~/arbitrage_bot/
├── arbitrage_bot.py      # Main bot
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── bot_env/             # Virtual environment
├── arbitrage_bot.log    # Log file
├── deploy.sh            # Auto-deploy script
└── monitor.sh           # Monitoring script

~/.bot_env               # Environment variables
```

**Ready to trade! 🎯**