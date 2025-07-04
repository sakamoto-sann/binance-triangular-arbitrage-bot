# 🚀 Contabo Server Deployment Guide

## Ultimate Trading Bot v4.2.0 Production Deployment

### 📋 Prerequisites

**Server Requirements:**
- Ubuntu 20.04+ or CentOS 8+
- 2GB+ RAM (4GB recommended)
- 20GB+ storage
- Python 3.8+
- Git installed

**Access Requirements:**
- SSH access to Contabo server
- Sudo privileges
- Binance API credentials (optional for live trading)

### 🔧 Quick Deployment Script

Use the automated deployment script:

```bash
# Download and run deployment script
curl -sSL https://raw.githubusercontent.com/your-repo/binance-bot/main/deploy.sh | bash
```

Or follow manual steps below.

### 📖 Manual Deployment Steps

#### 1. Connect to Server
```bash
ssh root@your-contabo-server-ip
```

#### 2. Update System
```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip git screen htop
```

#### 3. Clone Repository
```bash
cd /opt
git clone https://github.com/your-username/binance-bot-v4-atr-enhanced.git
cd binance-bot-v4-atr-enhanced
```

#### 4. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Download Historical Data
```bash
# Download BTC historical data for backtesting
wget -O btc_2024_2024_1h_binance.csv "https://data.binance.vision/..."
# Or upload your existing data files
```

#### 6. Configure Environment
```bash
cp .env.example .env
nano .env  # Edit with your settings
```

#### 7. Test Installation
```bash
python ULTIMATE_TRADING_BOT.py
```

### 🔒 Security Configuration

#### Firewall Setup
```bash
ufw enable
ufw allow ssh
ufw allow 8080  # For monitoring dashboard (optional)
```

#### API Key Security
```bash
# Store API keys securely
echo "BINANCE_API_KEY=your_api_key" >> .env
echo "BINANCE_SECRET_KEY=your_secret_key" >> .env
chmod 600 .env
```

### 🔄 Production Run

#### Background Execution
```bash
# Run in screen session
screen -S trading-bot
source venv/bin/activate
python ULTIMATE_TRADING_BOT.py

# Detach: Ctrl+A, D
# Reattach: screen -r trading-bot
```

#### Systemd Service (Recommended)
```bash
sudo cp deployment/trading-bot.service /etc/systemd/system/
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

### 📊 Monitoring

#### Log Monitoring
```bash
# Real-time logs
tail -f logs/trading_bot.log

# System resource monitoring
htop
```

#### Performance Dashboard
```bash
# Optional: Run monitoring dashboard
python monitoring/dashboard.py
# Access at: http://your-server-ip:8080
```

### 🔄 Updates and Maintenance

#### Update Bot
```bash
cd /opt/binance-bot-v4-atr-enhanced
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart trading-bot
```

#### Backup Configuration
```bash
# Backup important files
tar -czf backup-$(date +%Y%m%d).tar.gz .env logs/ data/
```

### 🆘 Troubleshooting

#### Common Issues
```bash
# Check service status
sudo systemctl status trading-bot

# View recent logs
journalctl -u trading-bot -n 50

# Check resource usage
free -h
df -h
```

#### Performance Optimization
```bash
# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 📞 Support

- Check logs first: `/opt/binance-bot-v4-atr-enhanced/logs/`
- Review configuration: `.env` file
- Monitor resources: `htop` and `df -h`
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)