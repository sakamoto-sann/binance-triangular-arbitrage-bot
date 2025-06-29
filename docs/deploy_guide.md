# Complete Deployment Guide: Triangular Arbitrage Bot on Contabo Ubuntu 20.04

## 📋 Prerequisites

- Contabo VPS with Ubuntu 20.04
- Root or sudo access
- Binance API keys (testnet recommended first)
- SSH client (Terminal/PuTTY)

---

## 🔧 Step 1: Server Setup and Security

### 1.1 Connect to Your Server
```bash
ssh root@your_contabo_ip
# Or if you have a user account:
ssh username@your_contabo_ip
```

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.3 Install Essential Packages
```bash
sudo apt install -y curl wget git htop nano ufw python3 python3-pip python3-venv
```

### 1.4 Configure Firewall (Optional but Recommended)
```bash
# Enable UFW
sudo ufw enable

# Allow SSH (adjust port if you use non-standard)
sudo ufw allow 22/tcp

# Allow only essential ports
sudo ufw status
```

### 1.5 Create Non-Root User (Security Best Practice)
```bash
# Create user for the bot
sudo useradd -m -s /bin/bash botuser
sudo usermod -aG sudo botuser

# Set password
sudo passwd botuser

# Switch to bot user
sudo su - botuser
```

---

## 🐍 Step 2: Python Environment Setup

### 2.1 Verify Python Version
```bash
python3 --version
# Should show Python 3.8+ (Ubuntu 20.04 has 3.8)
```

### 2.2 Create Project Directory
```bash
mkdir -p ~/arbitrage_bot
cd ~/arbitrage_bot
```

### 2.3 Create Python Virtual Environment
```bash
python3 -m venv bot_env
source bot_env/bin/activate

# Verify virtual environment
which python
# Should show: /home/botuser/arbitrage_bot/bot_env/bin/python
```

### 2.4 Upgrade pip
```bash
pip install --upgrade pip
```

---

## 📁 Step 3: Upload Bot Files

### Option A: Using SCP (from your local machine)
```bash
# From your local machine, upload the bot files
scp -r /Users/tetsu/Documents/Binance_bot/v0.3/* botuser@your_contabo_ip:~/arbitrage_bot/
```

### Option B: Using Git (if you pushed to GitHub)
```bash
# On the server
git clone https://github.com/sakamoto-sann/url-summarizer-api.git temp_repo
cp temp_repo/arbitrage_bot/* ~/arbitrage_bot/
rm -rf temp_repo
```

### Option C: Manual File Transfer
Create each file manually using nano:

```bash
# Create config.py
nano ~/arbitrage_bot/config.py
# Copy and paste the config.py content, then Ctrl+X, Y, Enter

# Create arbitrage_bot.py  
nano ~/arbitrage_bot/arbitrage_bot.py
# Copy and paste the arbitrage_bot.py content, then Ctrl+X, Y, Enter

# Create requirements.txt
nano ~/arbitrage_bot/requirements.txt
# Copy and paste the requirements.txt content, then Ctrl+X, Y, Enter
```

---

## 📦 Step 4: Install Dependencies

### 4.1 Install Python Packages
```bash
cd ~/arbitrage_bot
source bot_env/bin/activate
pip install -r requirements.txt
```

### 4.2 Verify Installation
```bash
pip list
# Should show: python-binance, websockets, etc.
```

---

## 🔐 Step 5: Configure API Keys and Settings

### 5.1 Set Environment Variables
```bash
# Create environment file
nano ~/.bot_env
```

Add this content:
```bash
export BINANCE_API_KEY="your_binance_api_key_here"
export BINANCE_API_SECRET="your_binance_api_secret_here"
export BINANCE_TESTNET="true"  # Remove this line for live trading
```

Save and exit (Ctrl+X, Y, Enter)

### 5.2 Load Environment Variables
```bash
source ~/.bot_env

# Add to bashrc for permanent loading
echo "source ~/.bot_env" >> ~/.bashrc
```

### 5.3 Configure Bot Settings
```bash
nano ~/arbitrage_bot/config.py
```

Adjust these settings for your needs:
```python
# Start with smaller amounts for testing
TRADE_AMOUNT_USDT = 10.0  # Start small!
MIN_PROFIT_THRESHOLD = 0.002  # 0.2% minimum profit
MAX_DAILY_LOSS_USDT = 50.0  # Conservative daily loss limit
```

---

## 🧪 Step 6: Test the Bot

### 6.1 Test Python Environment
```bash
cd ~/arbitrage_bot
source bot_env/bin/activate
python3 -c "import binance; print('Binance module imported successfully')"
```

### 6.2 Test API Connection
```bash
python3 -c "
import os
from binance.client import Client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
if api_key and api_secret:
    print('API keys loaded successfully')
    try:
        client = Client(api_key, api_secret, testnet=True)
        info = client.get_account()
        print('API connection successful!')
        print(f'Account type: {info.get(\"accountType\", \"Unknown\")}')
    except Exception as e:
        print(f'API connection failed: {e}')
else:
    print('API keys not found in environment')
"
```

### 6.3 Dry Run Test
```bash
# Test the bot without actual trading
python3 arbitrage_bot.py
# Press Ctrl+C to stop after seeing some log output
```

---

## 🚀 Step 7: Production Deployment with systemd

### 7.1 Create systemd Service File
```bash
sudo nano /etc/systemd/system/arbitrage-bot.service
```

Add this content:
```ini
[Unit]
Description=Triangular Arbitrage Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
Group=botuser
WorkingDirectory=/home/botuser/arbitrage_bot
Environment=PATH=/home/botuser/arbitrage_bot/bot_env/bin
EnvironmentFile=/home/botuser/.bot_env
ExecStart=/home/botuser/arbitrage_bot/bot_env/bin/python arbitrage_bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/botuser/arbitrage_bot

[Install]
WantedBy=multi-user.target
```

### 7.2 Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable arbitrage-bot.service

# Start the service
sudo systemctl start arbitrage-bot.service

# Check status
sudo systemctl status arbitrage-bot.service
```

---

## 📊 Step 8: Monitoring and Maintenance

### 8.1 View Real-time Logs
```bash
# Follow live logs
sudo journalctl -u arbitrage-bot.service -f

# View recent logs
sudo journalctl -u arbitrage-bot.service --since "1 hour ago"

# View bot log file
tail -f ~/arbitrage_bot/arbitrage_bot.log
```

### 8.2 Control the Service
```bash
# Stop the bot
sudo systemctl stop arbitrage-bot.service

# Start the bot  
sudo systemctl start arbitrage-bot.service

# Restart the bot
sudo systemctl restart arbitrage-bot.service

# Check if running
sudo systemctl is-active arbitrage-bot.service
```

### 8.3 Monitor System Resources
```bash
# Check CPU and memory usage
htop

# Check disk space
df -h

# Check network connections
netstat -tulpn | grep python
```

---

## 🔧 Step 9: Maintenance Scripts

### 9.1 Create Update Script
```bash
nano ~/update_bot.sh
```

Add this content:
```bash
#!/bin/bash
echo "Updating Arbitrage Bot..."

# Stop the service
sudo systemctl stop arbitrage-bot.service

# Backup current version
cp -r ~/arbitrage_bot ~/arbitrage_bot_backup_$(date +%Y%m%d_%H%M%S)

# Update dependencies
cd ~/arbitrage_bot
source bot_env/bin/activate
pip install --upgrade -r requirements.txt

# Restart service
sudo systemctl start arbitrage-bot.service
sudo systemctl status arbitrage-bot.service

echo "Bot updated and restarted!"
```

Make it executable:
```bash
chmod +x ~/update_bot.sh
```

### 9.2 Create Backup Script
```bash
nano ~/backup_bot.sh
```

Add this content:
```bash
#!/bin/bash
BACKUP_DIR="~/bot_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup bot files and logs
tar -czf "$BACKUP_DIR/arbitrage_bot_backup_$TIMESTAMP.tar.gz" \
    -C ~/ arbitrage_bot/ .bot_env

echo "Backup created: $BACKUP_DIR/arbitrage_bot_backup_$TIMESTAMP.tar.gz"

# Keep only last 7 backups
ls -t $BACKUP_DIR/*.tar.gz | tail -n +8 | xargs rm -f
```

Make it executable:
```bash
chmod +x ~/backup_bot.sh
```

### 9.3 Setup Automatic Backups (Optional)
```bash
# Edit crontab
crontab -e

# Add this line for daily backups at 2 AM
0 2 * * * /home/botuser/backup_bot.sh
```

---

## 🚨 Step 10: Security Hardening

### 10.1 Secure File Permissions
```bash
# Protect environment file
chmod 600 ~/.bot_env

# Protect bot directory
chmod 750 ~/arbitrage_bot
```

### 10.2 Log Rotation Setup
```bash
sudo nano /etc/logrotate.d/arbitrage-bot
```

Add this content:
```
/home/botuser/arbitrage_bot/arbitrage_bot.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su botuser botuser
}
```

---

## 📋 Step 11: Final Checklist

✅ **Before Going Live:**

1. **Test on Binance Testnet first**
   ```bash
   # In config.py or environment
   export BINANCE_TESTNET="true"
   ```

2. **Start with small amounts**
   ```python
   TRADE_AMOUNT_USDT = 10.0  # Very small for testing
   ```

3. **Monitor closely for first 24 hours**
   ```bash
   sudo journalctl -u arbitrage-bot.service -f
   ```

4. **Verify API permissions**
   - Spot trading enabled
   - No futures/margin permissions needed
   - IP whitelist configured (if used)

5. **Check balance regularly**
   ```bash
   # Add to your monitoring routine
   python3 -c "
   import os
   from binance.client import Client
   client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
   account = client.get_account()
   print('Balances:')
   for balance in account['balances']:
       if float(balance['free']) > 0:
           print(f'{balance[\"asset\"]}: {balance[\"free\"]}')
   "
   ```

---

## 🆘 Troubleshooting

### Common Issues:

**Bot won't start:**
```bash
# Check logs
sudo journalctl -u arbitrage-bot.service -n 50

# Check Python path
which python3
source ~/arbitrage_bot/bot_env/bin/activate
which python
```

**API connection failed:**
```bash
# Check environment variables
echo $BINANCE_API_KEY
echo $BINANCE_API_SECRET

# Test API manually
python3 -c "from binance.client import Client; print('Testing...'); c = Client('test', 'test'); print('Module works')"
```

**Permission denied:**
```bash
# Fix file permissions
sudo chown -R botuser:botuser ~/arbitrage_bot
chmod +x ~/arbitrage_bot/arbitrage_bot.py
```

**Out of memory:**
```bash
# Check memory usage
free -h
# Consider upgrading VPS if needed
```

## 🎯 Success Indicators

Your bot is working correctly when you see:
- WebSocket connection established
- Exchange info loaded successfully  
- Price updates in logs
- No critical errors in journalctl
- Service status shows "active (running)"

**Ready to trade! Start with testnet and small amounts.** 🚀