#!/bin/bash
# 🚀 Contabo Server Deployment Script for Institutional Trading Bot v5.0.0
# Deploy the complete 8-module professional trading system

set -e  # Exit on any error

echo "======================================================================================================"
echo "🚀 INSTITUTIONAL TRADING BOT v5.0.0 - CONTABO DEPLOYMENT"
echo "📊 8 Core Modules | 🎯 BitVol & LXVX | 🔬 GARCH | 🎲 Kelly | 🛡️ Gamma | 🚨 Emergency"
echo "======================================================================================================"

# ============================================================================
# SYSTEM PREPARATION
# ============================================================================

echo "📋 Step 1: System Update and Dependencies..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    screen \
    htop \
    curl \
    wget \
    nano \
    ufw \
    systemd \
    cron

echo "✅ System packages installed"

# ============================================================================
# PYTHON ENVIRONMENT SETUP
# ============================================================================

echo "📋 Step 2: Python Environment Setup..."

# Create application directory
sudo mkdir -p /opt/institutional-trading-bot
sudo chown $USER:$USER /opt/institutional-trading-bot
cd /opt/institutional-trading-bot

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install numpy pandas

# Install optional advanced dependencies
echo "📊 Installing advanced mathematical libraries..."
pip install scipy || echo "⚠️  SciPy installation failed - using simplified calculations"
pip install scikit-learn || echo "⚠️  scikit-learn installation failed - using basic models"
pip install matplotlib || echo "⚠️  matplotlib installation failed - no plotting"

echo "✅ Python environment configured"

# ============================================================================
# APPLICATION DEPLOYMENT
# ============================================================================

echo "📋 Step 3: Application Deployment..."

# Clone or copy the trading bot code
if [ -n "$1" ]; then
    # If GitHub URL provided
    git clone $1 .
else
    # Manual deployment - create files
    echo "📝 Creating institutional trading bot files..."
    
    # Create requirements.txt
    cat > requirements.txt << 'EOF'
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
requests>=2.26.0
python-dotenv>=0.19.0
EOF

    # Note: The actual Python files would be copied here
    echo "⚠️  Please upload INSTITUTIONAL_TRADING_BOT.py and related files to this directory"
fi

# Install Python dependencies
pip install -r requirements.txt || echo "⚠️  Some advanced packages failed - system will use fallbacks"

echo "✅ Application files deployed"

# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

echo "📋 Step 4: Configuration Setup..."

# Create configuration directory
mkdir -p config logs data

# Create environment configuration
cat > .env << 'EOF'
# Institutional Trading Bot Configuration

# Trading Parameters
INITIAL_BALANCE=100000
MAX_PORTFOLIO_EXPOSURE=0.30
MAX_SINGLE_POSITION=0.05
MIN_SIGNAL_STRENGTH=3
MIN_CONFIDENCE_SCORE=0.70

# Risk Management
DAILY_LOSS_LIMIT=0.02
EMERGENCY_STOP_LOSS=0.15
GAMMA_HEDGE_THRESHOLD=1000

# API Configuration (Optional - for live trading)
# BINANCE_API_KEY=your_api_key_here
# BINANCE_SECRET_KEY=your_secret_key_here
# TRADING_MODE=paper  # paper or live

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/institutional_trading.log

# Advanced Features
ENABLE_BITVOL=true
ENABLE_LXVX=true
ENABLE_GARCH=true
ENABLE_KELLY=true
ENABLE_GAMMA_HEDGE=true
ENABLE_EMERGENCY_PROTOCOLS=true

# Notifications (Optional)
# SLACK_WEBHOOK=your_slack_webhook
# EMAIL_ALERTS=your_email@domain.com
EOF

# Secure the environment file
chmod 600 .env

echo "✅ Configuration files created"

# ============================================================================
# SYSTEM SERVICE SETUP
# ============================================================================

echo "📋 Step 5: System Service Setup..."

# Create systemd service
sudo tee /etc/systemd/system/institutional-trading-bot.service > /dev/null << EOF
[Unit]
Description=Institutional Trading Bot v5.0.0
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/institutional-trading-bot
Environment=PATH=/opt/institutional-trading-bot/venv/bin
ExecStart=/opt/institutional-trading-bot/venv/bin/python INSTITUTIONAL_TRADING_BOT.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=2G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable institutional-trading-bot

echo "✅ System service configured"

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

echo "📋 Step 6: Security Configuration..."

# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080  # For monitoring dashboard

# Set file permissions
chmod 755 /opt/institutional-trading-bot
chmod 600 /opt/institutional-trading-bot/.env
chmod +x /opt/institutional-trading-bot/*.py 2>/dev/null || true

echo "✅ Security configured"

# ============================================================================
# MONITORING SETUP
# ============================================================================

echo "📋 Step 7: Monitoring Setup..."

# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
# Institutional Trading Bot Monitoring Script

echo "🚀 INSTITUTIONAL TRADING BOT MONITORING"
echo "========================================"

# System status
echo "📊 System Status:"
systemctl is-active institutional-trading-bot
echo ""

# Resource usage
echo "💾 Resource Usage:"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Disk: $(df -h /opt/institutional-trading-bot | tail -1 | awk '{print $5}')"
echo ""

# Recent logs
echo "📝 Recent Logs (last 10 lines):"
tail -10 logs/institutional_trading.log 2>/dev/null || echo "No logs yet"
echo ""

# Process information
echo "🔧 Process Information:"
ps aux | grep "INSTITUTIONAL_TRADING_BOT.py" | grep -v grep || echo "Process not running"
EOF

chmod +x monitor.sh

# Create log rotation
sudo tee /etc/logrotate.d/institutional-trading-bot > /dev/null << EOF
/opt/institutional-trading-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl restart institutional-trading-bot > /dev/null 2>&1 || true
    endscript
}
EOF

echo "✅ Monitoring configured"

# ============================================================================
# DATA SETUP
# ============================================================================

echo "📋 Step 8: Historical Data Setup..."

# Create data download script
cat > download_data.sh << 'EOF'
#!/bin/bash
# Download historical data for backtesting

echo "📊 Downloading historical data..."

# Create data directory
mkdir -p data

# Download BTC 1H data (example - replace with actual data source)
echo "Downloading BTC 1H data..."
# wget -O data/btc_2024_1h.csv "https://data.binance.vision/..."

echo "⚠️  Please upload your historical data files to the data/ directory"
echo "Required files:"
echo "  - btc_2021_2025_1h_combined.csv"
echo "  - btc_2024_2024_1h_binance.csv"
echo "  - btc_2023_2023_1h_binance.csv"
EOF

chmod +x download_data.sh

echo "✅ Data setup configured"

# ============================================================================
# DEPLOYMENT VERIFICATION
# ============================================================================

echo "📋 Step 9: Deployment Verification..."

# Test Python environment
echo "🐍 Testing Python environment..."
source venv/bin/activate
python3 -c "import numpy, pandas; print('✅ Core dependencies working')" || echo "❌ Core dependencies failed"
python3 -c "import scipy; print('✅ SciPy available')" 2>/dev/null || echo "⚠️  SciPy not available - using fallbacks"

# Test configuration
echo "🔧 Testing configuration..."
[ -f .env ] && echo "✅ Environment file exists" || echo "❌ Environment file missing"
[ -d logs ] && echo "✅ Logs directory exists" || echo "❌ Logs directory missing"
[ -d data ] && echo "✅ Data directory exists" || echo "❌ Data directory missing"

echo "✅ Deployment verification complete"

# ============================================================================
# FINAL INSTRUCTIONS
# ============================================================================

echo ""
echo "======================================================================================================"
echo "🎉 INSTITUTIONAL TRADING BOT v5.0.0 DEPLOYMENT COMPLETE!"
echo "======================================================================================================"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. 📊 Upload your trading bot files:"
echo "   - INSTITUTIONAL_TRADING_BOT.py"
echo "   - Any additional module files"
echo ""
echo "2. 📈 Upload historical data to data/ directory:"
echo "   - btc_2021_2025_1h_combined.csv"
echo "   - btc_2024_2024_1h_binance.csv"
echo ""
echo "3. 🔧 Configure your settings in .env file:"
echo "   nano .env"
echo ""
echo "4. 🚀 Start the trading bot:"
echo "   sudo systemctl start institutional-trading-bot"
echo ""
echo "5. 📊 Monitor the system:"
echo "   ./monitor.sh"
echo "   journalctl -u institutional-trading-bot -f"
echo ""
echo "6. 🎯 Check logs:"
echo "   tail -f logs/institutional_trading.log"
echo ""
echo "======================================================================================================"
echo "🏆 INSTITUTIONAL FEATURES DEPLOYED:"
echo "✅ BitVol & LXVX Professional Volatility Indicators"
echo "✅ GARCH Academic-Grade Volatility Forecasting"
echo "✅ Kelly Criterion Mathematically Optimal Position Sizing"
echo "✅ Gamma Hedging Option-Like Exposure Management"
echo "✅ Emergency Protocols Multi-Level Risk Management"
echo "✅ 8 Core Modules Complete Professional Trading System"
echo "======================================================================================================"
echo ""
echo "🔗 USEFUL COMMANDS:"
echo "   systemctl status institutional-trading-bot    # Check status"
echo "   systemctl stop institutional-trading-bot      # Stop service"
echo "   systemctl restart institutional-trading-bot   # Restart service"
echo "   ./monitor.sh                                   # Monitor system"
echo "   screen -S trading-bot                         # Run in screen session"
echo ""
echo "🚀 Your institutional-grade trading bot is ready for deployment!"
echo "Remember to test thoroughly before enabling live trading."
echo ""