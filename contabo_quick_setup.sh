#!/bin/bash
# 🚀 Contabo Quick Setup Script for Enhanced Trading Bot v3.1.0
# Run this script on your Contabo server to automatically set up the enhanced trading bot

set -e  # Exit on any error

echo "🚀 Starting Enhanced Trading Bot v3.1.0 Setup on Contabo Server"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}$1${NC}"
    echo "----------------------------------------"
}

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   print_error "This script should be run as root or with sudo"
   exit 1
fi

print_header "Step 1: System Update and Dependencies"

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install essential packages
print_status "Installing essential packages..."
apt install -y git curl wget htop screen nano python3 python3-pip python3-venv python3-dev build-essential

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
print_status "Python version: $PYTHON_VERSION"

print_header "Step 2: Create Trading Bot Directory and Environment"

# Create bot directory
BOT_DIR="/opt/trading-bot"
print_status "Creating bot directory: $BOT_DIR"
mkdir -p $BOT_DIR
cd $BOT_DIR

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv bot-env

# Activate virtual environment
source bot-env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

print_header "Step 3: Clone Enhanced Trading Bot Repository"

# Clone the enhanced repository
print_status "Cloning enhanced trading bot v3.1.0..."
git clone https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git enhanced-bot
cd enhanced-bot

# Checkout specific version
print_status "Checking out v3.1.0 release..."
git checkout v3.1.0

# Verify enhanced features
if [[ -f "src/config/enhanced_features_config.py" ]]; then
    print_status "✅ Enhanced features configuration found"
else
    print_error "❌ Enhanced features configuration not found"
    exit 1
fi

print_header "Step 4: Install Python Dependencies"

# Install core dependencies
print_status "Installing Python packages..."
pip install numpy pandas matplotlib asyncio aiohttp websockets python-dotenv schedule

# Install Binance API
pip install python-binance

# Install additional packages for enhanced features
print_status "Installing enhanced features dependencies..."
pip install scikit-learn || print_warning "Could not install scikit-learn (optional)"

print_header "Step 5: Create Configuration Template"

# Create .env template
print_status "Creating environment configuration template..."
cat > .env.template << EOF
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=true

# Enhanced Features Configuration
ENHANCED_FEATURES_ENABLED=true
PAPER_TRADING_MODE=true

# Risk Management Settings (Conservative v3.1.0)
DRAWDOWN_HALT_THRESHOLD=0.20
RECOVERY_THRESHOLD=0.10
MAX_POSITION_SIZE=0.15

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/opt/trading-bot/logs/bot.log

# Server Configuration
ENVIRONMENT=production
SERVER_TYPE=contabo
EOF

# Set permissions
chmod 644 .env.template

print_header "Step 6: Create Directory Structure"

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p backups

# Set proper permissions
chmod 755 logs data backups

print_header "Step 7: Test Enhanced Features"

# Test enhanced features configuration
print_status "Testing enhanced features configuration..."
python3 -c "
try:
    import sys
    sys.path.append('src')
    from config.enhanced_features_config import get_config, is_feature_enabled
    config = get_config()
    print('✅ Enhanced Features Status:')
    print(f'  - Drawdown Control: {is_feature_enabled(\"drawdown_control\")}')
    print(f'  - Volatility Sizing: {is_feature_enabled(\"volatility_sizing\")}')
    print(f'  - Smart Execution: {is_feature_enabled(\"smart_execution\")}')
    print(f'  - Strategy: {config[\"philosophy\"]}')
    print(f'  - Version: {config[\"version\"]}')
    print('✅ Enhanced features successfully loaded!')
except Exception as e:
    print(f'❌ Error loading enhanced features: {e}')
    exit(1)
" && print_status "✅ Enhanced features test passed!" || print_error "❌ Enhanced features test failed!"

print_header "Step 8: Create Systemd Service"

# Create systemd service file
print_status "Creating systemd service..."
cat > /etc/systemd/system/trading-bot.service << EOF
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
EOF

# Reload systemd
systemctl daemon-reload

print_header "Step 9: Create Monitoring and Log Rotation"

# Create logrotate configuration
print_status "Setting up log rotation..."
cat > /etc/logrotate.d/trading-bot << EOF
/opt/trading-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Create monitoring script
print_status "Creating monitoring script..."
cat > /opt/trading-bot/monitor.sh << 'EOF'
#!/bin/bash
LOG_FILE="/opt/trading-bot/logs/monitor.log"
BOT_LOG="/opt/trading-bot/logs/bot.log"

echo "$(date): Checking enhanced bot status..." >> $LOG_FILE

if systemctl is-active --quiet trading-bot; then
    echo "$(date): Enhanced bot is running" >> $LOG_FILE
else
    echo "$(date): Enhanced bot is not running - attempting restart" >> $LOG_FILE
    systemctl restart trading-bot
fi

if [[ -f $BOT_LOG ]] && tail -100 $BOT_LOG | grep -i "error\|exception\|failed" > /dev/null; then
    echo "$(date): Errors detected in bot log" >> $LOG_FILE
fi
EOF

chmod +x /opt/trading-bot/monitor.sh

print_header "Step 10: Create Management Scripts"

# Create start script
cat > /opt/trading-bot/start_bot.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Enhanced Trading Bot v3.1.0..."
systemctl start trading-bot
systemctl status trading-bot
EOF

# Create stop script
cat > /opt/trading-bot/stop_bot.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Enhanced Trading Bot..."
systemctl stop trading-bot
EOF

# Create status script
cat > /opt/trading-bot/check_status.sh << 'EOF'
#!/bin/bash
echo "📊 Enhanced Trading Bot Status:"
echo "================================"
systemctl status trading-bot
echo ""
echo "📝 Recent Logs:"
echo "==============="
journalctl -u trading-bot -n 20 --no-pager
EOF

# Create performance check script
cat > /opt/trading-bot/check_performance.sh << 'EOF'
#!/bin/bash
cd /opt/trading-bot/enhanced-bot
source ../bot-env/bin/activate

echo "📊 Enhanced Trading Bot Performance Check"
echo "========================================="
echo "Target Performance (v3.1.0):"
echo "- Total Return: 189.2%"
echo "- Annual Return: 42.5%"
echo "- Sharpe Ratio: 3.50"
echo "- Max Drawdown: 6.0%"
echo ""

python3 -c "
import sys
sys.path.append('src')
from config.enhanced_features_config import get_config
config = get_config()
print('Current Configuration:')
print(f'Strategy: {config[\"philosophy\"]}')
print(f'Version: {config[\"version\"]}')
print(f'Optimization Results: {config[\"optimization_results\"]}')
"
EOF

# Make scripts executable
chmod +x /opt/trading-bot/*.sh

print_header "Setup Complete!"

print_status "✅ Enhanced Trading Bot v3.1.0 setup completed successfully!"
echo ""
echo -e "${GREEN}📋 Next Steps:${NC}"
echo "1. Configure your API keys:"
echo "   cp /opt/trading-bot/enhanced-bot/.env.template /opt/trading-bot/enhanced-bot/.env"
echo "   nano /opt/trading-bot/enhanced-bot/.env"
echo ""
echo "2. Test the configuration:"
echo "   cd /opt/trading-bot/enhanced-bot"
echo "   source ../bot-env/bin/activate"
echo "   python3 final_validation_test.py"
echo ""
echo "3. Start with paper trading:"
echo "   /opt/trading-bot/start_bot.sh"
echo ""
echo "4. Monitor the bot:"
echo "   /opt/trading-bot/check_status.sh"
echo "   /opt/trading-bot/check_performance.sh"
echo ""
echo -e "${YELLOW}⚠️  Important:${NC}"
echo "- Start with PAPER_TRADING_MODE=true and BINANCE_TESTNET=true"
echo "- Monitor for 24-48 hours before switching to live trading"
echo "- The enhanced features include conservative risk management"
echo ""
echo -e "${GREEN}🎯 Enhanced Features Active:${NC}"
echo "- Portfolio-level drawdown control (20% halt, 10% recovery)"
echo "- Volatility-adjusted position sizing (conservative 1.0x)"
echo "- Smart order execution with timeout controls"
echo ""
echo -e "${BLUE}🏆 Target Performance: 189.2% return, 42.5% annual, 3.50 Sharpe${NC}"
echo ""
echo "Setup log saved to: /opt/trading-bot/setup.log"

# Save setup info
echo "Enhanced Trading Bot v3.1.0 setup completed at $(date)" > /opt/trading-bot/setup.log
echo "Repository: https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot.git" >> /opt/trading-bot/setup.log
echo "Version: v3.1.0" >> /opt/trading-bot/setup.log
echo "Enhanced Features: Active" >> /opt/trading-bot/setup.log