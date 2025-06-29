#!/bin/bash

# Triangular Arbitrage Bot - Automated Deployment Script for Ubuntu 20.04
# Usage: ./deploy.sh

set -e  # Exit on any error

echo "🚀 Starting Triangular Arbitrage Bot Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   print_status "Please run as a regular user with sudo privileges"
   exit 1
fi

# Check if sudo is available
if ! command -v sudo &> /dev/null; then
    print_error "sudo is required but not installed"
    exit 1
fi

print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "Installing essential packages..."
sudo apt install -y curl wget git htop nano ufw python3 python3-pip python3-venv

print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
print_status "Python version: $PYTHON_VERSION"

# Create project directory
PROJECT_DIR="$HOME/arbitrage_bot"
print_status "Creating project directory at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv bot_env
source bot_env/bin/activate

print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
else
    print_status "Installing Python dependencies manually..."
    pip install python-binance websockets asyncio
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_status "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
python-binance>=1.0.16
websockets>=10.4
asyncio
decimal
EOF
fi

# Create environment file template
ENV_FILE="$HOME/.bot_env"
if [ ! -f "$ENV_FILE" ]; then
    print_status "Creating environment file template..."
    cat > "$ENV_FILE" << 'EOF'
# Binance API Configuration
export BINANCE_API_KEY="your_binance_api_key_here"
export BINANCE_API_SECRET="your_binance_api_secret_here"
export BINANCE_TESTNET="true"  # Set to false for live trading

# Optional: Telegram notifications (if implemented)
# export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
# export TELEGRAM_CHAT_ID="your_telegram_chat_id"
EOF
    chmod 600 "$ENV_FILE"
    
    print_warning "Environment file created at $ENV_FILE"
    print_warning "Please edit this file with your actual API keys!"
fi

# Add environment loading to bashrc
if ! grep -q "source ~/.bot_env" ~/.bashrc; then
    print_status "Adding environment loading to ~/.bashrc..."
    echo "source ~/.bot_env" >> ~/.bashrc
fi

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/arbitrage-bot.service > /dev/null << EOF
[Unit]
Description=Triangular Arbitrage Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/bot_env/bin
EnvironmentFile=$ENV_FILE
ExecStart=$PROJECT_DIR/bot_env/bin/python arbitrage_bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation config
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/arbitrage-bot > /dev/null << EOF
$PROJECT_DIR/arbitrage_bot.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $USER $USER
}
EOF

# Create management scripts
print_status "Creating management scripts..."

# Update script
cat > "$HOME/update_bot.sh" << 'EOF'
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
EOF

# Backup script
cat > "$HOME/backup_bot.sh" << 'EOF'
#!/bin/bash
BACKUP_DIR="$HOME/bot_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup bot files and logs
tar -czf "$BACKUP_DIR/arbitrage_bot_backup_$TIMESTAMP.tar.gz" \
    -C ~/ arbitrage_bot/ .bot_env

echo "Backup created: $BACKUP_DIR/arbitrage_bot_backup_$TIMESTAMP.tar.gz"

# Keep only last 7 backups
ls -t "$BACKUP_DIR"/*.tar.gz | tail -n +8 | xargs rm -f 2>/dev/null || true
EOF

# Status script
cat > "$HOME/bot_status.sh" << 'EOF'
#!/bin/bash
echo "=== Arbitrage Bot Status ==="
echo "Service Status:"
sudo systemctl status arbitrage-bot.service --no-pager -l

echo -e "\n=== Recent Logs ==="
sudo journalctl -u arbitrage-bot.service --since "1 hour ago" --no-pager | tail -20

echo -e "\n=== System Resources ==="
echo "Memory Usage:"
free -h
echo "Disk Usage:"
df -h /home
echo "CPU Load:"
uptime
EOF

# Make scripts executable
chmod +x "$HOME/update_bot.sh" "$HOME/backup_bot.sh" "$HOME/bot_status.sh"

# Set secure permissions
print_status "Setting secure file permissions..."
chmod 750 "$PROJECT_DIR"
chmod 600 "$ENV_FILE"

# Enable systemd service
print_status "Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable arbitrage-bot.service

print_status "Deployment completed successfully! 🎉"
echo
echo "📋 Next Steps:"
echo "1. Edit your API keys in: $ENV_FILE"
echo "2. Update bot configuration in: $PROJECT_DIR/config.py"
echo "3. Copy your bot files to: $PROJECT_DIR/"
echo "4. Test the bot: cd $PROJECT_DIR && source bot_env/bin/activate && python3 arbitrage_bot.py"
echo "5. Start the service: sudo systemctl start arbitrage-bot.service"
echo
echo "📊 Management Commands:"
echo "• Check status: ~/bot_status.sh"
echo "• Update bot: ~/update_bot.sh"
echo "• Backup bot: ~/backup_bot.sh"
echo "• View logs: sudo journalctl -u arbitrage-bot.service -f"
echo
echo "⚠️  IMPORTANT:"
echo "• Start with Binance TESTNET first!"
echo "• Use small trade amounts initially"
echo "• Monitor logs closely for the first 24 hours"
echo
print_status "Happy trading! 🚀"