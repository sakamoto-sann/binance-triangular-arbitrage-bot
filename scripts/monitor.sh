#!/bin/bash

# Triangular Arbitrage Bot - Advanced Monitoring Script
# Usage: ./monitor.sh [option]
# Options: status, logs, performance, trades, health

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BOT_SERVICE="arbitrage-bot.service"
LOG_FILE="$HOME/arbitrage_bot/arbitrage_bot.log"
PROJECT_DIR="$HOME/arbitrage_bot"

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                ${MAGENTA}Triangular Arbitrage Bot Monitor${NC}                ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

print_section() {
    echo -e "${BLUE}▶ $1${NC}"
    echo "─────────────────────────────────────────────────────────────"
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

show_service_status() {
    print_section "Service Status"
    
    if systemctl is-active --quiet "$BOT_SERVICE"; then
        print_status "Service is running"
        STATUS="Active"
    else
        print_error "Service is not running"
        STATUS="Inactive"
    fi
    
    echo "Service: $BOT_SERVICE"
    echo "Status: $STATUS"
    echo "Uptime: $(systemctl show -p ActiveEnterTimestamp "$BOT_SERVICE" --value | xargs -I {} date -d "{}" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "Unknown")"
    echo
    
    # Detailed status
    systemctl status "$BOT_SERVICE" --no-pager -l | head -20
    echo
}

show_logs() {
    print_section "Recent Logs"
    
    local lines=${1:-50}
    print_info "Showing last $lines lines from systemd journal"
    
    if journalctl -u "$BOT_SERVICE" --since "24 hours ago" --no-pager | tail -n "$lines" | grep -q .; then
        journalctl -u "$BOT_SERVICE" --since "24 hours ago" --no-pager | tail -n "$lines"
    else
        print_warning "No recent logs found"
    fi
    
    echo
    
    # Show bot log file if exists
    if [ -f "$LOG_FILE" ]; then
        print_info "Last 10 lines from bot log file:"
        tail -n 10 "$LOG_FILE"
    else
        print_warning "Bot log file not found at $LOG_FILE"
    fi
    echo
}

show_performance() {
    print_section "System Performance"
    
    # Memory usage
    echo "Memory Usage:"
    free -h
    echo
    
    # Disk usage
    echo "Disk Usage:"
    df -h "$HOME" 2>/dev/null || df -h /
    echo
    
    # CPU load
    echo "CPU Load:"
    uptime
    echo
    
    # Network connections
    echo "Network Connections (Bot related):"
    netstat -tulpn 2>/dev/null | grep python || echo "No Python network connections found"
    echo
    
    # Process information
    echo "Bot Process Information:"
    if pgrep -f "arbitrage_bot.py" > /dev/null; then
        ps aux | grep -v grep | grep arbitrage_bot.py || echo "Bot process not found"
    else
        print_warning "Bot process not running"
    fi
    echo
}

show_trades() {
    print_section "Trading Activity"
    
    if [ -f "$LOG_FILE" ]; then
        print_info "Recent arbitrage opportunities (last 24 hours):"
        grep -i "arbitrage opportunity" "$LOG_FILE" | tail -n 10 || echo "No arbitrage opportunities found in logs"
        echo
        
        print_info "Recent trade executions:"
        grep -i "triangular arbitrage completed\|successfully executed" "$LOG_FILE" | tail -n 10 || echo "No completed trades found in logs"
        echo
        
        print_info "Recent errors:"
        grep -i "error\|failed\|exception" "$LOG_FILE" | tail -n 5 || echo "No recent errors found"
        echo
    else
        print_warning "Log file not found - cannot show trading activity"
    fi
}

show_health() {
    print_section "Health Check"
    
    local health_score=100
    local issues=()
    
    # Check service status
    if ! systemctl is-active --quiet "$BOT_SERVICE"; then
        health_score=$((health_score - 50))
        issues+=("Service not running")
    fi
    
    # Check log file
    if [ ! -f "$LOG_FILE" ]; then
        health_score=$((health_score - 10))
        issues+=("Log file missing")
    fi
    
    # Check for recent errors
    if [ -f "$LOG_FILE" ] && grep -q -i "error\|failed\|exception" "$LOG_FILE" 2>/dev/null; then
        local error_count=$(grep -c -i "error\|failed\|exception" "$LOG_FILE" 2>/dev/null | head -1)
        if [ "$error_count" -gt 10 ]; then
            health_score=$((health_score - 20))
            issues+=("High error count: $error_count")
        fi
    fi
    
    # Check memory usage
    local mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
    if (( $(echo "$mem_usage > 90" | bc -l) )); then
        health_score=$((health_score - 15))
        issues+=("High memory usage: ${mem_usage}%")
    fi
    
    # Check disk space
    local disk_usage=$(df "$HOME" 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
    if [ "$disk_usage" -gt 90 ]; then
        health_score=$((health_score - 10))
        issues+=("High disk usage: ${disk_usage}%")
    fi
    
    # Display health score
    if [ $health_score -ge 90 ]; then
        print_status "Health Score: $health_score/100 (Excellent)"
    elif [ $health_score -ge 70 ]; then
        print_warning "Health Score: $health_score/100 (Good)"
    elif [ $health_score -ge 50 ]; then
        print_warning "Health Score: $health_score/100 (Fair)"
    else
        print_error "Health Score: $health_score/100 (Poor)"
    fi
    
    if [ ${#issues[@]} -eq 0 ]; then
        print_status "No issues detected"
    else
        print_warning "Issues detected:"
        for issue in "${issues[@]}"; do
            echo "  • $issue"
        done
    fi
    echo
}

show_api_test() {
    print_section "API Connection Test"
    
    if [ -f "$HOME/.bot_env" ]; then
        source "$HOME/.bot_env"
        
        if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
            print_info "Testing Binance API connection..."
            
            cd "$PROJECT_DIR"
            if [ -d "bot_env" ]; then
                source bot_env/bin/activate
                
                python3 -c "
import os
import sys
try:
    from binance.client import Client
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    client = Client(api_key, api_secret, testnet=testnet)
    info = client.get_account()
    
    print('✓ API connection successful!')
    print(f'Account type: {info.get(\"accountType\", \"Unknown\")}')
    print(f'Testnet mode: {testnet}')
    
    # Show balances
    balances = [b for b in info['balances'] if float(b['free']) > 0]
    if balances:
        print('Non-zero balances:')
        for balance in balances[:10]:  # Show first 10
            print(f'  {balance[\"asset\"]}: {balance[\"free\"]}')
    else:
        print('No balances found')
        
except ImportError:
    print('✗ python-binance not installed')
    sys.exit(1)
except Exception as e:
    print(f'✗ API connection failed: {e}')
    sys.exit(1)
" 2>/dev/null || print_error "API test failed"
            else
                print_warning "Virtual environment not found"
            fi
        else
            print_warning "API keys not configured"
        fi
    else
        print_warning "Environment file not found"
    fi
    echo
}

show_dashboard() {
    clear
    print_header
    show_service_status
    show_health
    show_performance
    show_trades
}

show_help() {
    print_header
    echo "Usage: $0 [option]"
    echo
    echo "Options:"
    echo "  status      - Show service status and basic info"
    echo "  logs        - Show recent logs"
    echo "  performance - Show system performance metrics"
    echo "  trades      - Show recent trading activity"
    echo "  health      - Perform health check"
    echo "  api         - Test API connection"
    echo "  dashboard   - Show comprehensive dashboard"
    echo "  live        - Live monitoring (refreshes every 30 seconds)"
    echo "  help        - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  $0 dashboard"
    echo
}

live_monitor() {
    print_info "Starting live monitoring (press Ctrl+C to stop)"
    echo "Refreshing every 30 seconds..."
    echo
    
    while true; do
        show_dashboard
        echo -e "${YELLOW}Refreshing in 30 seconds... (Ctrl+C to stop)${NC}"
        sleep 30
        clear
    done
}

# Main script logic
case "${1:-dashboard}" in
    "status")
        print_header
        show_service_status
        ;;
    "logs")
        print_header
        show_logs "${2:-50}"
        ;;
    "performance")
        print_header
        show_performance
        ;;
    "trades")
        print_header
        show_trades
        ;;
    "health")
        print_header
        show_health
        ;;
    "api")
        print_header
        show_api_test
        ;;
    "dashboard")
        show_dashboard
        ;;
    "live")
        live_monitor
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac