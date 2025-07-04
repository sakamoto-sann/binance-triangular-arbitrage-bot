"""
Version information for Triangular Arbitrage Bot
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# Version history
VERSION_HISTORY = {
    "1.0.0": {
        "date": "2025-01-29",
        "changes": [
            "🚨 CRITICAL FIXES: Resolved trading loss vulnerabilities",
            "✅ Fixed arbitrage profit calculation (was causing guaranteed losses)",
            "✅ Fixed trade amount logic with actual balance checking", 
            "✅ Added trade execution semaphore (prevents race conditions)",
            "🛡️ Enhanced security with comprehensive message sanitization",
            "⚡ Improved WebSocket handling with exponential backoff",
            "📱 Complete Telegram notifications integration",
            "🔧 Professional repository structure and documentation"
        ],
        "breaking_changes": [
            "Repository structure reorganized - update import paths",
            "Profit calculation formula corrected - affects opportunity detection"
        ],
        "migration_guide": "See docs/MIGRATION.md for upgrade instructions"
    }
}

def get_version():
    """Get the current version string"""
    return __version__

def get_version_info():
    """Get the current version as a tuple"""
    return __version_info__

def print_version():
    """Print version information"""
    print(f"Triangular Arbitrage Bot v{__version__}")
    print("A production-ready triangular arbitrage bot for Binance")
    print("Repository: https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot")