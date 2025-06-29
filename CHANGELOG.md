# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Planned: Docker containerization
- Planned: Multi-exchange support
- Planned: Advanced portfolio management

## [1.0.0] - 2025-01-29

### 🚨 Critical Release - Security and Trading Fixes

This is a major release that resolves critical vulnerabilities that could cause trading losses and security breaches. **Immediate upgrade recommended for all users.**

### 🔴 CRITICAL FIXES
- **Fixed arbitrage profit calculation**: Corrected formula that was causing guaranteed trading losses
- **Fixed trade amount logic**: Now properly calculates trade sizes based on actual account balance
- **Added trade execution semaphore**: Prevents dangerous race conditions that could cause over-leveraging
- **Enhanced security**: Comprehensive message sanitization prevents API key leakage

### ✨ Added
- **Telegram Notifications**: Complete integration with rate limiting and security features
  - Real-time trade execution alerts with P&L details
  - Arbitrage opportunity notifications
  - Bot status updates (start/stop/reconnect/errors)
  - Daily trading summaries
  - Configurable notification rates and types
- **Advanced WebSocket Handling**: Exponential backoff, max reconnection attempts, health monitoring
- **Professional Repository Structure**: Proper Python package structure with semantic versioning
- **Comprehensive Documentation**: Detailed guides, API docs, and deployment instructions
- **GitHub Workflows**: CI/CD with testing, security checks, and automated releases
- **Version Management**: Semantic versioning with changelog and migration guides

### 🛡️ Security
- **Message Sanitization**: All Telegram messages sanitized to prevent sensitive data leakage
- **Enhanced Regex Patterns**: Detects and hides API keys, secrets, tokens in all formats
- **Input Validation**: Improved validation for configuration parameters
- **Error Handling**: Secure error messages that don't expose internal state

### 🔧 Changed
- **Project Structure**: Reorganized into professional Python package layout
- **Import Paths**: Updated to use proper package structure (`arbitrage_bot.module`)
- **Configuration**: Environment variable handling with better validation
- **Logging**: Improved logging with rotation and security considerations

### 🐛 Fixed
- **WebSocket Reconnection**: Fixed infinite reconnection loops with exponential backoff
- **Balance Checking**: Proper validation before trade execution
- **Rate Limiting**: Conservative API usage to prevent bans
- **Memory Management**: Fixed potential memory leaks in rate limiter

### ⚠️ Breaking Changes
- **Repository Structure**: Files moved to `src/arbitrage_bot/` directory
- **Import Paths**: Update imports from `import arbitrage_bot` to `from arbitrage_bot import arbitrage_bot`
- **Profit Calculation**: Formula corrected - may affect opportunity detection thresholds
- **Configuration**: Some environment variables renamed for consistency

### 📦 Dependencies
- **Added**: `python-telegram-bot==20.7` for notifications
- **Updated**: `python-binance==1.0.16` for compatibility
- **Added**: Development dependencies for testing and linting

### 🎯 Migration Guide
See [docs/MIGRATION.md](docs/MIGRATION.md) for detailed upgrade instructions.

### 📊 Risk Assessment
- **Before v1.0.0**: Multiple critical vulnerabilities, guaranteed trading losses
- **After v1.0.0**: All critical issues resolved, safe for production use with proper testing

---

## [0.3.0] - 2025-01-28 (DEPRECATED - CRITICAL VULNERABILITIES)

### ⚠️ WARNING: This version contains critical vulnerabilities
- **DO NOT USE**: Contains arbitrage calculation errors causing guaranteed losses
- **SECURITY RISK**: Potential API key exposure through error messages
- **RACE CONDITIONS**: Multiple trades can execute simultaneously

### Added
- Initial triangular arbitrage implementation
- WebSocket price monitoring
- Basic Binance API integration
- Rate limiting framework
- Risk management features

### Known Critical Issues (Fixed in v1.0.0)
- ❌ Incorrect profit percentage calculation
- ❌ Broken trade amount formula
- ❌ Race conditions in trade execution
- ❌ API key exposure in error messages
- ❌ Infinite WebSocket reconnection loops

---

## Release Notes

### v1.0.0 Highlights

This release represents a complete security and reliability overhaul of the triangular arbitrage bot. Key improvements include:

**🔒 Security First**: All communication is now sanitized to prevent sensitive data leakage, with comprehensive regex patterns covering various authentication formats.

**💰 Trading Safety**: Fixed critical calculation errors that would have caused guaranteed losses, implemented proper balance checking, and added trade execution synchronization.

**📱 Professional Monitoring**: Complete Telegram integration with rate-limited notifications for all bot activities, providing real-time visibility into trading operations.

**🏗️ Production Ready**: Professional code organization, comprehensive testing, CI/CD workflows, and detailed documentation make this suitable for production deployment.

**⚡ Reliability**: Robust WebSocket handling with exponential backoff, proper error recovery, and comprehensive monitoring ensure stable operation.

### Upgrade Priority

- **v0.3.x users**: **CRITICAL** - Immediate upgrade required due to trading loss vulnerabilities
- **New users**: Start with v1.0.0 for safe and reliable operation

### Support

For upgrade assistance or questions:
- 📖 [Migration Guide](docs/MIGRATION.md)
- 🐛 [GitHub Issues](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/issues)
- 💬 [Discussions](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/discussions)