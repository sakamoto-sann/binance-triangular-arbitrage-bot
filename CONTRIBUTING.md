# Contributing to Triangular Arbitrage Bot

First off, thank you for considering contributing to the Triangular Arbitrage Bot! 🎉

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- Use the bug report template
- Include error messages and logs (sanitized of sensitive data)
- Describe the exact steps to reproduce
- Include your environment details (OS, Python version, etc.)
- **NEVER include API keys, secrets, or private information**

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use the feature request template
- Provide a clear description of the problem and solution
- Include examples of the desired behavior
- Consider the security and safety implications

### 🔧 Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the development setup** instructions below
3. **Make your changes** with proper testing
4. **Follow coding standards** outlined below
5. **Update documentation** if needed
6. **Create a pull request** with a clear description

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tools

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/binance-triangular-arbitrage-bot.git
cd binance-triangular-arbitrage-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arbitrage_bot

# Run specific test file
pytest tests/test_arbitrage_bot.py

# Run linting
flake8 src/
mypy src/arbitrage_bot
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters (Black default)
- Use type hints for all functions and methods

### Code Quality

- **Security First**: All code must be security-reviewed
- **No Hardcoded Secrets**: Use environment variables
- **Error Handling**: Comprehensive exception handling
- **Logging**: Appropriate logging levels and sanitization
- **Testing**: Minimum 80% test coverage for new code

### Trading Logic Standards

- **Safety Checks**: Always validate inputs and balances
- **Rate Limiting**: Respect API limits
- **Risk Management**: Implement proper safeguards
- **Documentation**: Clear docstrings for trading functions

### Commit Messages

Follow [Conventional Commits](https://conventionalcommits.org/):

```
feat: add support for new trading pairs
fix: resolve WebSocket reconnection issue
docs: update deployment guide
test: add tests for profit calculation
refactor: improve error handling
security: enhance API key sanitization
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

## Testing Guidelines

### Unit Tests

- Test all public methods
- Mock external dependencies (Binance API, WebSocket)
- Test error conditions and edge cases
- Use meaningful test names

### Integration Tests

- Test complete trading workflows
- Use Binance testnet for API integration tests
- Test WebSocket connection handling
- Test Telegram notifications

### Security Tests

- Test API key sanitization
- Verify no secrets in logs
- Test input validation
- Test rate limiting

## Documentation

### Code Documentation

- Clear docstrings for all functions/classes
- Include parameter types and return types
- Document exceptions that can be raised
- Include usage examples for complex functions

### User Documentation

- Update README.md for user-facing changes
- Update deployment guides for configuration changes
- Include migration guides for breaking changes
- Add examples for new features

## Security Guidelines

### Critical Rules

1. **Never commit secrets**: Use environment variables
2. **Sanitize all outputs**: Especially error messages and logs
3. **Validate all inputs**: From users and external APIs
4. **Test security features**: Include security tests
5. **Review carefully**: All trading logic must be reviewed

### Trading Safety

1. **Testnet first**: All trading features must work on testnet
2. **Small amounts**: Test with minimal amounts initially
3. **Rate limiting**: Respect all API limits
4. **Error recovery**: Handle partial failures gracefully
5. **Balance protection**: Never trade more than intended

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Testnet testing completed
- [ ] Performance testing completed

## Getting Help

### Questions

- 💬 [GitHub Discussions](https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/discussions)
- 📧 Email: noreply@example.com

### Development Help

- Check existing issues and PRs
- Look at the project documentation
- Review similar implementations
- Ask in discussions before starting large changes

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes for significant contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to making cryptocurrency trading safer and more accessible!** 🚀