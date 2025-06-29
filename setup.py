"""
Setup script for Triangular Arbitrage Bot
"""

from setuptools import setup, find_packages
import os

# Read version
version_file = os.path.join("src", "arbitrage_bot", "__version__.py")
with open(version_file) as f:
    exec(f.read())

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="triangular-arbitrage-bot",
    version=__version__,
    author="Binance Trading Team",
    author_email="noreply@example.com",
    description="A production-ready triangular arbitrage bot for Binance with real-time WebSocket data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "coverage>=6.0.0",
        ],
        "telegram": [
            "python-telegram-bot==20.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "arbitrage-bot=arbitrage_bot.arbitrage_bot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "arbitrage_bot": ["*.py"],
    },
    keywords="trading, arbitrage, binance, cryptocurrency, bot, websocket, telegram",
    project_urls={
        "Bug Reports": "https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/issues",
        "Source": "https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot",
        "Documentation": "https://github.com/sakamoto-sann/binance-triangular-arbitrage-bot/blob/main/docs/",
    },
)