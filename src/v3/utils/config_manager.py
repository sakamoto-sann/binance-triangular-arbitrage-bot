"""
Grid Trading Bot v3.0 - Configuration Manager
Handles loading, validation, and management of configuration settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    symbol: str = "BTCUSDT"
    initial_balance: float = 10000.0
    base_grid_interval: float = 1000.0
    base_position_size: float = 0.001
    target_btc_ratio: float = 0.3
    commission_rate: float = 0.001
    min_order_size: float = 0.0001
    max_order_size: float = 0.01
    loop_interval_seconds: int = 60

@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    grid_count: int = 10
    volatility_lookback: int = 168
    trend_lookback: int = 720
    rebalance_frequency: int = 4
    regime_detection_frequency: int = 1
    max_portfolio_drawdown: float = 0.05
    circuit_breaker_volatility: float = 0.05
    kelly_multiplier: float = 0.25
    trend_threshold: float = 0.1
    volatility_threshold: float = 0.02
    # NEW OPTIMIZATION PARAMETERS
    min_data_points: int = 100
    market_confidence_threshold: float = 0.8

@dataclass
class RiskManagementConfig:
    """Risk management configuration parameters."""
    max_position_size: float = 0.01
    stop_loss_pct: float = 0.02
    max_daily_trades: int = 100
    max_open_orders: int = 20
    inventory_target_min: float = 0.2
    inventory_target_max: float = 0.8
    risk_free_rate: float = 0.02
    # NEW OPTIMIZATION PARAMETERS
    trailing_stop_pct: float = 0.04
    profit_target_pct: float = 0.08
    position_size_high_confidence: float = 0.08
    position_size_medium_confidence: float = 0.05
    position_size_low_confidence: float = 0.02
    correlation_limit: float = 0.6
    transaction_cost_rate: float = 0.001

@dataclass
class MarketRegimeConfig:
    """Market regime detection configuration."""
    ma_fast: int = 50
    ma_slow: int = 200
    atr_period: int = 14
    volatility_periods: list = field(default_factory=lambda: [24, 168, 720])
    breakout_threshold: float = 2.0
    sideways_threshold: float = 0.015
    # NEW OPTIMIZATION PARAMETERS
    ma_regime_filter: int = 200
    bear_market_position_reduction: float = 0.5
    bull_market_position_multiplier: float = 1.5
    # SUPERTREND ENHANCEMENT PARAMETERS
    supertrend_enabled: bool = True
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adaptive_supertrend_enabled: bool = True
    adaptive_supertrend_base_period: int = 10
    adaptive_supertrend_base_multiplier: float = 2.5
    supertrend_signal_weight: float = 0.4  # Weight of supertrend vs MA signals
    signal_agreement_bonus: float = 0.1   # Confidence bonus when signals agree

@dataclass
class GridEngineConfig:
    """Grid engine configuration."""
    min_spacing_multiplier: float = 0.5
    max_spacing_multiplier: float = 3.0
    bull_bias: float = 0.3
    bear_bias: float = 0.7
    sideways_bias: float = 0.5
    adaptive_spacing: bool = True
    spacing_smoothing: float = 0.1
    # NEW OPTIMIZATION PARAMETERS
    atr_spacing_multiplier_min: float = 1.0
    atr_spacing_multiplier_max: float = 2.0
    grid_levels_min: int = 15
    grid_levels_max: int = 25
    staggered_entry: bool = True
    initial_levels: int = 3
    dynamic_adjustment_factor: float = 0.15

@dataclass
class BotConfig:
    """Complete bot configuration."""
    version: str = "3.0.0"
    name: str = "Adaptive Grid Trading Bot"
    description: str = ""
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    grid_engine: GridEngineConfig = field(default_factory=GridEngineConfig)
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        "fast": "1h", "medium": "4h", "slow": "1d", "analysis": "1h"
    })
    performance: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    database: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.yaml"
        self.config: Optional[BotConfig] = None
        self._raw_config: Dict[str, Any] = {}
        
    def load_config(self) -> BotConfig:
        """
        Load configuration from YAML file with environment variable substitution.
        
        Returns:
            Parsed configuration object.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
            ValueError: If configuration validation fails.
        """
        try:
            # Check if config file exists
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Load YAML file
            with open(self.config_path, 'r') as file:
                self._raw_config = yaml.safe_load(file)
            
            # Substitute environment variables
            self._substitute_env_vars(self._raw_config)
            
            # Parse into structured config
            self.config = self._parse_config(self._raw_config)
            
            # Validate configuration
            self._validate_config(self.config)
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _substitute_env_vars(self, config_dict: Dict[str, Any]) -> None:
        """
        Recursively substitute environment variables in configuration.
        
        Args:
            config_dict: Configuration dictionary to process.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                default_value = None
                
                # Handle default values (${VAR:default})
                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)
                
                # Get environment variable value
                env_value = os.getenv(env_var, default_value)
                if env_value is not None:
                    # Try to convert to appropriate type
                    config_dict[key] = self._convert_type(env_value)
                elif default_value is None:
                    logger.warning(f"Environment variable {env_var} not found and no default provided")
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert string value to appropriate Python type.
        
        Args:
            value: String value to convert.
            
        Returns:
            Converted value.
        """
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> BotConfig:
        """
        Parse raw configuration dictionary into structured config object.
        
        Args:
            config_dict: Raw configuration dictionary.
            
        Returns:
            Parsed configuration object.
        """
        # Extract section configurations
        trading_config = TradingConfig(**config_dict.get('trading', {}))
        strategy_config = StrategyConfig(**config_dict.get('strategy', {}))
        risk_config = RiskManagementConfig(**config_dict.get('risk_management', {}))
        regime_config = MarketRegimeConfig(**config_dict.get('market_regime', {}))
        grid_config = GridEngineConfig(**config_dict.get('grid_engine', {}))
        
        # Create main config object
        return BotConfig(
            version=config_dict.get('version', '3.0.0'),
            name=config_dict.get('name', 'Adaptive Grid Trading Bot'),
            description=config_dict.get('description', ''),
            trading=trading_config,
            strategy=strategy_config,
            risk_management=risk_config,
            market_regime=regime_config,
            grid_engine=grid_config,
            timeframes=config_dict.get('timeframes', {}),
            performance=config_dict.get('performance', {}),
            logging=config_dict.get('logging', {}),
            database=config_dict.get('database', {}),
            api=config_dict.get('api', {}),
            development=config_dict.get('development', {})
        )
    
    def _validate_config(self, config: BotConfig) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration object to validate.
            
        Raises:
            ValueError: If validation fails.
        """
        # Validate trading config
        if config.trading.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if config.trading.base_position_size <= 0:
            raise ValueError("Base position size must be positive")
        
        if not 0 < config.trading.target_btc_ratio < 1:
            raise ValueError("Target BTC ratio must be between 0 and 1")
        
        if config.trading.commission_rate < 0:
            raise ValueError("Commission rate cannot be negative")
        
        # Validate strategy config
        if config.strategy.grid_count <= 0:
            raise ValueError("Grid count must be positive")
        
        if config.strategy.volatility_lookback <= 0:
            raise ValueError("Volatility lookback must be positive")
        
        if not 0 < config.strategy.max_portfolio_drawdown <= 1:
            raise ValueError("Max portfolio drawdown must be between 0 and 1")
        
        # Validate risk management
        if config.risk_management.max_position_size <= 0:
            raise ValueError("Max position size must be positive")
        
        if config.risk_management.stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive")
        
        # Validate market regime config
        if config.market_regime.ma_fast >= config.market_regime.ma_slow:
            raise ValueError("Fast MA period must be less than slow MA period")
        
        if config.market_regime.atr_period <= 0:
            raise ValueError("ATR period must be positive")
        
        # Validate grid engine config
        if config.grid_engine.min_spacing_multiplier <= 0:
            raise ValueError("Min spacing multiplier must be positive")
        
        if config.grid_engine.max_spacing_multiplier <= config.grid_engine.min_spacing_multiplier:
            raise ValueError("Max spacing multiplier must be greater than min")
        
        logger.info("Configuration validation passed")
    
    def get_config(self) -> BotConfig:
        """
        Get loaded configuration.
        
        Returns:
            Configuration object.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        # Apply updates to raw config
        self._deep_update(self._raw_config, updates)
        
        # Re-parse configuration
        self.config = self._parse_config(self._raw_config)
        
        # Re-validate
        self._validate_config(self.config)
        
        logger.info("Configuration updated successfully")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update.
            update_dict: Updates to apply.
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            file_path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(file_path) if file_path else self.config_path
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as file:
                yaml.dump(self._raw_config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get specific configuration section.
        
        Args:
            section: Section name to retrieve.
            
        Returns:
            Configuration section.
        """
        if section not in self._raw_config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        return self._raw_config[section]
    
    def set_development_mode(self, enabled: bool) -> None:
        """
        Enable or disable development mode.
        
        Args:
            enabled: Whether to enable development mode.
        """
        self.update_config({
            'development': {
                'paper_trading': enabled,
                'debug_mode': enabled
            }
        })
        
        logger.info(f"Development mode {'enabled' if enabled else 'disabled'}")