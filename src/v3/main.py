"""
Grid Trading Bot v3.0 - Main Entry Point
Orchestrates all components for complete trading system.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# Core components
from .core.market_analyzer import MarketAnalyzer, MarketAnalysis
from .core.grid_engine import AdaptiveGridEngine, GridLevel
from .core.risk_manager import RiskManager, RiskMetrics
from .core.order_manager import SmartOrderManager, OrderRequest, OrderType
from .core.performance_tracker import PerformanceTracker

# Strategy components
from .strategies.adaptive_grid import AdaptiveGridStrategy
from .strategies.base_strategy import StrategyConfig

# Utility components
from .utils.config_manager import BotConfig, ConfigManager
from .utils.data_manager import DataManager
from .utils.indicators import TechnicalIndicators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/grid_bot_v3.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class GridTradingBot:
    """
    Main Grid Trading Bot v3.0 class that orchestrates all components.
    """
    
    def __init__(self, config_path: str = "src/v3/config.yaml"):
        """
        Initialize the Grid Trading Bot.
        
        Args:
            config_path: Path to configuration file.
        """
        try:
            # Load configuration
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.load_config()
            
            # Initialize core components
            self.data_manager = DataManager(self.config)
            self.market_analyzer = MarketAnalyzer(self.config)
            self.grid_engine = AdaptiveGridEngine(self.config)
            self.risk_manager = RiskManager(self.config)
            self.order_manager = SmartOrderManager(self.config, self.data_manager)
            self.performance_tracker = PerformanceTracker(self.config, self.data_manager)
            
            # Initialize strategy
            strategy_config = self._create_strategy_config()
            self.strategy = AdaptiveGridStrategy(self.config, strategy_config)
            
            # Bot state
            self.is_running = False
            self.is_trading = False
            self.last_price_update = datetime.now()
            self.last_analysis_update = datetime.now()
            
            # Data storage
            self.current_price = 0.0
            self.price_history = pd.DataFrame()
            self.current_positions = []
            self.active_orders = []
            
            # Performance tracking
            self.start_time = datetime.now()
            self.total_cycles = 0
            self.successful_cycles = 0
            
            logger.info("Grid Trading Bot v3.0 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Grid Trading Bot: {e}")
            raise
    
    async def start(self) -> None:
        """Start the trading bot."""
        try:
            logger.info("🚀 Starting Grid Trading Bot v3.0")
            
            # Validate configuration
            if not self._validate_configuration():
                raise ValueError("Invalid configuration")
            
            # Initialize data feeds
            await self._initialize_data_feeds()
            
            # Start main trading loop
            self.is_running = True
            self.is_trading = True
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Main execution loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        try:
            logger.info("🛑 Stopping Grid Trading Bot v3.0")
            
            self.is_running = False
            self.is_trading = False
            
            # Cancel all open orders
            await self._cancel_all_orders()
            
            # Generate final performance report
            final_report = self.performance_tracker.generate_performance_report()
            logger.info("📊 Final Performance Report Generated")
            
            # Save final state
            await self._save_final_state()
            
            logger.info("✅ Grid Trading Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        loop_interval = self.config.trading.loop_interval_seconds
        
        logger.info(f"🔄 Starting main loop (interval: {loop_interval}s)")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Update market data
                await self._update_market_data()
                
                # Perform market analysis
                market_analysis = await self._analyze_market()
                
                # Assess risk
                risk_metrics = await self._assess_risk(market_analysis)
                
                # Update performance tracking
                await self._update_performance(market_analysis, risk_metrics)
                
                # Generate trading signals
                if self.is_trading:
                    signal = await self._generate_signal(market_analysis, risk_metrics)
                    
                    if signal:
                        await self._execute_signal(signal, risk_metrics)
                
                # Update order statuses
                await self._update_orders()
                
                # Manage grid levels
                await self._manage_grid_levels()
                
                # Check for rebalancing
                await self._check_rebalancing(market_analysis)
                
                # Update cycle statistics
                self.total_cycles += 1
                self.successful_cycles += 1
                
                # Calculate cycle time and sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, loop_interval - cycle_time)
                
                if cycle_time > loop_interval:
                    logger.warning(f"⚠️ Cycle took {cycle_time:.2f}s (target: {loop_interval}s)")
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                # Periodic logging
                if self.total_cycles % 60 == 0:  # Every 60 cycles
                    await self._log_status()
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _update_market_data(self) -> None:
        """Update market data from various sources."""
        try:
            # Get latest price data (simulated for now)
            # In production, this would fetch from exchange APIs
            current_time = datetime.now()
            
            # Simulate price movement (replace with real data feed)
            if self.current_price == 0:
                self.current_price = 50000.0  # Initial BTC price
            else:
                # Simple random walk simulation
                import random
                change_pct = random.uniform(-0.005, 0.005)  # ±0.5% max change
                self.current_price *= (1 + change_pct)
            
            # Update price history
            new_row = pd.DataFrame({
                'timestamp': [current_time],
                'open': [self.current_price * 0.999],
                'high': [self.current_price * 1.001],
                'low': [self.current_price * 0.998],
                'close': [self.current_price],
                'volume': [random.uniform(1000, 5000)]
            })
            
            if len(self.price_history) == 0:
                self.price_history = new_row
            else:
                self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)
            
            # Keep only recent data (last 1000 points)
            if len(self.price_history) > 1000:
                self.price_history = self.price_history.tail(1000).reset_index(drop=True)
            
            # Update order manager with current price
            self.order_manager.update_market_price(
                self.config.trading.symbol, self.current_price
            )
            
            # Update performance tracker with benchmark price
            self.performance_tracker.update_benchmark_price(
                "BTC", self.current_price
            )
            
            self.last_price_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _analyze_market(self) -> MarketAnalysis:
        """Perform comprehensive market analysis."""
        try:
            if len(self.price_history) < 50:  # Need minimum data
                logger.warning("Insufficient price history for analysis")
                # Return default analysis
                from .core.market_analyzer import MarketRegime, VolatilityMetrics, TrendMetrics
                return MarketAnalysis(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.5,
                    volatility_metrics=VolatilityMetrics(0.02, 0.5, "stable", "normal", 0.02, 0.02),
                    trend_metrics=TrendMetrics(0, "sideways", 0.5, 0, 0, 0.5),
                    price_level=self.current_price,
                    support_levels=[],
                    resistance_levels=[],
                    breakout_probability=0.3,
                    mean_reversion_probability=0.7,
                    timestamp=datetime.now()
                )
            
            # Perform market analysis
            market_analysis = self.market_analyzer.analyze_market(self.price_history)
            
            # Update strategy with market state
            self.strategy.update_market_state(
                self.price_history, market_analysis, 
                self.risk_manager.assess_portfolio_risk(
                    self._get_portfolio_value(), 
                    self.current_positions, 
                    self.active_orders, 
                    market_analysis
                )
            )
            
            self.last_analysis_update = datetime.now()
            return market_analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            # Return safe default
            from .core.market_analyzer import MarketRegime, VolatilityMetrics, TrendMetrics
            return MarketAnalysis(
                regime=MarketRegime.UNKNOWN,
                confidence=0.1,
                volatility_metrics=VolatilityMetrics(0.02, 0.5, "stable", "normal", 0.02, 0.02),
                trend_metrics=TrendMetrics(0, "sideways", 0.5, 0, 0, 0.5),
                price_level=self.current_price,
                support_levels=[],
                resistance_levels=[],
                breakout_probability=0.3,
                mean_reversion_probability=0.7,
                timestamp=datetime.now()
            )
    
    async def _assess_risk(self, market_analysis: MarketAnalysis) -> RiskMetrics:
        """Assess current risk levels."""
        try:
            portfolio_value = self._get_portfolio_value()
            
            risk_metrics = self.risk_manager.assess_portfolio_risk(
                portfolio_value,
                self.current_positions,
                self.active_orders,
                market_analysis
            )
            
            # Check for circuit breakers
            circuit_breaker_activated = self.risk_manager.check_circuit_breakers(
                market_analysis, risk_metrics
            )
            
            if circuit_breaker_activated:
                logger.warning("🚨 Circuit breaker activated!")
                self.is_trading = False
                await self._handle_circuit_breaker()
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            # Return safe default risk metrics
            from .core.risk_manager import RiskLevel
            from .core.risk_manager import RiskMetrics as RM
            return RM(
                portfolio_value=self._get_portfolio_value(),
                unrealized_pnl=0, realized_pnl=0, total_exposure=0, leverage=0,
                current_drawdown=0, max_drawdown=0, var_1d=0, var_5d=0,
                sharpe_ratio=0, sortino_ratio=0, risk_level=RiskLevel.LOW,
                last_updated=datetime.now()
            )
    
    async def _update_performance(self, market_analysis: MarketAnalysis, 
                                risk_metrics: RiskMetrics) -> None:
        """Update performance tracking."""
        try:
            portfolio_value = self._get_portfolio_value()
            
            self.performance_tracker.update_portfolio_value(
                portfolio_value, self.current_positions, market_analysis
            )
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def _generate_signal(self, market_analysis: MarketAnalysis,
                             risk_metrics: RiskMetrics) -> Optional[Any]:
        """Generate trading signal using the strategy."""
        try:
            signal = self.strategy.generate_signal(
                self.price_history,
                market_analysis,
                risk_metrics,
                self.current_positions
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    async def _execute_signal(self, signal: Any, risk_metrics: RiskMetrics) -> None:
        """Execute a trading signal."""
        try:
            # Calculate position size
            available_capital = self._get_available_capital()
            position_size = self.strategy.calculate_position_size(
                signal, available_capital, risk_metrics
            )
            
            if position_size <= 0:
                logger.debug("Position size too small, skipping signal")
                return
            
            # Check position limits
            is_allowed, reason = self.risk_manager.check_position_limits(
                position_size, signal.symbol, self.current_positions
            )
            
            if not is_allowed:
                logger.warning(f"Position limit check failed: {reason}")
                return
            
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                order_type=OrderType.LIMIT,
                quantity=position_size,
                price=signal.price,
                execution_strategy=self._get_execution_strategy(signal)
            )
            
            # Place order
            success, message, order_id = await self.order_manager.place_order(order_request)
            
            if success:
                logger.info(f"✅ Order placed: {order_id} - {signal.signal_type.value} "
                          f"{position_size:.6f} {signal.symbol} @ ${signal.price:.2f}")
            else:
                logger.error(f"❌ Order failed: {message}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _update_orders(self) -> None:
        """Update status of all active orders."""
        try:
            await self.order_manager.update_order_statuses()
            self.active_orders = self.order_manager.get_active_orders()
            
        except Exception as e:
            logger.error(f"Error updating orders: {e}")
    
    async def _manage_grid_levels(self) -> None:
        """Manage active grid levels."""
        try:
            # Get current grid status from strategy
            grid_status = self.strategy.get_grid_status()
            
            # Update grid levels if needed
            if grid_status.get('active_levels', 0) == 0:
                available_balance = {
                    'USDT': self._get_available_capital(),
                    'BTC': self._get_btc_balance()
                }
                
                new_levels = self.strategy.update_grid_levels(available_balance)
                
                # Place grid orders
                if new_levels:
                    order_mapping = await self.order_manager.place_grid_orders(new_levels)
                    logger.info(f"📊 Grid orders placed: {len(order_mapping)} levels")
            
        except Exception as e:
            logger.error(f"Error managing grid levels: {e}")
    
    async def _check_rebalancing(self, market_analysis: MarketAnalysis) -> None:
        """Check if portfolio rebalancing is needed."""
        try:
            performance_metrics = self.performance_tracker.get_performance_summary()
            
            if self.strategy.should_rebalance(market_analysis, performance_metrics):
                logger.info("🔄 Rebalancing portfolio...")
                await self._rebalance_portfolio()
            
        except Exception as e:
            logger.error(f"Error checking rebalancing: {e}")
    
    async def _rebalance_portfolio(self) -> None:
        """Perform portfolio rebalancing."""
        try:
            # Cancel existing grid orders
            for order in self.active_orders:
                await self.order_manager.cancel_order(order.order_id)
            
            # Generate new grid levels
            available_balance = {
                'USDT': self._get_available_capital(),
                'BTC': self._get_btc_balance()
            }
            
            new_levels = self.strategy.update_grid_levels(available_balance)
            
            # Place new grid orders
            if new_levels:
                order_mapping = await self.order_manager.place_grid_orders(new_levels)
                logger.info(f"✅ Portfolio rebalanced: {len(order_mapping)} new grid levels")
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            for order in self.active_orders:
                await self.order_manager.cancel_order(order.order_id)
            
            logger.info("🚫 All orders cancelled")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _handle_circuit_breaker(self) -> None:
        """Handle circuit breaker activation."""
        try:
            logger.warning("🚨 Handling circuit breaker activation")
            
            # Cancel all orders
            await self._cancel_all_orders()
            
            # Close risky positions (if any)
            # This would be implemented based on specific risk criteria
            
            # Wait for cooldown period
            cooldown_time = 300  # 5 minutes
            logger.info(f"⏰ Circuit breaker cooldown: {cooldown_time} seconds")
            await asyncio.sleep(cooldown_time)
            
            # Re-enable trading
            self.is_trading = True
            logger.info("✅ Trading re-enabled after circuit breaker")
            
        except Exception as e:
            logger.error(f"Error handling circuit breaker: {e}")
    
    async def _log_status(self) -> None:
        """Log current status."""
        try:
            portfolio_value = self._get_portfolio_value()
            performance_summary = self.performance_tracker.get_performance_summary()
            
            uptime = datetime.now() - self.start_time
            success_rate = self.successful_cycles / self.total_cycles if self.total_cycles > 0 else 0
            
            logger.info(
                f"📈 Status | Portfolio: ${portfolio_value:,.2f} | "
                f"P&L: ${performance_summary.get('total_return', 0):,.2f} "
                f"({performance_summary.get('total_return_pct', 0):.1%}) | "
                f"Trades: {performance_summary.get('total_trades', 0)} | "
                f"Win Rate: {performance_summary.get('win_rate', 0):.1%} | "
                f"Uptime: {uptime} | Success Rate: {success_rate:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    def _validate_configuration(self) -> bool:
        """Validate bot configuration."""
        try:
            # Check required configuration fields
            required_fields = [
                'trading.symbol', 'trading.initial_capital', 'trading.base_position_size'
            ]
            
            for field in required_fields:
                if not self._get_config_value(field):
                    logger.error(f"Missing required configuration: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    async def _initialize_data_feeds(self) -> None:
        """Initialize data feeds and connections."""
        try:
            # Initialize database connections
            self.data_manager.initialize_database()
            
            # Load historical data if available
            # In production, this would load from database or fetch from APIs
            
            logger.info("📡 Data feeds initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data feeds: {e}")
            raise
    
    async def _save_final_state(self) -> None:
        """Save final state before shutdown."""
        try:
            # Export performance data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"data/performance_export_{timestamp}.json"
            
            self.performance_tracker.export_performance_data(export_path)
            
            # Save configuration
            self.config_manager.save_config(self.config, f"data/final_config_{timestamp}.yaml")
            
            logger.info("💾 Final state saved")
            
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _create_strategy_config(self) -> StrategyConfig:
        """Create strategy configuration."""
        return StrategyConfig(
            name="adaptive_grid_v3",
            version="3.0.0",
            enabled=True,
            parameters={
                'min_signal_confidence': 0.6,
                'rebalance_threshold': 0.3,
                'max_position_ratio': 0.8,
                'volatility_adjustment': True,
                'regime_adaptation': True
            },
            risk_parameters={
                'max_drawdown': 0.15,
                'max_position_size': self.config.risk_management.max_position_size,
                'stop_loss_pct': self.config.risk_management.stop_loss_pct
            },
            performance_targets={
                'target_sharpe': 2.0,
                'target_annual_return': 0.25,
                'max_monthly_drawdown': 0.08
            }
        )
    
    def _get_execution_strategy(self, signal: Any):
        """Get execution strategy based on signal characteristics."""
        from .core.order_manager import ExecutionStrategy
        
        if signal.strength.value == 'very_strong':
            return ExecutionStrategy.AGGRESSIVE
        elif signal.strength.value == 'strong':
            return ExecutionStrategy.IMMEDIATE
        else:
            return ExecutionStrategy.PATIENT
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # Simplified calculation - in production this would be more complex
        base_value = self.config.trading.initial_balance
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.current_positions)
        return base_value + unrealized_pnl
    
    def _get_available_capital(self) -> float:
        """Get available trading capital."""
        # Simplified - in production this would check actual balances
        return self._get_portfolio_value() * 0.8  # Use 80% of portfolio
    
    def _get_btc_balance(self) -> float:
        """Get current BTC balance."""
        # Simplified - would get from exchange in production
        return sum(pos.quantity for pos in self.current_positions if pos.symbol == 'BTC')
    
    def _get_config_value(self, path: str) -> Any:
        """Get configuration value by dot-separated path."""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return None
        
        return value


async def main():
    """Main entry point."""
    try:
        # Create and start the bot
        bot = GridTradingBot()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())