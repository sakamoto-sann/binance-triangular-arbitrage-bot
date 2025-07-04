"""
Delta-Neutral Market Making Grid Bot - Core Manager
Professional-grade delta-neutral position management and coordination.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .data_structures import (
    DeltaNeutralPosition, GridLevel, ExecutionResult, FundingRateData,
    BasisData, MarketNeutralMetrics, RiskLimits, GridConfiguration,
    SystemState, PositionSide, GridType, OrderStatus
)

logger = logging.getLogger(__name__)

class DeltaNeutralManager:
    """
    Central coordination brain for delta-neutral market making strategy.
    
    This is the core component that orchestrates all other systems:
    - Maintains delta neutrality through real-time monitoring
    - Coordinates spot profit grid and futures hedge ladder
    - Manages funding rate optimization and position flipping
    - Enforces multi-layer risk controls
    - Handles execution risk and emergency procedures
    """
    
    def __init__(self, 
                 risk_limits: RiskLimits,
                 grid_config: GridConfiguration,
                 spot_exchange_client,
                 futures_exchange_client):
        """
        Initialize the delta-neutral manager.
        
        Args:
            risk_limits: Risk management limits and thresholds
            grid_config: Grid trading configuration parameters
            spot_exchange_client: Spot trading exchange client
            futures_exchange_client: Futures trading exchange client
        """
        self.risk_limits = risk_limits
        self.grid_config = grid_config
        self.spot_client = spot_exchange_client
        self.futures_client = futures_exchange_client
        
        # Core state management
        self.position = DeltaNeutralPosition()
        self.metrics = MarketNeutralMetrics()
        self.system_state = SystemState()
        
        # Component coordination
        self.spot_grid = None       # Will be injected
        self.futures_ladder = None  # Will be injected
        self.funding_monitor = None # Will be injected
        self.risk_manager = None    # Will be injected
        
        # Risk and emergency management
        self.last_risk_check = datetime.now()
        self.emergency_hedge_pending = False
        self.consecutive_failures = 0
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.pnl_history: List[float] = []
        self.delta_history: List[float] = []
        
        logger.info("DeltaNeutralManager initialized with professional-grade controls")
    
    async def start_strategy(self) -> bool:
        """
        Start the delta-neutral market making strategy.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            logger.info("🚀 Starting Delta-Neutral Market Making Strategy")
            
            # 1. Validate system readiness
            if not await self._validate_system_readiness():
                logger.error("System readiness validation failed")
                return False
            
            # 2. Initialize grid systems
            await self._initialize_grids()
            
            # 3. Start monitoring loops
            asyncio.create_task(self._delta_monitoring_loop())
            asyncio.create_task(self._funding_optimization_loop())
            asyncio.create_task(self._risk_monitoring_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            # 4. Activate system
            self.system_state.is_active = True
            self.system_state.update_heartbeat()
            
            logger.info("✅ Delta-neutral strategy activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start strategy: {e}")
            await self._emergency_shutdown()
            return False
    
    async def stop_strategy(self) -> bool:
        """Stop the strategy gracefully."""
        try:
            logger.info("🛑 Stopping Delta-Neutral Strategy")
            
            # 1. Stop new orders
            self.system_state.is_active = False
            
            # 2. Cancel all pending orders
            await self._cancel_all_orders()
            
            # 3. Close positions if required
            if self.system_state.is_emergency_mode:
                await self._emergency_position_closure()
            
            logger.info("✅ Strategy stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            return False
    
    async def _validate_system_readiness(self) -> bool:
        """Validate that all systems are ready for trading."""
        try:
            # Check exchange connections
            spot_connected = await self._test_spot_connection()
            futures_connected = await self._test_futures_connection()
            
            if not spot_connected or not futures_connected:
                logger.error("Exchange connection validation failed")
                return False
            
            # Check account balances
            if not await self._validate_account_balances():
                logger.error("Account balance validation failed")
                return False
            
            # Check market data availability
            if not await self._validate_market_data():
                logger.error("Market data validation failed")
                return False
            
            # Update system state
            self.system_state.spot_exchange_connected = spot_connected
            self.system_state.futures_exchange_connected = futures_connected
            
            return True
            
        except Exception as e:
            logger.error(f"System readiness validation error: {e}")
            return False
    
    async def _delta_monitoring_loop(self):
        """
        Core delta monitoring and rebalancing loop.
        Maintains delta neutrality through continuous monitoring.
        """
        while self.system_state.is_active:
            try:
                # Calculate current delta exposure
                current_delta = self.position.calculate_delta()
                self.delta_history.append(current_delta)
                
                # Check if rebalancing is needed
                if self.position.needs_rebalance(self.grid_config.delta_rebalance_threshold):
                    logger.info(f"🔄 Delta rebalancing needed: {current_delta:.4f}")
                    
                    # Execute delta hedge
                    success = await self._execute_delta_hedge(current_delta)
                    
                    if not success:
                        logger.warning("Delta hedge execution failed")
                        self.consecutive_failures += 1
                        
                        if self.consecutive_failures >= self.risk_limits.max_failed_executions:
                            logger.error("Too many consecutive failures - entering emergency mode")
                            await self._activate_emergency_mode()
                    else:
                        self.consecutive_failures = 0
                
                # Update metrics
                self.metrics.delta_exposure = current_delta
                self.metrics.last_updated = datetime.now()
                
                # Short sleep to prevent excessive CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in delta monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _funding_optimization_loop(self):
        """
        Funding rate optimization and position flipping loop.
        Monitors funding rates and optimizes position direction.
        """
        while self.system_state.is_active:
            try:
                # Get current funding rate data
                funding_data = await self._get_funding_rate_data()
                
                if funding_data:
                    # Update position funding rate
                    self.position.current_funding_rate = funding_data.funding_rate
                    
                    # Check if position flip is needed
                    if await self._should_flip_position(funding_data):
                        logger.info("💱 Position flip required for funding optimization")
                        await self._execute_position_flip(funding_data)
                    
                    # Update funding metrics
                    self._update_funding_metrics(funding_data)
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in funding optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _risk_monitoring_loop(self):
        """
        Multi-layer risk monitoring and control loop.
        Enforces all risk limits and triggers emergency procedures.
        """
        while self.system_state.is_active:
            try:
                # Update system heartbeat
                self.system_state.update_heartbeat()
                
                # Check all risk metrics
                risk_violations = await self._check_risk_violations()
                
                if risk_violations:
                    logger.warning(f"⚠️ Risk violations detected: {risk_violations}")
                    await self._handle_risk_violations(risk_violations)
                
                # Check for emergency conditions
                if await self._check_emergency_conditions():
                    logger.critical("🚨 Emergency conditions detected")
                    await self._activate_emergency_mode()
                
                # Update risk level
                self._update_risk_level()
                
                # Risk check every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_delta_hedge(self, current_delta: float) -> bool:
        """
        Execute delta hedge to restore neutrality.
        
        Args:
            current_delta: Current delta exposure
            
        Returns:
            bool: True if hedge executed successfully
        """
        try:
            # Calculate required hedge size
            hedge_size = -current_delta  # Opposite direction to neutralize
            
            # Determine execution method
            if abs(hedge_size) > self.risk_limits.max_position_usd / 50000:  # Significant size
                # Use futures for large hedges (more efficient)
                result = await self._execute_futures_hedge(hedge_size)
            else:
                # Use spot grid rebalancing for small hedges
                result = await self._execute_spot_rebalance(hedge_size)
            
            if result.success:
                # Update position
                if result.partial_fill:
                    logger.warning(f"Partial hedge execution: {result.filled_quantity}")
                
                self.position.last_rebalance = datetime.now()
                
                # Log successful hedge
                logger.info(f"✅ Delta hedge executed: {hedge_size:.4f} BTC")
                return True
            else:
                logger.error(f"Delta hedge failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing delta hedge: {e}")
            return False
    
    async def _should_flip_position(self, funding_data: FundingRateData) -> bool:
        """
        Determine if position should be flipped for funding optimization.
        
        Args:
            funding_data: Current funding rate information
            
        Returns:
            bool: True if position should be flipped
        """
        try:
            # Calculate 3-day moving average funding rate
            avg_funding = funding_data.historical_avg
            
            # Current position configuration
            is_long_spot = self.position.spot_position > 0
            is_short_futures = self.position.futures_position < 0
            is_standard_position = is_long_spot and is_short_futures
            
            # Decision logic based on funding rate trends
            if is_standard_position:
                # Currently in +Spot/-Futures position
                # Flip if funding becomes consistently negative
                return avg_funding < -0.003  # -0.3% threshold
            else:
                # Currently in -Spot/+Futures position  
                # Flip back if funding becomes positive
                return avg_funding > 0.001   # +0.1% threshold
                
        except Exception as e:
            logger.error(f"Error determining position flip: {e}")
            return False
    
    async def _activate_emergency_mode(self):
        """Activate emergency mode with immediate risk reduction."""
        try:
            logger.critical("🚨 ACTIVATING EMERGENCY MODE")
            
            self.system_state.is_emergency_mode = True
            self.system_state.is_active = False
            
            # Cancel all pending orders immediately
            await self._cancel_all_orders()
            
            # Assess position closure need
            current_delta = abs(self.position.calculate_delta())
            current_drawdown = self.metrics.current_drawdown
            
            if (current_delta > self.risk_limits.max_delta_exposure * 2 or 
                current_drawdown > self.risk_limits.circuit_breaker_drawdown):
                
                logger.critical("Emergency position closure required")
                await self._emergency_position_closure()
            
            # Send emergency alerts
            self.system_state.add_alert("EMERGENCY_MODE_ACTIVATED")
            
        except Exception as e:
            logger.critical(f"Error in emergency mode activation: {e}")
    
    async def _emergency_position_closure(self):
        """Emergency closure of all positions."""
        try:
            logger.critical("⚡ Executing emergency position closure")
            
            # Close spot position
            if abs(self.position.spot_position) > 0.001:
                spot_result = await self._close_spot_position()
                logger.info(f"Spot position closure: {spot_result.success}")
            
            # Close futures position
            if abs(self.position.futures_position) > 0.001:
                futures_result = await self._close_futures_position()
                logger.info(f"Futures position closure: {futures_result.success}")
            
            # Reset position tracking
            self.position = DeltaNeutralPosition()
            
        except Exception as e:
            logger.critical(f"Error in emergency position closure: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            "system_active": self.system_state.is_active,
            "emergency_mode": self.system_state.is_emergency_mode,
            "position": {
                "spot_btc": self.position.spot_position,
                "futures_btc": self.position.futures_position,
                "net_delta": self.position.net_delta,
                "funding_rate": self.position.current_funding_rate,
                "basis_spread": self.position.basis_spread
            },
            "metrics": {
                "total_pnl": self.metrics.total_pnl,
                "grid_pnl": self.metrics.grid_pnl,
                "funding_pnl": self.metrics.funding_pnl,
                "current_drawdown": self.metrics.current_drawdown,
                "win_rate": self.metrics.win_rate,
                "sharpe_ratio": self.metrics.sharpe_ratio
            },
            "risk": {
                "delta_exposure": self.metrics.delta_exposure,
                "risk_level": self.system_state.current_risk_level,
                "active_alerts": self.system_state.active_alerts
            },
            "last_update": self.metrics.last_updated.isoformat()
        }
    
    # Additional helper methods would be implemented here:
    # - _test_spot_connection()
    # - _test_futures_connection()
    # - _validate_account_balances()
    # - _validate_market_data()
    # - _initialize_grids()
    # - _get_funding_rate_data()
    # - _execute_futures_hedge()
    # - _execute_spot_rebalance()
    # - _execute_position_flip()
    # - _check_risk_violations()
    # - _check_emergency_conditions()
    # - _handle_risk_violations()
    # - _update_risk_level()
    # - _update_funding_metrics()
    # - _cancel_all_orders()
    # - _close_spot_position()
    # - _close_futures_position()
    # - _performance_tracking_loop()