"""
Delta-Neutral Market Making - Futures Hedge Ladder
Sparse ladder for cost-effective delta management and hedging.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from .data_structures import (
    GridLevel, ExecutionResult, GridType, PositionSide, 
    OrderStatus, GridConfiguration, RiskLimits, DeltaNeutralPosition
)

logger = logging.getLogger(__name__)

class FuturesHedgeLadder:
    """
    Sparse futures ladder optimized for cost-effective delta management.
    
    This component implements a wide-spaced order ladder on futures markets
    designed specifically for maintaining delta neutrality rather than profit generation.
    The sparse structure minimizes trading fees while effectively managing delta drift.
    
    Key Features:
    - Sparse ladder levels (8-15) for cost efficiency
    - Wide spacing (1.5-3.0% ATR) to minimize execution frequency
    - Larger position sizes for efficient delta correction
    - Perfect hedge ratio (0.95-1.05) for precise neutrality
    - Coordinated with spot grid for overall position management
    """
    
    def __init__(self, 
                 futures_client,
                 grid_config: GridConfiguration,
                 risk_limits: RiskLimits):
        """
        Initialize the futures hedge ladder.
        
        Args:
            futures_client: Futures exchange trading client
            grid_config: Grid configuration parameters
            risk_limits: Risk management limits
        """
        self.futures = futures_client
        self.config = grid_config
        self.risk_limits = risk_limits
        
        # Hedge ladder state
        self.hedge_levels: Dict[str, GridLevel] = {}
        self.active_hedge_orders: Dict[str, str] = {}
        self.pending_hedges: List[Dict] = []
        
        # Delta management
        self.current_delta = 0.0
        self.target_delta = 0.0
        self.delta_tolerance = 0.04  # 4% tolerance
        self.last_hedge_time = datetime.now()
        
        # Position tracking
        self.total_futures_position = 0.0
        self.futures_pnl = 0.0
        self.hedge_executions_today = 0
        
        # Market data
        self.futures_price = 0.0
        self.spot_price = 0.0
        self.basis_spread = 0.0
        
        # Hedge parameters (dynamic)
        self.hedge_ratio = self.config.futures_hedge_ratio
        self.ladder_spacing = self.config.futures_grid_spacing_pct
        self.ladder_levels = self.config.futures_grid_levels
        
        logger.info("FuturesHedgeLadder initialized for delta management")
    
    async def initialize_hedge_ladder(self, 
                                    spot_position: float,
                                    futures_price: float,
                                    spot_price: float) -> bool:
        """
        Initialize the futures hedge ladder based on current position.
        
        Args:
            spot_position: Current spot BTC position
            futures_price: Current futures price
            spot_price: Current spot price
            
        Returns:
            bool: True if ladder initialized successfully
        """
        try:
            logger.info(f"🏗️ Initializing futures hedge ladder")
            logger.info(f"Spot position: {spot_position:.4f} BTC")
            
            self.futures_price = futures_price
            self.spot_price = spot_price
            self.basis_spread = spot_price - futures_price
            
            # Calculate required hedge position
            required_hedge = -spot_position * self.hedge_ratio
            
            # Generate hedge ladder around required position
            hedge_levels = self._generate_hedge_levels(required_hedge, futures_price)
            
            # Place hedge orders
            success_count = 0
            for level in hedge_levels:
                if await self._place_hedge_order(level):
                    success_count += 1
                    self.hedge_levels[level.order_id] = level
                else:
                    logger.warning(f"Failed to place hedge order at {level.price}")
            
            logger.info(f"✅ Hedge ladder initialized: {success_count}/{len(hedge_levels)} orders placed")
            return success_count > len(hedge_levels) * 0.7  # 70% success threshold
            
        except Exception as e:
            logger.error(f"Error initializing hedge ladder: {e}")
            return False
    
    async def update_delta_hedge(self, 
                               current_delta: float,
                               spot_position: float,
                               futures_price: float) -> bool:
        """
        Update hedge ladder based on current delta exposure.
        
        Args:
            current_delta: Current net delta exposure
            spot_position: Current spot position
            futures_price: Current futures price
            
        Returns:
            bool: True if hedge updated successfully
        """
        try:
            self.current_delta = current_delta
            self.futures_price = futures_price
            
            # Check if immediate hedge is needed
            if abs(current_delta) > self.delta_tolerance:
                logger.info(f"⚡ Immediate delta hedge needed: {current_delta:.4f}")
                await self._execute_immediate_hedge(current_delta)
            
            # Process any filled hedge orders
            await self._process_filled_hedge_orders()
            
            # Update ladder positioning if needed
            if self._needs_ladder_repositioning(spot_position):
                await self._reposition_hedge_ladder(spot_position)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating delta hedge: {e}")
            return False
    
    def _generate_hedge_levels(self, required_hedge: float, center_price: float) -> List[GridLevel]:
        """
        Generate hedge ladder levels around the required hedge position.
        
        Args:
            required_hedge: Required hedge position size
            center_price: Price to center the ladder around
            
        Returns:
            List[GridLevel]: Generated hedge levels
        """
        levels = []
        
        # Calculate position size per level
        position_per_level = abs(required_hedge) / self.ladder_levels if required_hedge != 0 else 0.1
        
        # Generate levels above and below center price
        half_levels = self.ladder_levels // 2
        
        for i in range(-half_levels, half_levels + 1):
            if i == 0:
                continue  # Skip center level
            
            # Calculate price with wide spacing
            price_multiplier = 1 + (i * self.ladder_spacing)
            level_price = center_price * price_multiplier
            
            # Determine side based on required hedge direction and level position
            if required_hedge < 0:  # Need short hedge
                if i > 0:  # Upper levels - sell orders
                    side = PositionSide.SHORT
                    quantity = position_per_level
                else:  # Lower levels - buy orders (to cover shorts)
                    side = PositionSide.LONG
                    quantity = position_per_level
            else:  # Need long hedge
                if i > 0:  # Upper levels - sell orders (to reduce longs)
                    side = PositionSide.SHORT
                    quantity = position_per_level
                else:  # Lower levels - buy orders
                    side = PositionSide.LONG
                    quantity = position_per_level
            
            level = GridLevel(
                price=level_price,
                quantity=quantity,
                side=side,
                grid_type=GridType.FUTURES_HEDGE
            )
            levels.append(level)
        
        return levels
    
    async def _execute_immediate_hedge(self, delta_exposure: float) -> bool:
        """
        Execute immediate delta hedge using market orders.
        
        Args:
            delta_exposure: Current delta exposure to hedge
            
        Returns:
            bool: True if hedge executed successfully
        """
        try:
            # Calculate hedge size (opposite direction to delta)
            hedge_size = -delta_exposure * self.hedge_ratio
            
            if abs(hedge_size) < 0.001:  # Too small to hedge
                return True
            
            # Determine order side
            side = "SELL" if hedge_size < 0 else "BUY"
            quantity = abs(hedge_size)
            
            logger.info(f"⚡ Executing immediate hedge: {side} {quantity:.4f} BTC")
            
            # Place market order for immediate execution
            result = await self.futures.place_market_order(
                symbol="BTCUSDT",
                side=side,
                quantity=quantity
            )
            
            if result.get("orderId"):
                # Update position tracking
                if side == "SELL":
                    self.total_futures_position -= quantity
                else:
                    self.total_futures_position += quantity
                
                self.hedge_executions_today += 1
                self.last_hedge_time = datetime.now()
                
                logger.info(f"✅ Immediate hedge executed: {side} {quantity:.4f} BTC")
                return True
            else:
                logger.error(f"Failed to execute immediate hedge: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing immediate hedge: {e}")
            return False
    
    async def _place_hedge_order(self, level: GridLevel) -> bool:
        """
        Place a hedge order on the futures exchange.
        
        Args:
            level: Hedge level to place order for
            
        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side
            side = "SELL" if level.side == PositionSide.SHORT else "BUY"
            
            # Place limit order with post-only to ensure maker fees
            result = await self.futures.place_limit_order(
                symbol="BTCUSDT",
                side=side,
                quantity=level.quantity,
                price=level.price,
                timeInForce="GTX"  # Good Till Crossing (post-only)
            )
            
            if result.get("orderId"):
                level.order_id = result["orderId"]
                level.status = OrderStatus.NEW
                self.active_hedge_orders[level.order_id] = level.order_id
                
                logger.debug(f"Hedge order placed: {side} {level.quantity:.4f} BTC @ ${level.price:,.2f}")
                return True
            else:
                logger.warning(f"Failed to place hedge order: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing hedge order: {e}")
            return False
    
    async def _process_filled_hedge_orders(self):
        """Process any filled hedge orders and update position tracking."""
        try:
            filled_orders = []
            
            # Check status of all active hedge orders
            for order_id in list(self.active_hedge_orders.keys()):
                level = self.hedge_levels.get(order_id)
                if not level:
                    continue
                
                # Query order status
                order_status = await self.futures.get_order_status(order_id)
                
                if order_status.get("status") == "FILLED":
                    filled_orders.append((order_id, level, order_status))
                    self.active_hedge_orders.pop(order_id)
            
            # Process filled orders
            for order_id, level, order_status in filled_orders:
                await self._handle_filled_hedge_order(level, order_status)
        
        except Exception as e:
            logger.error(f"Error processing filled hedge orders: {e}")
    
    async def _handle_filled_hedge_order(self, level: GridLevel, order_status: dict):
        """
        Handle a filled hedge order.
        
        Args:
            level: The filled hedge level
            order_status: Order status from exchange
        """
        try:
            fill_price = float(order_status.get("price", level.price))
            fill_quantity = float(order_status.get("executedQty", level.quantity))
            
            # Update level status
            level.status = OrderStatus.FILLED
            level.filled_at = datetime.now()
            
            # Update position tracking
            if level.side == PositionSide.SHORT:
                self.total_futures_position -= fill_quantity
            else:
                self.total_futures_position += fill_quantity
            
            # Calculate basis PnL
            basis_pnl = self._calculate_basis_pnl(level, fill_price, fill_quantity)
            self.futures_pnl += basis_pnl
            
            logger.info(f"🔄 Hedge order filled: {level.side.value} {fill_quantity:.4f} BTC @ ${fill_price:,.2f}")
            
            # Place replacement order if needed
            await self._place_replacement_hedge_order(level, fill_price)
            
            # Update hedge execution count
            self.hedge_executions_today += 1
            
        except Exception as e:
            logger.error(f"Error handling filled hedge order: {e}")
    
    def _calculate_basis_pnl(self, level: GridLevel, fill_price: float, fill_quantity: float) -> float:
        """
        Calculate P&L from basis trading on hedge execution.
        
        Args:
            level: The filled hedge level
            fill_price: Actual fill price
            fill_quantity: Actual fill quantity
            
        Returns:
            float: Basis P&L in USD
        """
        # Simple basis P&L calculation
        # In practice, this would track the basis at entry vs exit
        basis_spread = self.spot_price - fill_price
        
        if level.side == PositionSide.SHORT:
            # Short futures, long spot - profit when basis widens
            pnl = basis_spread * fill_quantity * 0.1  # Small basis profit
        else:
            # Long futures, short spot - profit when basis narrows
            pnl = -basis_spread * fill_quantity * 0.1
        
        return pnl
    
    async def _place_replacement_hedge_order(self, filled_level: GridLevel, fill_price: float):
        """
        Place replacement hedge order to maintain ladder structure.
        
        Args:
            filled_level: The level that was just filled
            fill_price: Price at which the order was filled
        """
        try:
            # Calculate replacement price (further out)
            if filled_level.side == PositionSide.SHORT:
                # Was a sell, place new sell higher
                replacement_price = fill_price * (1 + self.ladder_spacing * 2)
                replacement_side = PositionSide.SHORT
            else:
                # Was a buy, place new buy lower
                replacement_price = fill_price * (1 - self.ladder_spacing * 2)
                replacement_side = PositionSide.LONG
            
            # Create replacement level
            replacement_level = GridLevel(
                price=replacement_price,
                quantity=filled_level.quantity,
                side=replacement_side,
                grid_type=GridType.FUTURES_HEDGE
            )
            
            # Place the replacement order
            if await self._place_hedge_order(replacement_level):
                self.hedge_levels[replacement_level.order_id] = replacement_level
                logger.debug(f"Replacement hedge order placed: {replacement_side.value} @ ${replacement_price:,.2f}")
        
        except Exception as e:
            logger.error(f"Error placing replacement hedge order: {e}")
    
    def _needs_ladder_repositioning(self, spot_position: float) -> bool:
        """
        Check if hedge ladder needs repositioning based on spot position changes.
        
        Args:
            spot_position: Current spot position
            
        Returns:
            bool: True if repositioning needed
        """
        # Calculate ideal hedge position
        ideal_hedge = -spot_position * self.hedge_ratio
        
        # Check if current futures position deviates significantly
        position_deviation = abs(self.total_futures_position - ideal_hedge)
        
        return position_deviation > 0.1  # 0.1 BTC deviation threshold
    
    async def _reposition_hedge_ladder(self, spot_position: float):
        """
        Reposition the entire hedge ladder based on new spot position.
        
        Args:
            spot_position: Current spot position
        """
        try:
            logger.info("🔄 Repositioning hedge ladder")
            
            # Cancel existing hedge orders
            await self._cancel_all_hedge_orders()
            
            # Clear ladder state
            self.hedge_levels.clear()
            self.active_hedge_orders.clear()
            
            # Reinitialize ladder with new positioning
            await self.initialize_hedge_ladder(spot_position, self.futures_price, self.spot_price)
            
        except Exception as e:
            logger.error(f"Error repositioning hedge ladder: {e}")
    
    async def _cancel_all_hedge_orders(self):
        """Cancel all active hedge orders."""
        try:
            for order_id in list(self.active_hedge_orders.keys()):
                await self.futures.cancel_order(order_id)
                self.active_hedge_orders.pop(order_id, None)
                
        except Exception as e:
            logger.error(f"Error canceling hedge orders: {e}")
    
    async def close_all_positions(self) -> bool:
        """
        Close all futures positions in emergency situations.
        
        Returns:
            bool: True if positions closed successfully
        """
        try:
            if abs(self.total_futures_position) < 0.001:
                return True  # No position to close
            
            # Determine closing order side
            side = "BUY" if self.total_futures_position < 0 else "SELL"
            quantity = abs(self.total_futures_position)
            
            logger.warning(f"🚨 Emergency closing futures position: {side} {quantity:.4f} BTC")
            
            # Place market order to close
            result = await self.futures.place_market_order(
                symbol="BTCUSDT",
                side=side,
                quantity=quantity
            )
            
            if result.get("orderId"):
                self.total_futures_position = 0.0
                logger.info("✅ Futures position closed successfully")
                return True
            else:
                logger.error(f"Failed to close futures position: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing futures positions: {e}")
            return False
    
    def get_hedge_metrics(self) -> Dict:
        """Get current hedge ladder performance metrics."""
        return {
            "total_futures_position": self.total_futures_position,
            "futures_pnl": self.futures_pnl,
            "current_delta": self.current_delta,
            "hedge_ratio": self.hedge_ratio,
            "active_orders": len(self.active_hedge_orders),
            "hedge_executions_today": self.hedge_executions_today,
            "ladder_spacing": self.ladder_spacing,
            "ladder_levels": self.ladder_levels,
            "basis_spread": self.basis_spread,
            "last_hedge_time": self.last_hedge_time.isoformat()
        }