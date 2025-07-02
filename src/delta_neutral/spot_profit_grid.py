"""
Delta-Neutral Market Making - Spot Profit Grid
Dense grid for volatility capture and profit generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from .data_structures import (
    GridLevel, ExecutionResult, GridType, PositionSide, 
    OrderStatus, GridConfiguration, RiskLimits
)

logger = logging.getLogger(__name__)

class SpotProfitGrid:
    """
    Dense spot grid optimized for volatility capture and profit generation.
    
    This component implements a high-frequency grid trading strategy on spot markets
    designed to capture profits from market volatility while the futures ladder
    maintains delta neutrality.
    
    Key Features:
    - Dense grid levels (30-50) for maximum rebalancing opportunities
    - Tight spacing (0.3-0.8% ATR) to capture small price movements
    - Small position sizes (1-3%) for high frequency trading
    - Dynamic spacing based on volatility and market conditions
    - Integrated with delta-neutral position management
    """
    
    def __init__(self, 
                 exchange_client,
                 grid_config: GridConfiguration,
                 risk_limits: RiskLimits):
        """
        Initialize the spot profit grid.
        
        Args:
            exchange_client: Spot exchange trading client
            grid_config: Grid configuration parameters
            risk_limits: Risk management limits
        """
        self.exchange = exchange_client
        self.config = grid_config
        self.risk_limits = risk_limits
        
        # Grid state management
        self.grid_levels: Dict[str, GridLevel] = {}
        self.active_orders: Dict[str, str] = {}  # level_id -> order_id mapping
        self.filled_levels: List[GridLevel] = []
        
        # Performance tracking
        self.total_grid_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.grid_fills_today = 0
        
        # Market data
        self.current_price = 0.0
        self.atr_value = 0.0
        self.volatility = 0.0
        
        # Grid parameters (dynamic)
        self.current_spacing = self.config.spot_grid_spacing_pct
        self.current_levels = self.config.spot_grid_levels
        self.position_size = self.config.spot_position_size_pct
        
        logger.info("SpotProfitGrid initialized for volatility capture")
    
    async def initialize_grid(self, current_price: float, atr: float) -> bool:
        """
        Initialize the spot profit grid around current market price.
        
        Args:
            current_price: Current BTC spot price
            atr: Average True Range for dynamic spacing
            
        Returns:
            bool: True if grid initialized successfully
        """
        try:
            logger.info(f"🏗️ Initializing spot profit grid at ${current_price:,.2f}")
            
            self.current_price = current_price
            self.atr_value = atr
            
            # Calculate dynamic grid parameters
            self._update_dynamic_parameters()
            
            # Generate grid levels
            grid_levels = self._generate_grid_levels(current_price)
            
            # Place initial orders
            success_count = 0
            for level in grid_levels:
                if await self._place_grid_order(level):
                    success_count += 1
                    self.grid_levels[level.order_id] = level
                else:
                    logger.warning(f"Failed to place grid order at {level.price}")
            
            logger.info(f"✅ Grid initialized: {success_count}/{len(grid_levels)} orders placed")
            return success_count > len(grid_levels) * 0.8  # 80% success threshold
            
        except Exception as e:
            logger.error(f"Error initializing spot grid: {e}")
            return False
    
    async def update_grid(self, current_price: float, atr: float) -> bool:
        """
        Update grid based on current market conditions.
        
        Args:
            current_price: Current BTC spot price
            atr: Current Average True Range
            
        Returns:
            bool: True if update successful
        """
        try:
            self.current_price = current_price
            self.atr_value = atr
            
            # Update dynamic parameters
            self._update_dynamic_parameters()
            
            # Check for filled orders and handle them
            await self._process_filled_orders()
            
            # Rebalance grid if needed
            if self._needs_grid_rebalancing():
                await self._rebalance_grid()
            
            # Add new levels if gaps exist
            await self._fill_grid_gaps()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating spot grid: {e}")
            return False
    
    def _generate_grid_levels(self, center_price: float) -> List[GridLevel]:
        """
        Generate grid levels around the center price.
        
        Args:
            center_price: Price to center the grid around
            
        Returns:
            List[GridLevel]: Generated grid levels
        """
        levels = []
        
        # Calculate spacing based on ATR
        spacing = self._calculate_dynamic_spacing()
        
        # Generate buy levels (below current price)
        buy_levels = self.current_levels // 2
        for i in range(1, buy_levels + 1):
            price = center_price * (1 - spacing * i)
            quantity = self._calculate_position_size(price)
            
            level = GridLevel(
                price=price,
                quantity=quantity,
                side=PositionSide.LONG,
                grid_type=GridType.SPOT_PROFIT
            )
            levels.append(level)
        
        # Generate sell levels (above current price)
        sell_levels = self.current_levels - buy_levels
        for i in range(1, sell_levels + 1):
            price = center_price * (1 + spacing * i)
            quantity = self._calculate_position_size(price)
            
            level = GridLevel(
                price=price,
                quantity=quantity,
                side=PositionSide.SHORT,
                grid_type=GridType.SPOT_PROFIT
            )
            levels.append(level)
        
        return levels
    
    def _calculate_dynamic_spacing(self) -> float:
        """
        Calculate dynamic grid spacing based on market conditions.
        
        Returns:
            float: Grid spacing as percentage
        """
        # Base spacing from configuration
        base_spacing = self.config.spot_grid_spacing_pct
        
        # ATR-based adjustment
        if self.atr_value > 0:
            atr_multiplier = np.clip(
                self.atr_value / self.current_price,
                self.config.atr_multiplier_min,
                self.config.atr_multiplier_max
            )
            dynamic_spacing = base_spacing * atr_multiplier
        else:
            dynamic_spacing = base_spacing
        
        # Volatility scaling
        if self.config.volatility_scaling and self.volatility > 0:
            vol_multiplier = 1.0 + (self.volatility - 0.02) * 2  # Scale around 2% base vol
            vol_multiplier = np.clip(vol_multiplier, 0.5, 2.0)
            dynamic_spacing *= vol_multiplier
        
        # Ensure minimum and maximum bounds
        min_spacing = 0.001  # 0.1% minimum
        max_spacing = 0.02   # 2% maximum
        
        return np.clip(dynamic_spacing, min_spacing, max_spacing)
    
    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size for a grid level.
        
        Args:
            price: Price level for the order
            
        Returns:
            float: Position size in BTC
        """
        # Base position size in USD
        base_usd_size = self.risk_limits.max_position_usd * self.position_size
        
        # Convert to BTC quantity
        btc_quantity = base_usd_size / price
        
        # Apply minimum and maximum bounds
        min_quantity = 0.0001  # Minimum BTC order size
        max_quantity = self.risk_limits.max_spot_position / self.current_levels
        
        return np.clip(btc_quantity, min_quantity, max_quantity)
    
    async def _place_grid_order(self, level: GridLevel) -> bool:
        """
        Place a grid order on the exchange.
        
        Args:
            level: Grid level to place order for
            
        Returns:
            bool: True if order placed successfully
        """
        try:
            # Determine order side
            side = "BUY" if level.side == PositionSide.LONG else "SELL"
            
            # Place limit order
            result = await self.exchange.place_limit_order(
                symbol="BTCUSDT",
                side=side,
                quantity=level.quantity,
                price=level.price
            )
            
            if result.get("orderId"):
                level.order_id = result["orderId"]
                level.status = OrderStatus.NEW
                self.active_orders[level.order_id] = level.order_id
                
                logger.debug(f"Grid order placed: {side} {level.quantity:.4f} BTC @ ${level.price:,.2f}")
                return True
            else:
                logger.warning(f"Failed to place grid order: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing grid order: {e}")
            return False
    
    async def _process_filled_orders(self):
        """Process any filled grid orders and calculate profits."""
        try:
            filled_orders = []
            
            # Check status of all active orders
            for order_id in list(self.active_orders.keys()):
                level = self.grid_levels.get(order_id)
                if not level:
                    continue
                
                # Query order status
                order_status = await self.exchange.get_order_status(order_id)
                
                if order_status.get("status") == "FILLED":
                    filled_orders.append((order_id, level, order_status))
                    self.active_orders.pop(order_id)
            
            # Process filled orders
            for order_id, level, order_status in filled_orders:
                await self._handle_filled_order(level, order_status)
        
        except Exception as e:
            logger.error(f"Error processing filled orders: {e}")
    
    async def _handle_filled_order(self, level: GridLevel, order_status: dict):
        """
        Handle a filled grid order and place the opposite order.
        
        Args:
            level: The filled grid level
            order_status: Order status from exchange
        """
        try:
            fill_price = float(order_status.get("price", level.price))
            fill_quantity = float(order_status.get("executedQty", level.quantity))
            
            # Update level status
            level.status = OrderStatus.FILLED
            level.filled_at = datetime.now()
            self.filled_levels.append(level)
            
            # Calculate profit if this completes a round trip
            profit = self._calculate_round_trip_profit(level, fill_price, fill_quantity)
            if profit > 0:
                self.total_grid_pnl += profit
                self.winning_trades += 1
                logger.info(f"💰 Grid profit: ${profit:.2f} from {level.side.value} order")
            
            # Place opposite order for grid continuation
            await self._place_opposite_order(level, fill_price)
            
            # Update statistics
            self.total_trades += 1
            self.grid_fills_today += 1
            
        except Exception as e:
            logger.error(f"Error handling filled order: {e}")
    
    def _calculate_round_trip_profit(self, level: GridLevel, fill_price: float, fill_quantity: float) -> float:
        """
        Calculate profit from a completed grid round trip.
        
        Args:
            level: The filled grid level
            fill_price: Actual fill price
            fill_quantity: Actual fill quantity
            
        Returns:
            float: Profit in USD
        """
        # Look for matching opposite trade in recent history
        opposite_side = PositionSide.SHORT if level.side == PositionSide.LONG else PositionSide.LONG
        
        for filled_level in reversed(self.filled_levels[-10:]):  # Check last 10 trades
            if (filled_level.side == opposite_side and 
                abs(filled_level.quantity - fill_quantity) < 0.0001):
                
                # Calculate round trip profit
                if level.side == PositionSide.LONG:
                    # Buy low, sell high
                    profit = (filled_level.price - fill_price) * fill_quantity
                else:
                    # Sell high, buy low
                    profit = (fill_price - filled_level.price) * fill_quantity
                
                return max(0, profit)  # Only positive profits
        
        return 0.0
    
    async def _place_opposite_order(self, filled_level: GridLevel, fill_price: float):
        """
        Place opposite order to continue grid trading.
        
        Args:
            filled_level: The level that was just filled
            fill_price: Price at which the order was filled
        """
        try:
            # Calculate opposite side price
            spacing = self._calculate_dynamic_spacing()
            
            if filled_level.side == PositionSide.LONG:
                # Was a buy, place sell above
                opposite_price = fill_price * (1 + spacing)
                opposite_side = PositionSide.SHORT
            else:
                # Was a sell, place buy below
                opposite_price = fill_price * (1 - spacing)
                opposite_side = PositionSide.LONG
            
            # Create opposite level
            opposite_level = GridLevel(
                price=opposite_price,
                quantity=filled_level.quantity,
                side=opposite_side,
                grid_type=GridType.SPOT_PROFIT
            )
            
            # Place the order
            if await self._place_grid_order(opposite_level):
                self.grid_levels[opposite_level.order_id] = opposite_level
                logger.debug(f"Opposite order placed: {opposite_side.value} @ ${opposite_price:,.2f}")
        
        except Exception as e:
            logger.error(f"Error placing opposite order: {e}")
    
    def _update_dynamic_parameters(self):
        """Update dynamic grid parameters based on market conditions."""
        # Update spacing based on volatility
        self.current_spacing = self._calculate_dynamic_spacing()
        
        # Adjust number of levels based on volatility
        if self.volatility > 0.05:  # High volatility
            self.current_levels = max(20, self.config.spot_grid_levels - 10)
        elif self.volatility < 0.02:  # Low volatility
            self.current_levels = min(50, self.config.spot_grid_levels + 10)
        else:
            self.current_levels = self.config.spot_grid_levels
    
    def _needs_grid_rebalancing(self) -> bool:
        """Check if grid needs rebalancing based on price movement."""
        if not self.grid_levels:
            return True
        
        # Check if price has moved significantly from grid center
        grid_prices = [level.price for level in self.grid_levels.values()]
        if not grid_prices:
            return True
        
        grid_center = (max(grid_prices) + min(grid_prices)) / 2
        price_deviation = abs(self.current_price - grid_center) / grid_center
        
        return price_deviation > 0.1  # 10% deviation threshold
    
    async def _rebalance_grid(self):
        """Rebalance the entire grid around current price."""
        try:
            logger.info("🔄 Rebalancing spot profit grid")
            
            # Cancel all existing orders
            await self._cancel_all_grid_orders()
            
            # Clear grid state
            self.grid_levels.clear()
            self.active_orders.clear()
            
            # Reinitialize grid at current price
            await self.initialize_grid(self.current_price, self.atr_value)
            
        except Exception as e:
            logger.error(f"Error rebalancing grid: {e}")
    
    async def _fill_grid_gaps(self):
        """Fill any gaps in the grid due to filled orders."""
        # Implementation for filling grid gaps
        pass
    
    async def _cancel_all_grid_orders(self):
        """Cancel all active grid orders."""
        try:
            for order_id in list(self.active_orders.keys()):
                await self.exchange.cancel_order(order_id)
                self.active_orders.pop(order_id, None)
                
        except Exception as e:
            logger.error(f"Error canceling grid orders: {e}")
    
    def get_grid_metrics(self) -> Dict:
        """Get current grid performance metrics."""
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        return {
            "total_pnl": self.total_grid_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "active_orders": len(self.active_orders),
            "grid_fills_today": self.grid_fills_today,
            "current_spacing": self.current_spacing,
            "current_levels": self.current_levels,
            "position_size": self.position_size
        }