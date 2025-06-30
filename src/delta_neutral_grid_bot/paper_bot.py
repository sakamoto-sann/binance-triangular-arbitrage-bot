import asyncio
import logging
from datetime import datetime
from src.delta_neutral_grid_bot.config import SYMBOL, GRID_SIZE, GRID_INTERVAL, ORDER_SIZE, FUTURES_SYMBOL
from src.delta_neutral_grid_bot.state_manager import StateManager
from src.paper_trading import PaperTradingClient
from src.volume_tracker import VIPVolumeTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperDeltaNeutralGridBot:
    """
    Delta-neutral grid bot for paper trading (safe testing)
    Uses PaperTradingClient instead of real Binance API
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.client = PaperTradingClient(initial_balance)
        self.state_manager = StateManager()
        self.volume_tracker = VIPVolumeTracker()
        self.shutdown_event = asyncio.Event()
        self.grid_orders = []
        self.total_funding_fees = 0.0
        self.total_profits = 0.0
        
    async def run(self):
        """Run the paper trading bot"""
        logging.info("Starting paper trading delta-neutral grid bot...")
        
        try:
            # Set initial BTC price
            current_price = 45000.0
            self.client.update_price(SYMBOL, current_price)
            
            # Setup initial grid
            await self.setup_grid()
            
            # Main trading loop
            iteration = 0
            while not self.shutdown_event.is_set() and iteration < 1000:  # Limit for testing
                try:
                    # Simulate price movement
                    current_price = await self.simulate_price_movement(current_price, iteration)
                    self.client.update_price(SYMBOL, current_price)
                    
                    # Check grid and process fills
                    await self.check_grid()
                    
                    # Simulate hedge position (simplified for paper trading)
                    await self.hedge_position()
                    
                    # Log status every 50 iterations
                    if iteration % 50 == 0:
                        self.client.log_status()
                        self.volume_tracker.log_vip_status()
                    
                    iteration += 1
                    await asyncio.sleep(0.1)  # Fast simulation
                    
                except KeyboardInterrupt:
                    logging.info("Stopping paper trading bot...")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    
        except Exception as e:
            logging.error(f"Fatal error in paper trading bot: {e}")
        finally:
            await self.cleanup()
    
    async def simulate_price_movement(self, current_price: float, iteration: int) -> float:
        """Simulate realistic price movement"""
        import random
        
        # Create some volatility patterns
        base_volatility = 0.001  # 0.1% base volatility
        
        # Add some trending behavior
        trend_factor = 0
        if iteration % 200 < 100:  # Uptrend for first half
            trend_factor = 0.0005
        else:  # Downtrend for second half
            trend_factor = -0.0005
        
        # Random walk with trend
        change = random.gauss(trend_factor, base_volatility)
        new_price = current_price * (1 + change)
        
        # Ensure price stays within reasonable bounds
        new_price = max(30000, min(60000, new_price))
        
        return new_price
    
    async def setup_grid(self):
        """Setup initial grid of orders"""
        logging.info("Setting up paper trading grid...")
        
        try:
            current_price = self.client.current_prices.get(SYMBOL, 45000)
            buy_prices, sell_prices = self.calculate_grid_lines(current_price)
            
            # Create BUY orders below current price
            for price in buy_prices:
                try:
                    buy_order = await self.client.create_order(SYMBOL, 'BUY', 'LIMIT', ORDER_SIZE, price)
                    if buy_order:
                        self.grid_orders.append(buy_order)
                        logging.info(f"Placed paper BUY order at {price}")
                except Exception as e:
                    logging.error(f"Failed to place BUY order at {price}: {e}")
            
            # Create SELL orders above current price  
            for price in sell_prices:
                try:
                    sell_order = await self.client.create_order(SYMBOL, 'SELL', 'LIMIT', ORDER_SIZE, price)
                    if sell_order:
                        self.grid_orders.append(sell_order)
                        logging.info(f"Placed paper SELL order at {price}")
                except Exception as e:
                    logging.error(f"Failed to place SELL order at {price}: {e}")
            
            logging.info(f"Setup paper grid with {len(self.grid_orders)} orders")
            
        except Exception as e:
            logging.error(f"Error setting up paper grid: {e}")
    
    def calculate_grid_lines(self, current_price: float):
        """Calculate grid buy and sell prices"""
        buy_prices = []
        sell_prices = []
        
        for i in range(GRID_SIZE // 2):
            buy_prices.append(current_price - (i + 1) * GRID_INTERVAL)
            sell_prices.append(current_price + (i + 1) * GRID_INTERVAL)
        
        return sorted(buy_prices, reverse=True), sorted(sell_prices)
    
    async def check_grid(self):
        """Check for filled orders and replace them"""
        try:
            for order in self.grid_orders[:]:  # Copy to avoid modification during iteration
                try:
                    order_status = await self.client.get_order(SYMBOL, order['orderId'])
                    
                    if order_status['status'] == 'FILLED':
                        logging.info(f"Paper order {order['orderId']} filled!")
                        self.grid_orders.remove(order)
                        
                        # Track volume for VIP status
                        trade_volume = float(order_status['price']) * float(order_status['executedQty'])
                        self.volume_tracker.track_trade_volume(trade_volume, SYMBOL)
                        
                        # Replace filled order with new one on opposite side
                        if order_status['side'] == 'BUY':
                            # BUY filled, place SELL above
                            new_price = float(order_status['price']) + GRID_INTERVAL
                            new_order = await self.client.create_order(SYMBOL, 'SELL', 'LIMIT', ORDER_SIZE, new_price)
                            logging.info(f"BUY filled at {order_status['price']}, placing SELL at {new_price}")
                        else:
                            # SELL filled, place BUY below
                            new_price = float(order_status['price']) - GRID_INTERVAL
                            new_order = await self.client.create_order(SYMBOL, 'BUY', 'LIMIT', ORDER_SIZE, new_price)
                            logging.info(f"SELL filled at {order_status['price']}, placing BUY at {new_price}")
                        
                        if new_order:
                            self.grid_orders.append(new_order)
                            
                except Exception as e:
                    logging.debug(f"Error checking order status: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error checking grid: {e}")
    
    async def hedge_position(self):
        """Simulate hedging (simplified for paper trading)"""
        try:
            # Get balances
            btc_balance = await self.client.get_asset_balance('BTC')
            usdt_balance = await self.client.get_asset_balance('USDT')
            
            btc_amount = float(btc_balance['free'])
            usdt_amount = float(usdt_balance['free'])
            
            # Simple hedging simulation
            current_price = self.client.current_prices.get(SYMBOL, 45000)
            btc_value = btc_amount * current_price
            
            # Log position info
            if btc_amount > 0.001:  # Only log if we have significant BTC
                logging.debug(f"Current position - BTC: {btc_amount:.6f} (${btc_value:.2f}), USDT: {usdt_amount:.2f}")
                
        except Exception as e:
            logging.error(f"Error in hedge position: {e}")
    
    async def cleanup(self):
        """Cleanup and final reporting"""
        logging.info("Cleaning up paper trading bot...")
        
        try:
            # Cancel remaining orders
            open_orders = await self.client.get_open_orders(SYMBOL)
            for order in open_orders:
                try:
                    await self.client.cancel_order(SYMBOL, order['orderId'])
                except:
                    pass
            
            # Final status report
            logging.info("=== PAPER TRADING FINAL RESULTS ===")
            self.client.log_status()
            self.volume_tracker.log_vip_status()
            
            # Generate detailed report
            report = self.client.get_trading_report()
            pnl_data = report['pnl_data']
            
            logging.info(f"Total Trades: {report['total_trades']}")
            logging.info(f"Final P&L: {pnl_data.get('pnl', 0):.2f} USDT")
            logging.info(f"Return: {pnl_data.get('pnl_percentage', 0):.2f}%")
            logging.info("===================================")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

async def main():
    """Main function to run paper trading bot"""
    logging.info("Starting paper trading simulation...")
    
    # Create and run paper trading bot
    bot = PaperDeltaNeutralGridBot(initial_balance=10000)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logging.info("Paper trading simulation interrupted by user")
    except Exception as e:
        logging.error(f"Error in paper trading simulation: {e}")
    
    logging.info("Paper trading simulation completed")

if __name__ == "__main__":
    asyncio.run(main())