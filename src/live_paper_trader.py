import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from paper_trading import PaperTradingClient
from data_fetcher import HistoricalDataFetcher
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LivePaperTrader:
    """
    Live paper trading interface with real-time price simulation
    Uses historical data to simulate live trading conditions
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.paper_client = PaperTradingClient(initial_balance)
        self.data_fetcher = HistoricalDataFetcher()
        self.historical_data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self.is_running = False
        self.config = {
            'symbol': 'BTCUSDT',
            'grid_size': 10,
            'grid_interval': 500,
            'order_size': 0.001,
            'speed_multiplier': 1  # 1 = real time, higher = faster simulation
        }
        self.grid_orders = []
        self.session_stats = {
            'start_time': None,
            'start_balance': initial_balance,
            'trades_executed': 0,
            'session_pnl': 0
        }
    
    async def load_simulation_data(self, data_file: str = None, period: str = 'recent') -> bool:
        """Load historical data for simulation"""
        try:
            if data_file and os.path.exists(data_file):
                self.historical_data = self.data_fetcher.load_data(data_file)
                logging.info(f"Loaded simulation data from {data_file}")
                return True
            
            # Load recent data for simulation
            if period == 'recent':
                # Try to load existing 2021 data
                data_files = [
                    "data/btc_2021_1h_binance.csv",
                    "data/btc_2021_daily_coingecko.csv"
                ]
                
                for file_path in data_files:
                    if os.path.exists(file_path):
                        self.historical_data = self.data_fetcher.load_data(file_path)
                        logging.info(f"Loaded existing data from {file_path}")
                        return True
                
                # If no existing data, fetch new
                logging.info("No existing data found, fetching fresh data...")
                self.historical_data = self.data_fetcher.fetch_btc_2021_data(interval='1h')
                
                if self.historical_data is not None:
                    logging.info("Successfully loaded fresh simulation data")
                    return True
            
            logging.error("Failed to load simulation data")
            return False
            
        except Exception as e:
            logging.error(f"Error loading simulation data: {e}")
            return False
    
    async def setup_grid_strategy(self, current_price: float):
        """Setup initial grid trading strategy"""
        try:
            logging.info(f"Setting up grid strategy around price ${current_price:,.2f}")
            
            # Clear existing orders
            self.grid_orders = []
            
            # Create buy orders below current price
            for i in range(1, self.config['grid_size'] // 2 + 1):
                buy_price = current_price - (i * self.config['grid_interval'])
                if buy_price > 0:
                    order = await self.paper_client.create_order(
                        symbol=self.config['symbol'],
                        side='BUY',
                        type='LIMIT',
                        quantity=self.config['order_size'],
                        price=buy_price
                    )
                    self.grid_orders.append(order)
            
            # Create sell orders above current price
            for i in range(1, self.config['grid_size'] // 2 + 1):
                sell_price = current_price + (i * self.config['grid_interval'])
                order = await self.paper_client.create_order(
                    symbol=self.config['symbol'],
                    side='SELL',
                    type='LIMIT',
                    quantity=self.config['order_size'],
                    price=sell_price
                )
                self.grid_orders.append(order)
            
            logging.info(f"Grid setup complete with {len(self.grid_orders)} orders")
            
        except Exception as e:
            logging.error(f"Error setting up grid strategy: {e}")
    
    async def check_and_replace_orders(self, current_price: float):
        """Check for filled orders and replace them"""
        try:
            filled_orders = []
            
            # Check all grid orders for fills
            for order in self.grid_orders[:]:
                try:
                    order_status = await self.paper_client.get_order(
                        self.config['symbol'], order['orderId']
                    )
                    
                    if order_status['status'] == 'FILLED':
                        filled_orders.append(order_status)
                        self.grid_orders.remove(order)
                        self.session_stats['trades_executed'] += 1
                        
                        logging.info(f"Order filled: {order_status['side']} {order_status['quantity']} at ${float(order_status['price']):,.2f}")
                        
                except:
                    continue
            
            # Replace filled orders
            for filled_order in filled_orders:
                try:
                    if filled_order['side'] == 'BUY':
                        # BUY filled, place SELL above
                        new_price = float(filled_order['price']) + self.config['grid_interval']
                        new_order = await self.paper_client.create_order(
                            symbol=self.config['symbol'],
                            side='SELL',
                            type='LIMIT',
                            quantity=self.config['order_size'],
                            price=new_price
                        )
                    else:
                        # SELL filled, place BUY below
                        new_price = float(filled_order['price']) - self.config['grid_interval']
                        if new_price > 0:
                            new_order = await self.paper_client.create_order(
                                symbol=self.config['symbol'],
                                side='BUY',
                                type='LIMIT',
                                quantity=self.config['order_size'],
                                price=new_price
                            )
                    
                    self.grid_orders.append(new_order)
                    
                except Exception as e:
                    logging.warning(f"Failed to replace filled order: {e}")
            
        except Exception as e:
            logging.error(f"Error checking orders: {e}")
    
    async def simulate_live_trading(self, duration_minutes: int = 60):
        """Simulate live trading for specified duration"""
        try:
            if self.historical_data is None:
                logging.error("No simulation data loaded")
                return False
            
            self.is_running = True
            self.session_stats['start_time'] = datetime.now()
            start_balance = self.paper_client.calculate_pnl().get('total_value', self.session_stats['start_balance'])
            
            logging.info(f"Starting live paper trading simulation for {duration_minutes} minutes")
            logging.info(f"Starting balance: ${start_balance:,.2f}")
            
            # Start from a random point in the data
            import random
            max_start_index = len(self.historical_data) - (duration_minutes * 60)  # Assuming hourly data
            self.current_index = random.randint(0, max(0, max_start_index))
            
            # Get initial price and setup grid
            initial_price = self.historical_data.iloc[self.current_index]['price']
            self.paper_client.update_price(self.config['symbol'], initial_price)
            await self.setup_grid_strategy(initial_price)
            
            end_time = time.time() + (duration_minutes * 60 / self.config['speed_multiplier'])
            last_update = time.time()
            
            while self.is_running and time.time() < end_time and self.current_index < len(self.historical_data) - 1:
                # Get current price data
                current_row = self.historical_data.iloc[self.current_index]
                current_price = current_row['price']
                timestamp = current_row['timestamp']
                
                # Update price in paper client
                self.paper_client.update_price(self.config['symbol'], current_price)
                
                # Check and replace filled orders
                await self.check_and_replace_orders(current_price)
                
                # Log status every 5 minutes (in simulation time)
                if time.time() - last_update >= 300 / self.config['speed_multiplier']:
                    await self.log_status(current_price, timestamp)
                    last_update = time.time()
                
                # Move to next data point
                self.current_index += 1
                
                # Wait based on speed multiplier
                await asyncio.sleep(1 / self.config['speed_multiplier'])
            
            # Final status
            await self.log_final_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Error in live trading simulation: {e}")
            return False
        finally:
            self.is_running = False
    
    async def log_status(self, current_price: float, timestamp):
        """Log current trading status"""
        try:
            pnl_data = self.paper_client.calculate_pnl()
            
            logging.info(f"\n=== STATUS UPDATE ===")
            logging.info(f"Time: {timestamp}")
            logging.info(f"Current Price: ${current_price:,.2f}")
            logging.info(f"Total Value: ${pnl_data.get('total_value', 0):,.2f}")
            logging.info(f"P&L: ${pnl_data.get('pnl', 0):,.2f} ({pnl_data.get('pnl_percentage', 0):.2f}%)")
            logging.info(f"Open Orders: {len(self.grid_orders)}")
            logging.info(f"Trades Executed: {self.session_stats['trades_executed']}")
            logging.info(f"Balances: {self.paper_client.balances}")
            logging.info("====================")
            
        except Exception as e:
            logging.error(f"Error logging status: {e}")
    
    async def log_final_status(self):
        """Log final session results"""
        try:
            final_pnl = self.paper_client.calculate_pnl()
            session_duration = (datetime.now() - self.session_stats['start_time']).total_seconds() / 60
            
            logging.info(f"\n" + "="*50)
            logging.info("PAPER TRADING SESSION COMPLETE")
            logging.info("="*50)
            logging.info(f"Session Duration: {session_duration:.1f} minutes")
            logging.info(f"Starting Balance: ${self.session_stats['start_balance']:,.2f}")
            logging.info(f"Final Balance: ${final_pnl.get('total_value', 0):,.2f}")
            logging.info(f"Total P&L: ${final_pnl.get('pnl', 0):,.2f}")
            logging.info(f"P&L Percentage: {final_pnl.get('pnl_percentage', 0):.2f}%")
            logging.info(f"Total Trades: {self.session_stats['trades_executed']}")
            logging.info(f"Trades per Hour: {(self.session_stats['trades_executed'] / (session_duration / 60)):.1f}")
            logging.info("="*50)
            
            # Save session report
            report = {
                'session_start': self.session_stats['start_time'].isoformat(),
                'session_end': datetime.now().isoformat(),
                'duration_minutes': session_duration,
                'starting_balance': self.session_stats['start_balance'],
                'final_balance': final_pnl.get('total_value', 0),
                'total_pnl': final_pnl.get('pnl', 0),
                'pnl_percentage': final_pnl.get('pnl_percentage', 0),
                'total_trades': self.session_stats['trades_executed'],
                'final_balances': self.paper_client.balances.copy(),
                'config': self.config.copy()
            }
            
            # Save to file
            os.makedirs('session_reports', exist_ok=True)
            filename = f"session_reports/paper_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Session report saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error in final status log: {e}")
    
    def update_config(self, **kwargs):
        """Update trading configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logging.info(f"Updated {key} to {value}")
    
    async def stop_trading(self):
        """Stop the trading simulation"""
        self.is_running = False
        logging.info("Trading simulation stopped by user")
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time status"""
        try:
            pnl_data = self.paper_client.calculate_pnl()
            
            if self.historical_data is not None and self.current_index < len(self.historical_data):
                current_price = self.historical_data.iloc[self.current_index]['price']
                timestamp = self.historical_data.iloc[self.current_index]['timestamp']
            else:
                current_price = 0
                timestamp = datetime.now()
            
            status = {
                'is_running': self.is_running,
                'current_price': current_price,
                'timestamp': timestamp,
                'total_value': pnl_data.get('total_value', 0),
                'pnl': pnl_data.get('pnl', 0),
                'pnl_percentage': pnl_data.get('pnl_percentage', 0),
                'open_orders': len(self.grid_orders),
                'trades_executed': self.session_stats['trades_executed'],
                'balances': self.paper_client.balances.copy(),
                'config': self.config.copy()
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Error getting real-time status: {e}")
            return {}

class InteractivePaperTrader:
    """
    Interactive interface for paper trading
    """
    
    def __init__(self):
        self.trader = LivePaperTrader()
        self.commands = {
            'start': self.start_trading,
            'stop': self.stop_trading,
            'status': self.show_status,
            'config': self.show_config,
            'update': self.update_config,
            'help': self.show_help,
            'quit': self.quit_trader
        }
    
    async def start_trading(self, duration: int = 60):
        """Start paper trading simulation"""
        try:
            # Load data
            print("Loading simulation data...")
            if not await self.trader.load_simulation_data():
                print("❌ Failed to load simulation data")
                return
            
            print(f"✅ Starting paper trading simulation for {duration} minutes")
            await self.trader.simulate_live_trading(duration)
            
        except Exception as e:
            print(f"❌ Error starting trading: {e}")
    
    async def stop_trading(self):
        """Stop trading simulation"""
        await self.trader.stop_trading()
        print("🛑 Trading stopped")
    
    def show_status(self):
        """Show current status"""
        status = self.trader.get_real_time_status()
        
        print("\n" + "="*40)
        print("CURRENT STATUS")
        print("="*40)
        print(f"Running: {'✅ Yes' if status.get('is_running') else '❌ No'}")
        print(f"Current Price: ${status.get('current_price', 0):,.2f}")
        print(f"Total Value: ${status.get('total_value', 0):,.2f}")
        print(f"P&L: ${status.get('pnl', 0):,.2f} ({status.get('pnl_percentage', 0):.2f}%)")
        print(f"Open Orders: {status.get('open_orders', 0)}")
        print(f"Trades: {status.get('trades_executed', 0)}")
        print(f"Balances: {status.get('balances', {})}")
        print("="*40)
    
    def show_config(self):
        """Show current configuration"""
        config = self.trader.config
        
        print("\n" + "="*40)
        print("CURRENT CONFIGURATION")
        print("="*40)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("="*40)
    
    def update_config(self, **kwargs):
        """Update configuration"""
        self.trader.update_config(**kwargs)
        print("✅ Configuration updated")
    
    def show_help(self):
        """Show available commands"""
        print("\n" + "="*50)
        print("AVAILABLE COMMANDS")
        print("="*50)
        print("start <duration>  - Start trading (default: 60 minutes)")
        print("stop             - Stop trading")
        print("status           - Show current status")
        print("config           - Show configuration")
        print("update key=value - Update configuration")
        print("help             - Show this help")
        print("quit             - Exit the program")
        print("="*50)
    
    def quit_trader(self):
        """Quit the application"""
        print("👋 Goodbye!")
        return True
    
    async def run_interactive(self):
        """Run interactive paper trading interface"""
        print("🚀 Welcome to Interactive Paper Trading!")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command_input = input("\n📊 Paper Trader > ").strip().lower()
                
                if not command_input:
                    continue
                
                parts = command_input.split()
                command = parts[0]
                
                if command == 'quit':
                    break
                elif command == 'start':
                    duration = int(parts[1]) if len(parts) > 1 else 60
                    await self.start_trading(duration)
                elif command == 'stop':
                    await self.stop_trading()
                elif command == 'status':
                    self.show_status()
                elif command == 'config':
                    self.show_config()
                elif command == 'help':
                    self.show_help()
                elif command == 'update':
                    if len(parts) > 1:
                        # Parse key=value pairs
                        updates = {}
                        for arg in parts[1:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                try:
                                    # Try to convert to appropriate type
                                    if value.isdigit():
                                        value = int(value)
                                    elif value.replace('.', '').isdigit():
                                        value = float(value)
                                    updates[key] = value
                                except:
                                    updates[key] = value
                        
                        if updates:
                            self.update_config(**updates)
                        else:
                            print("❌ Invalid update format. Use: update key=value")
                    else:
                        print("❌ Please specify configuration updates. Use: update key=value")
                else:
                    print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Clean exit
        if self.trader.is_running:
            await self.trader.stop_trading()
        
        print("👋 Paper trading session ended")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Run interactive mode
        interactive_trader = InteractivePaperTrader()
        asyncio.run(interactive_trader.run_interactive())
    else:
        # Run simple simulation
        async def main():
            trader = LivePaperTrader(initial_balance=10000)
            
            print("Loading simulation data...")
            if await trader.load_simulation_data():
                print("Starting 30-minute paper trading simulation...")
                await trader.simulate_live_trading(duration_minutes=30)
            else:
                print("Failed to load simulation data")
        
        asyncio.run(main())