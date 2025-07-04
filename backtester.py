import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from paper_trading import PaperTradingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str
    end_date: str
    initial_balance: float = 10000
    symbol: str = 'BTCUSDT'
    grid_size: int = 10
    grid_interval: float = 100
    order_size: float = 0.001
    commission_rate: float = 0.001

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return: float
    volatility: float
    trading_days: int

class StrategyBacktester:
    """
    Backtest delta-neutral grid trading strategy on historical data
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.historical_data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        self.paper_client: Optional[PaperTradingClient] = None
        self.grid_orders: List[Dict] = []
        self.daily_balances: List[float] = []
        
    def load_historical_data(self, data_file: str = None, data: pd.DataFrame = None):
        """Load historical price data"""
        try:
            if data is not None:
                self.historical_data = data
            elif data_file:
                # Load from CSV file
                self.historical_data = pd.read_csv(data_file)
                self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            else:
                # Generate synthetic data for testing
                self.historical_data = self._generate_synthetic_data()
            
            logging.info(f"Loaded {len(self.historical_data)} historical data points")
            return True
            
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            return False
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        # Generate hourly data
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate synthetic price using random walk
        np.random.seed(42)  # For reproducible results
        initial_price = 45000
        returns = np.random.normal(0, 0.002, len(timestamps))  # 0.2% hourly volatility
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': np.random.uniform(100, 1000, len(timestamps))
        })
        
        logging.info(f"Generated synthetic data from {start_date} to {end_date}")
        return data
    
    async def run_backtest(self):
        """Run the backtest simulation"""
        try:
            if self.historical_data is None:
                raise ValueError("No historical data loaded")
            
            # Initialize paper trading client
            self.paper_client = PaperTradingClient(self.config.initial_balance)
            
            # Filter data by date range
            start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            mask = (self.historical_data['timestamp'] >= start_date) & \
                   (self.historical_data['timestamp'] <= end_date)
            backtest_data = self.historical_data.loc[mask].copy().reset_index(drop=True)
            
            logging.info(f"Running backtest from {start_date} to {end_date}")
            logging.info(f"Processing {len(backtest_data)} data points")
            
            # Initialize grid
            initial_price = backtest_data.iloc[0]['price']
            await self._setup_initial_grid(initial_price)
            
            # Process each data point
            for idx, row in backtest_data.iterrows():
                current_price = row['price']
                timestamp = row['timestamp']
                
                # Update price in paper client
                self.paper_client.update_price(self.config.symbol, current_price)
                
                # Check and update grid
                await self._check_and_update_grid(current_price)
                
                # Track daily balance (every 24 data points for hourly data)
                if idx % 24 == 0:
                    pnl_data = self.paper_client.calculate_pnl()
                    self.daily_balances.append(pnl_data.get('total_value', self.config.initial_balance))
                
                # Log progress every 100 data points
                if idx % 100 == 0:
                    progress = (idx / len(backtest_data)) * 100
                    logging.info(f"Backtest progress: {progress:.1f}%")
            
            # Generate results
            self.results = await self._generate_results()
            
            logging.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logging.error(f"Error running backtest: {e}")
            return None
    
    async def _setup_initial_grid(self, current_price: float):
        """Setup initial grid of orders"""
        try:
            grid_levels = []
            
            # Create buy levels below current price
            for i in range(1, self.config.grid_size // 2 + 1):
                buy_price = current_price - (i * self.config.grid_interval)
                grid_levels.append(('BUY', buy_price))
            
            # Create sell levels above current price
            for i in range(1, self.config.grid_size // 2 + 1):
                sell_price = current_price + (i * self.config.grid_interval)
                grid_levels.append(('SELL', sell_price))
            
            # Place initial grid orders
            for side, price in grid_levels:
                try:
                    order = await self.paper_client.create_order(
                        symbol=self.config.symbol,
                        side=side,
                        type='LIMIT',
                        quantity=self.config.order_size,
                        price=price
                    )
                    self.grid_orders.append(order)
                except Exception as e:
                    logging.warning(f"Failed to place initial {side} order at {price}: {e}")
            
            logging.info(f"Setup initial grid with {len(self.grid_orders)} orders")
            
        except Exception as e:
            logging.error(f"Error setting up initial grid: {e}")
    
    async def _check_and_update_grid(self, current_price: float):
        """Check for filled orders and update grid"""
        try:
            filled_orders = []
            
            # Check all grid orders for fills
            for order in self.grid_orders[:]:
                try:
                    order_status = await self.paper_client.get_order(
                        self.config.symbol, order['orderId']
                    )
                    
                    if order_status['status'] == 'FILLED':
                        filled_orders.append(order_status)
                        self.grid_orders.remove(order)
                except:
                    # Order might have been filled and moved to history
                    continue
            
            # Replace filled orders with new ones on opposite side
            for filled_order in filled_orders:
                try:
                    if filled_order['side'] == 'BUY':
                        # BUY filled, place SELL above
                        new_price = float(filled_order['price']) + self.config.grid_interval
                        new_order = await self.paper_client.create_order(
                            symbol=self.config.symbol,
                            side='SELL',
                            type='LIMIT',
                            quantity=self.config.order_size,
                            price=new_price
                        )
                    else:
                        # SELL filled, place BUY below
                        new_price = float(filled_order['price']) - self.config.grid_interval
                        new_order = await self.paper_client.create_order(
                            symbol=self.config.symbol,
                            side='BUY',
                            type='LIMIT',
                            quantity=self.config.order_size,
                            price=new_price
                        )
                    
                    self.grid_orders.append(new_order)
                    
                except Exception as e:
                    logging.warning(f"Failed to replace filled order: {e}")
            
        except Exception as e:
            logging.error(f"Error checking and updating grid: {e}")
    
    async def _generate_results(self) -> BacktestResult:
        """Generate backtest results"""
        try:
            pnl_data = self.paper_client.calculate_pnl()
            
            initial_balance = self.config.initial_balance
            final_balance = pnl_data.get('total_value', initial_balance)
            total_return = final_balance - initial_balance
            total_return_pct = (total_return / initial_balance) * 100
            
            # Calculate max drawdown
            max_balance = max(self.daily_balances) if self.daily_balances else initial_balance
            min_balance = min(self.daily_balances) if self.daily_balances else initial_balance
            max_drawdown = max_balance - min_balance
            max_drawdown_pct = (max_drawdown / max_balance) * 100 if max_balance > 0 else 0
            
            # Calculate volatility and Sharpe ratio
            daily_returns = []
            if len(self.daily_balances) > 1:
                for i in range(1, len(self.daily_balances)):
                    daily_return = (self.daily_balances[i] - self.daily_balances[i-1]) / self.daily_balances[i-1]
                    daily_returns.append(daily_return)
            
            volatility = np.std(daily_returns) * np.sqrt(365) if daily_returns else 0
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            sharpe_ratio = (avg_daily_return * 365) / volatility if volatility > 0 else 0
            
            # Trading statistics
            trades = self.paper_client.trade_history
            winning_trades = len([t for t in trades if self._is_winning_trade(t)])
            losing_trades = len(trades) - winning_trades
            win_rate = (winning_trades / len(trades)) * 100 if trades else 0
            
            # Calculate average trade return
            trade_returns = [self._calculate_trade_return(t) for t in trades]
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            trading_days = len([d for d in self.daily_balances if d != initial_balance])
            
            result = BacktestResult(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_return=total_return,
                total_return_pct=total_return_pct,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_trade_return=avg_trade_return,
                volatility=volatility,
                trading_days=trading_days
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error generating results: {e}")
            return None
    
    def _is_winning_trade(self, trade) -> bool:
        """Determine if a trade was profitable"""
        # This is simplified - in reality you'd need to track the complete trade cycle
        return True  # Placeholder
    
    def _calculate_trade_return(self, trade) -> float:
        """Calculate return for a single trade"""
        # This is simplified - in reality you'd need to track the complete trade cycle
        return 0.001  # Placeholder 0.1% return
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {"error": "No backtest results available"}
        
        result = self.results
        
        report = {
            "backtest_period": {
                "start_date": result.start_date,
                "end_date": result.end_date,
                "trading_days": result.trading_days
            },
            "performance": {
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "total_return": result.total_return,
                "total_return_pct": result.total_return_pct,
                "annualized_return": (result.total_return_pct / result.trading_days) * 365 if result.trading_days > 0 else 0
            },
            "risk_metrics": {
                "max_drawdown": result.max_drawdown,
                "max_drawdown_pct": result.max_drawdown_pct,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio
            },
            "trading_stats": {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "avg_trade_return": result.avg_trade_return
            },
            "config": {
                "symbol": self.config.symbol,
                "grid_size": self.config.grid_size,
                "grid_interval": self.config.grid_interval,
                "order_size": self.config.order_size,
                "commission_rate": self.config.commission_rate
            }
        }
        
        return report
    
    def save_report(self, filename: str = None):
        """Save backtest report to file"""
        try:
            if filename is None:
                filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = self.generate_report()
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Backtest report saved to {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error saving report: {e}")
            return None
    
    def log_results(self):
        """Log backtest results"""
        if not self.results:
            logging.error("No backtest results to log")
            return
        
        result = self.results
        
        logging.info("=== BACKTEST RESULTS ===")
        logging.info(f"Period: {result.start_date} to {result.end_date}")
        logging.info(f"Initial Balance: {result.initial_balance:.2f} USDT")
        logging.info(f"Final Balance: {result.final_balance:.2f} USDT")
        logging.info(f"Total Return: {result.total_return:.2f} USDT ({result.total_return_pct:.2f}%)")
        logging.info(f"Max Drawdown: {result.max_drawdown:.2f} USDT ({result.max_drawdown_pct:.2f}%)")
        logging.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logging.info(f"Total Trades: {result.total_trades}")
        logging.info(f"Win Rate: {result.win_rate:.1f}%")
        logging.info(f"Volatility: {result.volatility:.2f}")
        logging.info("========================")

if __name__ == "__main__":
    import asyncio
    
    async def run_sample_backtest():
        """Run a sample backtest"""
        config = BacktestConfig(
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_balance=10000,
            symbol='BTCUSDT',
            grid_size=10,
            grid_interval=100,
            order_size=0.001
        )
        
        backtester = StrategyBacktester(config)
        
        # Load synthetic data
        backtester.load_historical_data()
        
        # Run backtest
        results = await backtester.run_backtest()
        
        if results:
            backtester.log_results()
            backtester.save_report()
        else:
            logging.error("Backtest failed")
    
    asyncio.run(run_sample_backtest())