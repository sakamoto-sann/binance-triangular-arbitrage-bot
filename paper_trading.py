import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PaperOrder:
    """Represents a paper trading order"""
    orderId: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    type: str  # 'LIMIT', 'MARKET', etc.
    quantity: float
    price: float
    status: str = 'NEW'  # 'NEW', 'FILLED', 'CANCELLED'
    executedQty: float = 0.0
    timestamp: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time() * 1000)

@dataclass
class PaperTrade:
    """Represents a paper trade execution"""
    tradeId: str
    orderId: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: int
    commission: float = 0.0
    commissionAsset: str = 'USDT'

class PaperTradingClient:
    """
    Paper trading client for safe testing without real money
    Simulates Binance API responses for testing purposes
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.balances = {
            'USDT': initial_balance,
            'BTC': 0.0,
            'ETH': 0.0
        }
        self.open_orders: List[PaperOrder] = []
        self.order_history: List[PaperOrder] = []
        self.trade_history: List[PaperTrade] = []
        self.current_prices: Dict[str, float] = {
            'BTCUSDT': 45000.0,  # Default BTC price
            'ETHUSDT': 3000.0    # Default ETH price
        }
        self.order_id_counter = 1
        self.trade_id_counter = 1
        self.commission_rate = 0.001  # 0.1% commission
        
        logging.info(f"Paper trading client initialized with {initial_balance} USDT")
    
    def update_price(self, symbol: str, price: float):
        """Update current price for a symbol"""
        self.current_prices[symbol] = price
        logging.debug(f"Updated {symbol} price to {price}")
        
        # Check if any limit orders should be filled
        self._check_limit_orders()
    
    def _check_limit_orders(self):
        """Check and fill limit orders based on current prices"""
        for order in self.open_orders[:]:  # Copy to avoid modification during iteration
            if order.type == 'LIMIT':
                current_price = self.current_prices.get(order.symbol, 0)
                
                should_fill = False
                if order.side == 'BUY' and current_price <= order.price:
                    should_fill = True
                elif order.side == 'SELL' and current_price >= order.price:
                    should_fill = True
                
                if should_fill:
                    self._fill_order(order, order.price)
    
    def _fill_order(self, order: PaperOrder, fill_price: float):
        """Fill an order and update balances"""
        try:
            base_asset = order.symbol.replace('USDT', '')
            quote_asset = 'USDT'
            
            # Calculate commission
            commission = order.quantity * fill_price * self.commission_rate
            
            if order.side == 'BUY':
                # Buy: spend USDT, get base asset
                total_cost = order.quantity * fill_price + commission
                if self.balances.get(quote_asset, 0) >= total_cost:
                    self.balances[quote_asset] -= total_cost
                    self.balances[base_asset] = self.balances.get(base_asset, 0) + order.quantity
                    
                    # Update order
                    order.status = 'FILLED'
                    order.executedQty = order.quantity
                    
                    # Create trade record
                    trade = PaperTrade(
                        tradeId=str(self.trade_id_counter),
                        orderId=order.orderId,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        timestamp=int(time.time() * 1000),
                        commission=commission,
                        commissionAsset=quote_asset
                    )
                    self.trade_history.append(trade)
                    self.trade_id_counter += 1
                    
                    logging.info(f"Paper trade executed: BUY {order.quantity} {base_asset} at {fill_price} USDT")
                else:
                    logging.warning(f"Insufficient {quote_asset} balance for BUY order")
                    return False
                    
            else:  # SELL
                # Sell: spend base asset, get USDT
                if self.balances.get(base_asset, 0) >= order.quantity:
                    self.balances[base_asset] -= order.quantity
                    proceeds = order.quantity * fill_price - commission
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0) + proceeds
                    
                    # Update order
                    order.status = 'FILLED'
                    order.executedQty = order.quantity
                    
                    # Create trade record
                    trade = PaperTrade(
                        tradeId=str(self.trade_id_counter),
                        orderId=order.orderId,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        timestamp=int(time.time() * 1000),
                        commission=commission,
                        commissionAsset=quote_asset
                    )
                    self.trade_history.append(trade)
                    self.trade_id_counter += 1
                    
                    logging.info(f"Paper trade executed: SELL {order.quantity} {base_asset} at {fill_price} USDT")
                else:
                    logging.warning(f"Insufficient {base_asset} balance for SELL order")
                    return False
            
            # Move from open orders to history
            self.open_orders.remove(order)
            self.order_history.append(order)
            
            return True
            
        except Exception as e:
            logging.error(f"Error filling paper order: {e}")
            return False
    
    async def create_order(self, symbol: str, side: str, type: str, quantity: float, 
                          price: Optional[float] = None, timeInForce: str = 'GTC'):
        """Create a paper trading order"""
        try:
            order = PaperOrder(
                orderId=str(self.order_id_counter),
                symbol=symbol,
                side=side,
                type=type,
                quantity=quantity,
                price=price or self.current_prices.get(symbol, 0),
                status='NEW'
            )
            self.order_id_counter += 1
            
            if type == 'MARKET':
                # Market orders fill immediately
                fill_price = self.current_prices.get(symbol, 0)
                if self._fill_order(order, fill_price):
                    return asdict(order)
                else:
                    raise Exception("Insufficient balance for market order")
            else:
                # Limit orders go to open orders
                self.open_orders.append(order)
                logging.info(f"Paper order placed: {side} {quantity} {symbol} at {price}")
                return asdict(order)
                
        except Exception as e:
            logging.error(f"Error creating paper order: {e}")
            raise
    
    async def get_order(self, symbol: str, orderId: str):
        """Get order status"""
        # Check open orders first
        for order in self.open_orders:
            if order.orderId == orderId and order.symbol == symbol:
                return asdict(order)
        
        # Check order history
        for order in self.order_history:
            if order.orderId == orderId and order.symbol == symbol:
                return asdict(order)
        
        raise Exception(f"Order {orderId} not found")
    
    async def get_open_orders(self, symbol: str = None):
        """Get open orders"""
        if symbol:
            return [asdict(order) for order in self.open_orders if order.symbol == symbol]
        return [asdict(order) for order in self.open_orders]
    
    async def cancel_order(self, symbol: str, orderId: str):
        """Cancel an order"""
        for order in self.open_orders:
            if order.orderId == orderId and order.symbol == symbol:
                order.status = 'CANCELLED'
                self.open_orders.remove(order)
                self.order_history.append(order)
                logging.info(f"Paper order cancelled: {orderId}")
                return asdict(order)
        
        raise Exception(f"Order {orderId} not found or already filled")
    
    async def get_asset_balance(self, asset: str):
        """Get asset balance"""
        balance = self.balances.get(asset, 0)
        return {
            'asset': asset,
            'free': str(balance),
            'locked': '0.0'
        }
    
    async def get_symbol_ticker(self, symbol: str):
        """Get symbol ticker"""
        price = self.current_prices.get(symbol, 0)
        return {
            'symbol': symbol,
            'price': str(price)
        }
    
    def calculate_pnl(self):
        """Calculate paper trading P&L"""
        try:
            initial_balance = 10000  # Assuming initial balance
            current_usdt = self.balances.get('USDT', 0)
            
            # Calculate value of crypto holdings
            crypto_value = 0
            for asset, balance in self.balances.items():
                if asset != 'USDT' and balance > 0:
                    symbol = f"{asset}USDT"
                    price = self.current_prices.get(symbol, 0)
                    crypto_value += balance * price
            
            total_value = current_usdt + crypto_value
            pnl = total_value - initial_balance
            pnl_percentage = (pnl / initial_balance) * 100
            
            return {
                'initial_balance': initial_balance,
                'current_usdt': current_usdt,
                'crypto_value': crypto_value,
                'total_value': total_value,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logging.error(f"Error calculating P&L: {e}")
            return {}
    
    def get_trading_report(self):
        """Generate comprehensive trading report"""
        pnl_data = self.calculate_pnl()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'balances': self.balances.copy(),
            'pnl_data': pnl_data,
            'open_orders': len(self.open_orders),
            'total_orders': len(self.order_history) + len(self.open_orders),
            'total_trades': len(self.trade_history),
            'current_prices': self.current_prices.copy()
        }
        
        return report
    
    def log_status(self):
        """Log current paper trading status"""
        report = self.get_trading_report()
        pnl_data = report['pnl_data']
        
        logging.info("=== PAPER TRADING STATUS ===")
        logging.info(f"Balances: {report['balances']}")
        logging.info(f"Total Value: {pnl_data.get('total_value', 0):.2f} USDT")
        logging.info(f"P&L: {pnl_data.get('pnl', 0):.2f} USDT ({pnl_data.get('pnl_percentage', 0):.2f}%)")
        logging.info(f"Open Orders: {report['open_orders']}")
        logging.info(f"Total Trades: {report['total_trades']}")
        logging.info("===========================")

if __name__ == "__main__":
    import asyncio
    
    async def test_paper_trading():
        """Test the paper trading system"""
        client = PaperTradingClient(10000)
        
        # Update BTC price
        client.update_price('BTCUSDT', 45000)
        
        # Place a buy order
        order = await client.create_order('BTCUSDT', 'BUY', 'LIMIT', 0.1, 44000)
        print(f"Buy order placed: {order}")
        
        # Update price to trigger fill
        client.update_price('BTCUSDT', 43000)
        
        # Check order status
        filled_order = await client.get_order('BTCUSDT', order['orderId'])
        print(f"Order status: {filled_order}")
        
        # Place a sell order
        sell_order = await client.create_order('BTCUSDT', 'SELL', 'LIMIT', 0.05, 46000)
        print(f"Sell order placed: {sell_order}")
        
        # Update price to trigger sell
        client.update_price('BTCUSDT', 47000)
        
        # Log final status
        client.log_status()
    
    asyncio.run(test_paper_trading())