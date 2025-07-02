"""
Grid Trading Bot v3.0 - Data Manager
Handles data loading, validation, persistence, and state management.
"""

import sqlite3
import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume'])
        )

@dataclass
class OrderData:
    """Order data structure."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'limit', 'market'
    quantity: float
    price: float
    status: str  # 'new', 'filled', 'cancelled'
    timestamp: datetime
    filled_quantity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'filled_quantity': self.filled_quantity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderData':
        """Create from dictionary."""
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=data['side'],
            order_type=data['order_type'],
            quantity=float(data['quantity']),
            price=float(data['price']),
            status=data['status'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            filled_quantity=float(data.get('filled_quantity', 0.0))
        )

@dataclass
class PositionData:
    """Position data structure."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionData':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            quantity=float(data['quantity']),
            avg_price=float(data['avg_price']),
            current_price=float(data['current_price']),
            unrealized_pnl=float(data['unrealized_pnl']),
            realized_pnl=float(data['realized_pnl']),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class TradeData:
    """Trade execution data structure."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    entry_value: float
    exit_value: float
    commission: float
    pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_value': self.entry_value,
            'exit_value': self.exit_value,
            'commission': self.commission,
            'pnl': self.pnl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeData':
        """Create from dictionary."""
        return cls(
            trade_id=data['trade_id'],
            symbol=data['symbol'],
            side=data['side'],
            quantity=float(data['quantity']),
            entry_price=float(data['entry_price']),
            exit_price=float(data['exit_price']),
            entry_time=datetime.fromisoformat(data['entry_time']),
            exit_time=datetime.fromisoformat(data['exit_time']),
            entry_value=float(data['entry_value']),
            exit_value=float(data['exit_value']),
            commission=float(data['commission']),
            pnl=float(data['pnl'])
        )

class DataManager:
    """Manages data loading, validation, and persistence."""
    
    def __init__(self, config_or_path: Union[str, Any] = None):
        """
        Initialize data manager.
        
        Args:
            config_or_path: Either a path string or a config object.
        """
        if isinstance(config_or_path, str) or config_or_path is None:
            self.db_path = config_or_path or "data/grid_bot_v3.db"
        else:
            # Assume it's a config object
            self.db_path = getattr(config_or_path, 'database', {}).get('path', "data/grid_bot_v3.db")
        
        self._ensure_database_exists()
        self._create_tables()
        
    def _ensure_database_exists(self) -> None:
        """Ensure database directory and file exist."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_db_connection() as conn:
            # Market data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            ''')
            
            # Orders table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Positions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Bot state table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def validate_market_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate market data for completeness and quality.
        
        Args:
            data: Market data DataFrame.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if len(errors) > 0:
            return False, errors
        
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} is not numeric")
        
        # Check OHLC relationships
        if len(data) > 0:
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            )
            if invalid_ohlc.any():
                errors.append(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} rows")
        
        # Check for negative values
        negative_values = (data[numeric_columns] < 0).any()
        if negative_values.any():
            errors.append(f"Negative values found in: {negative_values[negative_values].index.tolist()}")
        
        # Check timestamp ordering
        if len(data) > 1:
            timestamps = pd.to_datetime(data['timestamp'])
            if not timestamps.is_monotonic_increasing:
                errors.append("Timestamps are not in ascending order")
        
        return len(errors) == 0, errors
    
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare market data.
        
        Args:
            data: Raw market data DataFrame.
            
        Returns:
            Cleaned market data DataFrame.
        """
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Convert timestamp to datetime
        cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'])
        
        # Sort by timestamp
        cleaned_data = cleaned_data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Forward fill missing values (conservative approach)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(method='ffill')
        
        # Drop rows that still have NaN values
        cleaned_data = cleaned_data.dropna(subset=numeric_columns)
        
        # Ensure positive values
        for col in numeric_columns:
            cleaned_data[col] = cleaned_data[col].abs()
        
        # Fix OHLC inconsistencies
        cleaned_data['high'] = cleaned_data[['open', 'high', 'low', 'close']].max(axis=1)
        cleaned_data['low'] = cleaned_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        logger.info(f"Cleaned market data: {len(data)} -> {len(cleaned_data)} rows")
        return cleaned_data
    
    def save_market_data(self, data: pd.DataFrame, symbol: str = "BTCUSDT") -> bool:
        """
        Save market data to database.
        
        Args:
            data: Market data DataFrame.
            symbol: Trading symbol.
            
        Returns:
            Success status.
        """
        try:
            # Validate data first
            is_valid, errors = self.validate_market_data(data)
            if not is_valid:
                logger.error(f"Market data validation failed: {errors}")
                return False
            
            # Clean data
            cleaned_data = self.clean_market_data(data)
            
            with self._get_db_connection() as conn:
                for _, row in cleaned_data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (timestamp, symbol, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['timestamp'].isoformat(),
                        symbol,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(cleaned_data)} market data records for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            return False
    
    def load_market_data(self, symbol: str = "BTCUSDT", 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load market data from database.
        
        Args:
            symbol: Trading symbol.
            start_time: Start time filter.
            end_time: End time filter.
            limit: Maximum number of records.
            
        Returns:
            Market data DataFrame.
        """
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with self._get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
            logger.info(f"Loaded {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def save_order(self, order: OrderData) -> bool:
        """
        Save order to database.
        
        Args:
            order: Order data.
            
        Returns:
            Success status.
        """
        try:
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO orders 
                    (order_id, symbol, side, order_type, quantity, price, status, timestamp, filled_quantity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order.order_id,
                    order.symbol,
                    order.side,
                    order.order_type,
                    order.quantity,
                    order.price,
                    order.status,
                    order.timestamp.isoformat(),
                    order.filled_quantity
                ))
                
                conn.commit()
                logger.debug(f"Saved order: {order.order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            return False
    
    def load_orders(self, symbol: str = "BTCUSDT", 
                   status: Optional[str] = None,
                   limit: Optional[int] = None) -> List[OrderData]:
        """
        Load orders from database.
        
        Args:
            symbol: Trading symbol.
            status: Order status filter.
            limit: Maximum number of records.
            
        Returns:
            List of order data.
        """
        try:
            query = "SELECT * FROM orders WHERE symbol = ?"
            params = [symbol]
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            orders = []
            for row in rows:
                orders.append(OrderData(
                    order_id=row['order_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    order_type=row['order_type'],
                    quantity=row['quantity'],
                    price=row['price'],
                    status=row['status'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    filled_quantity=row['filled_quantity']
                ))
            
            logger.debug(f"Loaded {len(orders)} orders for {symbol}")
            return orders
            
        except Exception as e:
            logger.error(f"Error loading orders: {e}")
            return []
    
    def save_bot_state(self, key: str, value: Any) -> bool:
        """
        Save bot state to database.
        
        Args:
            key: State key.
            value: State value.
            
        Returns:
            Success status.
        """
        try:
            # Serialize value to JSON
            serialized_value = json.dumps(value, default=str)
            
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO bot_state (key, value, timestamp)
                    VALUES (?, ?, ?)
                ''', (key, serialized_value, datetime.now().isoformat()))
                
                conn.commit()
                logger.debug(f"Saved bot state: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving bot state: {e}")
            return False
    
    def load_bot_state(self, key: str, default: Any = None) -> Any:
        """
        Load bot state from database.
        
        Args:
            key: State key.
            default: Default value if key not found.
            
        Returns:
            State value or default.
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT value FROM bot_state WHERE key = ? ORDER BY timestamp DESC LIMIT 1",
                    (key,)
                )
                row = cursor.fetchone()
            
            if row:
                return json.loads(row['value'])
            else:
                return default
                
        except Exception as e:
            logger.error(f"Error loading bot state: {e}")
            return default
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Path for backup file.
            
        Returns:
            Success status.
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path}.backup_{timestamp}"
            
            # Create backup using SQLite backup API
            with self._get_db_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def save_performance_snapshot(self, snapshot) -> bool:
        """
        Save performance snapshot to database.
        
        Args:
            snapshot: Performance snapshot object or dictionary.
            
        Returns:
            Success status.
        """
        try:
            # Handle both object and dictionary formats
            if hasattr(snapshot, 'timestamp'):
                # Object format
                timestamp = snapshot.timestamp.isoformat()
                portfolio_value = snapshot.portfolio_value
                unrealized_pnl = snapshot.unrealized_pnl
                realized_pnl = snapshot.realized_pnl
                total_return_pct = snapshot.total_return_pct
                sharpe_ratio = snapshot.sharpe_ratio
                max_drawdown = snapshot.max_drawdown
                win_rate = snapshot.win_rate
                total_trades = snapshot.total_trades
            else:
                # Dictionary format
                timestamp = snapshot.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                portfolio_value = snapshot.get('portfolio_value', 0)
                unrealized_pnl = snapshot.get('unrealized_pnl', 0)
                realized_pnl = snapshot.get('realized_pnl', 0)
                total_return_pct = snapshot.get('total_return_pct', 0)
                sharpe_ratio = snapshot.get('sharpe_ratio', 0)
                max_drawdown = snapshot.get('max_drawdown', 0)
                win_rate = snapshot.get('win_rate', 0)
                total_trades = snapshot.get('total_trades', 0)
            
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, total_value, unrealized_pnl, realized_pnl, total_return_pct, 
                     sharpe_ratio, max_drawdown, win_rate, total_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    portfolio_value,
                    unrealized_pnl,
                    realized_pnl,
                    total_return_pct,
                    sharpe_ratio,
                    max_drawdown,
                    win_rate,
                    total_trades
                ))
                
                conn.commit()
                logger.debug(f"Saved performance snapshot at {timestamp}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
            return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary.
        """
        try:
            stats = {}
            
            with self._get_db_connection() as conn:
                # Market data stats
                cursor = conn.execute("SELECT COUNT(*) as count FROM market_data")
                stats['market_data_count'] = cursor.fetchone()['count']
                
                # Orders stats
                cursor = conn.execute("SELECT COUNT(*) as count FROM orders")
                stats['orders_count'] = cursor.fetchone()['count']
                
                # Positions stats
                cursor = conn.execute("SELECT COUNT(*) as count FROM positions")
                stats['positions_count'] = cursor.fetchone()['count']
                
                # Performance metrics stats
                cursor = conn.execute("SELECT COUNT(*) as count FROM performance_metrics")
                stats['performance_metrics_count'] = cursor.fetchone()['count']
                
                # Database size
                cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                stats['database_size_bytes'] = cursor.fetchone()['size']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}