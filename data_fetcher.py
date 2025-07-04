import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HistoricalDataFetcher:
    """
    Fetch historical cryptocurrency data from various sources
    """
    
    def __init__(self):
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
    def fetch_binance_klines(self, symbol: str, interval: str, start_time: str, end_time: str, 
                           limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch historical kline data from Binance API
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            start_time: Start time in 'YYYY-MM-DD' format
            end_time: End time in 'YYYY-MM-DD' format
            limit: Number of data points per request (max 1000)
        """
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)
            
            all_data = []
            current_start = start_ts
            
            while current_start < end_ts:
                url = f"{self.binance_base_url}/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': limit
                }
                
                logging.info(f"Fetching data from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(end_ts/1000)}")
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        break
                        
                    all_data.extend(data)
                    
                    # Update start time for next batch
                    current_start = data[-1][6] + 1  # Close time + 1ms
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                else:
                    logging.error(f"Error fetching data: {response.status_code} - {response.text}")
                    return None
            
            if not all_data:
                logging.warning("No data received")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['price'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Select relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'price', 'volume']]
            
            logging.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching Binance data: {e}")
            return None
    
    def fetch_coingecko_data(self, coin_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from CoinGecko API (alternative source)
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }
            
            logging.info(f"Fetching CoinGecko data for {coin_id} from {start_date} to {end_date}")
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract price data
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if not prices:
                    logging.warning("No price data received from CoinGecko")
                    return None
                
                # Create DataFrame
                df = pd.DataFrame(prices, columns=['timestamp_ms', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                
                # Add volume data if available
                if volumes:
                    volume_df = pd.DataFrame(volumes, columns=['timestamp_ms', 'volume'])
                    df = df.merge(volume_df, on='timestamp_ms', how='left')
                else:
                    df['volume'] = 0
                
                df = df[['timestamp', 'price', 'volume']].sort_values('timestamp').reset_index(drop=True)
                
                logging.info(f"Successfully fetched {len(df)} data points from CoinGecko")
                return df
                
            else:
                logging.error(f"CoinGecko API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error fetching CoinGecko data: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, filename: str, data_dir: str = "data") -> str:
        """Save DataFrame to CSV file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False)
            
            logging.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            return None
    
    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from CSV file"""
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logging.info(f"Loaded {len(df)} data points from {filepath}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None
    
    def fetch_btc_2021_data(self, interval: str = '1h', use_backup: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch BTC data for the entire year 2021
        
        Args:
            interval: Data interval ('1m', '5m', '1h', '1d')
            use_backup: If True, try CoinGecko if Binance fails
        """
        try:
            # Try Binance first
            df = self.fetch_binance_klines(
                symbol='BTCUSDT',
                interval=interval,
                start_time='2021-01-01',
                end_time='2021-12-31'
            )
            
            if df is not None:
                filename = f"btc_2021_{interval}_binance.csv"
                self.save_data(df, filename)
                return df
            
            # If Binance fails and backup is enabled, try CoinGecko
            if use_backup:
                logging.info("Binance fetch failed, trying CoinGecko as backup...")
                df = self.fetch_coingecko_data(
                    coin_id='bitcoin',
                    start_date='2021-01-01',
                    end_date='2021-12-31'
                )
                
                if df is not None:
                    filename = f"btc_2021_daily_coingecko.csv"
                    self.save_data(df, filename)
                    return df
            
            logging.error("Failed to fetch BTC 2021 data from all sources")
            return None
            
        except Exception as e:
            logging.error(f"Error fetching BTC 2021 data: {e}")
            return None
    
    def fetch_btc_multi_year_data(self, start_year: int = 2021, end_year: int = 2025, 
                                  interval: str = '1h', use_backup: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch BTC data for multiple years (2021-2025)
        
        Args:
            start_year: Starting year (default: 2021)
            end_year: Ending year (default: 2025)
            interval: Data interval ('1m', '5m', '1h', '1d')
            use_backup: If True, try CoinGecko if Binance fails
        """
        try:
            from datetime import datetime
            
            # Get current date to avoid fetching future data
            current_date = datetime.now()
            actual_end_year = min(end_year, current_date.year)
            
            # If we're in the current year, use current date as end
            if actual_end_year == current_date.year:
                end_date = current_date.strftime('%Y-%m-%d')
            else:
                end_date = f"{actual_end_year}-12-31"
            
            start_date = f"{start_year}-01-01"
            
            logging.info(f"Fetching BTC data from {start_date} to {end_date}")
            
            # Try Binance first
            df = self.fetch_binance_klines(
                symbol='BTCUSDT',
                interval=interval,
                start_time=start_date,
                end_time=end_date
            )
            
            if df is not None:
                filename = f"btc_{start_year}_{actual_end_year}_{interval}_binance.csv"
                self.save_data(df, filename)
                logging.info(f"Successfully fetched {len(df)} records from {start_date} to {end_date}")
                return df
            
            # If Binance fails and backup is enabled, try CoinGecko
            if use_backup:
                logging.info("Binance fetch failed, trying CoinGecko as backup...")
                df = self.fetch_coingecko_data(
                    coin_id='bitcoin',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None:
                    filename = f"btc_{start_year}_{actual_end_year}_daily_coingecko.csv"
                    self.save_data(df, filename)
                    logging.info(f"Successfully fetched {len(df)} records from CoinGecko")
                    return df
            
            logging.error(f"Failed to fetch BTC {start_year}-{actual_end_year} data from all sources")
            return None
            
        except Exception as e:
            logging.error(f"Error fetching BTC multi-year data: {e}")
            return None
    
    def fetch_yearly_data_chunks(self, start_year: int = 2021, end_year: int = 2025, 
                                interval: str = '1h') -> Dict[int, pd.DataFrame]:
        """
        Fetch data year by year to manage large datasets and API limits
        
        Args:
            start_year: Starting year
            end_year: Ending year
            interval: Data interval
            
        Returns:
            Dictionary with year as key and DataFrame as value
        """
        yearly_data = {}
        
        try:
            from datetime import datetime
            current_year = datetime.now().year
            
            for year in range(start_year, min(end_year + 1, current_year + 1)):
                logging.info(f"Fetching data for year {year}...")
                
                # Check if we already have this year's data
                filename = f"btc_{year}_{year}_1h_binance.csv"
                filepath = f"data/{filename}"
                
                if os.path.exists(filepath):
                    logging.info(f"Loading existing data for {year}")
                    yearly_data[year] = self.load_data(filepath)
                    continue
                
                # Determine date range for this year
                start_date = f"{year}-01-01"
                if year == current_year:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    end_date = f"{year}-12-31"
                
                # Fetch data for this year
                df = self.fetch_binance_klines(
                    symbol='BTCUSDT',
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if df is not None:
                    yearly_data[year] = df
                    # Save individual year data
                    self.save_data(df, f"btc_{year}_{year}_{interval}_binance.csv")
                    logging.info(f"Successfully fetched {len(df)} records for {year}")
                    
                    # Add delay to respect API limits
                    time.sleep(1)
                else:
                    logging.warning(f"Failed to fetch data for {year}")
            
            return yearly_data
            
        except Exception as e:
            logging.error(f"Error fetching yearly data chunks: {e}")
            return {}
    
    def combine_yearly_data(self, yearly_data: Dict[int, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Combine multiple years of data into a single DataFrame
        
        Args:
            yearly_data: Dictionary with year as key and DataFrame as value
            
        Returns:
            Combined DataFrame sorted by timestamp
        """
        try:
            if not yearly_data:
                logging.error("No yearly data to combine")
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(yearly_data.values(), ignore_index=True)
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any duplicate timestamps
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            logging.info(f"Combined data: {len(combined_df)} total records from {len(yearly_data)} years")
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error combining yearly data: {e}")
            return None
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the data"""
        try:
            summary = {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'price_statistics': {
                    'min': df['price'].min(),
                    'max': df['price'].max(),
                    'mean': df['price'].mean(),
                    'std': df['price'].std()
                }
            }
            
            if 'volume' in df.columns:
                summary['volume_statistics'] = {
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'mean': df['volume'].mean(),
                    'std': df['volume'].std()
                }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating data summary: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    fetcher = HistoricalDataFetcher()
    
    # Fetch BTC data for 2021
    logging.info("Fetching BTC data for 2021...")
    btc_data = fetcher.fetch_btc_2021_data(interval='1h')
    
    if btc_data is not None:
        # Print summary
        summary = fetcher.get_data_summary(btc_data)
        print("\n=== BTC 2021 Data Summary ===")
        print(json.dumps(summary, indent=2, default=str))
        
        # Display first few rows
        print("\n=== Sample Data ===")
        print(btc_data.head())
        print(f"\nTotal records: {len(btc_data)}")
    else:
        logging.error("Failed to fetch BTC 2021 data")