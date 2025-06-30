import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VIPVolumeTracker:
    """Track 30-day trading volume for VIP 1 status (1,000,000 USDT target)"""
    
    def __init__(self, data_file: str = "volume_data.json"):
        self.data_file = data_file
        self.daily_volumes: Dict[str, float] = {}
        self.target_monthly_volume = 1_000_000  # USDT for VIP 1
        self.target_daily_volume = 33_334  # ~1M/30 days
        self.load_data()
    
    def load_data(self):
        """Load volume data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.daily_volumes = data.get('daily_volumes', {})
                logging.info(f"Loaded volume data from {self.data_file}")
            else:
                logging.info("No existing volume data found, starting fresh")
        except Exception as e:
            logging.error(f"Error loading volume data: {e}")
            self.daily_volumes = {}
    
    def save_data(self):
        """Save volume data to file"""
        try:
            data = {
                'daily_volumes': self.daily_volumes,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.debug(f"Saved volume data to {self.data_file}")
        except Exception as e:
            logging.error(f"Error saving volume data: {e}")
    
    def track_trade_volume(self, trade_volume_usdt: float, symbol: str = "BTCUSDT"):
        """Track each trade's volume contribution"""
        try:
            today = datetime.now().date().isoformat()
            if today not in self.daily_volumes:
                self.daily_volumes[today] = 0
            
            self.daily_volumes[today] += trade_volume_usdt
            
            # Clean up old data (keep only last 35 days for buffer)
            cutoff_date = (datetime.now().date() - timedelta(days=35)).isoformat()
            self.daily_volumes = {
                date: volume for date, volume in self.daily_volumes.items() 
                if date >= cutoff_date
            }
            
            self.save_data()
            
            logging.info(f"Tracked {trade_volume_usdt:.2f} USDT volume for {symbol}")
            logging.info(f"Today's volume: {self.daily_volumes[today]:.2f} USDT")
            
            # Check progress
            monthly_volume = self.get_30_day_volume()
            progress = (monthly_volume / self.target_monthly_volume) * 100
            logging.info(f"30-day volume: {monthly_volume:.2f} USDT ({progress:.1f}% of VIP 1 target)")
            
        except Exception as e:
            logging.error(f"Error tracking trade volume: {e}")
    
    def get_30_day_volume(self) -> float:
        """Calculate rolling 30-day volume"""
        try:
            thirty_days_ago = (datetime.now().date() - timedelta(days=30)).isoformat()
            total_volume = sum(
                volume for date, volume in self.daily_volumes.items() 
                if date >= thirty_days_ago
            )
            return total_volume
        except Exception as e:
            logging.error(f"Error calculating 30-day volume: {e}")
            return 0.0
    
    def get_daily_volume(self, date: Optional[str] = None) -> float:
        """Get volume for a specific date (default: today)"""
        if date is None:
            date = datetime.now().date().isoformat()
        return self.daily_volumes.get(date, 0.0)
    
    def is_vip1_qualified(self) -> bool:
        """Check if VIP 1 requirements are met"""
        return self.get_30_day_volume() >= self.target_monthly_volume
    
    def get_vip_status_report(self) -> Dict:
        """Generate comprehensive VIP status report"""
        monthly_volume = self.get_30_day_volume()
        daily_volume = self.get_daily_volume()
        
        days_remaining = 30 - len([
            d for d in self.daily_volumes.keys() 
            if d >= (datetime.now().date() - timedelta(days=30)).isoformat()
        ])
        
        volume_needed = max(0, self.target_monthly_volume - monthly_volume)
        daily_volume_needed = volume_needed / max(1, days_remaining) if days_remaining > 0 else 0
        
        return {
            'current_30_day_volume': monthly_volume,
            'target_volume': self.target_monthly_volume,
            'progress_percentage': (monthly_volume / self.target_monthly_volume) * 100,
            'volume_needed': volume_needed,
            'daily_volume_today': daily_volume,
            'target_daily_volume': self.target_daily_volume,
            'daily_volume_needed': daily_volume_needed,
            'days_remaining_in_period': days_remaining,
            'is_vip1_qualified': self.is_vip1_qualified(),
            'on_track_for_vip1': daily_volume >= self.target_daily_volume
        }
    
    def log_vip_status(self):
        """Log current VIP status"""
        report = self.get_vip_status_report()
        
        logging.info("=== VIP VOLUME TRACKING REPORT ===")
        logging.info(f"30-day volume: {report['current_30_day_volume']:.2f} USDT")
        logging.info(f"VIP 1 target: {report['target_volume']:,.0f} USDT")
        logging.info(f"Progress: {report['progress_percentage']:.1f}%")
        logging.info(f"Volume needed: {report['volume_needed']:.2f} USDT")
        logging.info(f"Today's volume: {report['daily_volume_today']:.2f} USDT")
        logging.info(f"Daily target: {report['target_daily_volume']:,.0f} USDT")
        logging.info(f"VIP 1 qualified: {'YES' if report['is_vip1_qualified'] else 'NO'}")
        logging.info(f"On track: {'YES' if report['on_track_for_vip1'] else 'NO'}")
        logging.info("================================")

if __name__ == "__main__":
    # Test the VIP volume tracker
    tracker = VIPVolumeTracker()
    
    # Simulate some trades
    tracker.track_trade_volume(5000, "BTCUSDT")  # 5000 USDT trade
    tracker.track_trade_volume(3000, "BTCUSDT")  # 3000 USDT trade
    
    # Display status
    tracker.log_vip_status()