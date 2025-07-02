#!/usr/bin/env python3
"""
Supertrend Enhanced Market Regime Detection Backtest
Compares MA-only vs MA+Supertrend regime detection performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicators for backtesting"""
    
    @staticmethod
    def ema(data: np.ndarray, window: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2.0 / (window + 1)
        result = np.empty_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = 0
        tr3[0] = 0
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return TechnicalIndicators.ema(true_range, period)
    
    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Supertrend indicator"""
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = np.full_like(close, np.nan)
        trend_direction = np.full_like(close, 1)
        
        for i in range(1, len(close)):
            if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
                continue
                
            # Calculate final bands
            if np.isnan(upper_band[i-1]):
                final_upper_band = upper_band[i]
            else:
                final_upper_band = upper_band[i] if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1] else upper_band[i-1]
            
            if np.isnan(lower_band[i-1]):
                final_lower_band = lower_band[i]
            else:
                final_lower_band = lower_band[i] if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1] else lower_band[i-1]
            
            # Determine trend
            if i == 1:
                if close[i] <= final_lower_band:
                    supertrend[i] = final_upper_band
                    trend_direction[i] = -1
                else:
                    supertrend[i] = final_lower_band
                    trend_direction[i] = 1
            else:
                if supertrend[i-1] == lower_band[i-1] and close[i] <= final_lower_band:
                    supertrend[i] = final_upper_band
                    trend_direction[i] = -1
                elif supertrend[i-1] == upper_band[i-1] and close[i] >= final_upper_band:
                    supertrend[i] = final_lower_band
                    trend_direction[i] = 1
                elif supertrend[i-1] == lower_band[i-1] and close[i] > final_lower_band:
                    supertrend[i] = final_lower_band
                    trend_direction[i] = 1
                elif supertrend[i-1] == upper_band[i-1] and close[i] < final_upper_band:
                    supertrend[i] = final_upper_band
                    trend_direction[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
                    trend_direction[i] = trend_direction[i-1]
        
        return supertrend, trend_direction

class MarketRegimeDetector:
    """Market regime detection with multiple methods"""
    
    def __init__(self, ma_fast: int = 50, ma_slow: int = 200):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.indicators = TechnicalIndicators()
    
    def detect_ma_only_regime(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[List[str], List[float]]:
        """Original MA-only regime detection"""
        ma_fast = self.indicators.ema(close, self.ma_fast)
        ma_slow = self.indicators.ema(close, self.ma_slow)
        
        regimes = []
        confidences = []
        
        for i in range(len(close)):
            if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
                regimes.append('unknown')
                confidences.append(0.1)
                continue
            
            # Simple MA-based regime detection
            if ma_fast[i] > ma_slow[i] and close[i] > ma_fast[i]:
                regimes.append('bull')
                confidence = min(0.9, 0.6 + abs(ma_fast[i] - ma_slow[i]) / close[i] * 10)
            elif ma_fast[i] < ma_slow[i] and close[i] < ma_fast[i]:
                regimes.append('bear')
                confidence = min(0.9, 0.6 + abs(ma_fast[i] - ma_slow[i]) / close[i] * 10)
            else:
                regimes.append('sideways')
                confidence = 0.5
            
            confidences.append(confidence)
        
        return regimes, confidences
    
    def detect_ma_supertrend_regime(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[List[str], List[float]]:
        """Enhanced MA + Supertrend regime detection"""
        ma_fast = self.indicators.ema(close, self.ma_fast)
        ma_slow = self.indicators.ema(close, self.ma_slow)
        supertrend, st_direction = self.indicators.supertrend(high, low, close)
        
        regimes = []
        confidences = []
        
        for i in range(len(close)):
            if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or np.isnan(st_direction[i]):
                regimes.append('unknown')
                confidences.append(0.1)
                continue
            
            # MA signals
            ma_bullish = ma_fast[i] > ma_slow[i]
            price_above_fast = close[i] > ma_fast[i]
            price_above_slow = close[i] > ma_slow[i]
            
            # Supertrend signals
            st_bullish = st_direction[i] == 1
            price_above_st = close[i] > supertrend[i] if not np.isnan(supertrend[i]) else False
            
            # Combined signal analysis
            bullish_signals = sum([ma_bullish, price_above_fast, price_above_slow, st_bullish, price_above_st])
            
            # Enhanced regime detection
            if bullish_signals >= 4 and ma_bullish and st_bullish:
                regimes.append('bull')
                confidence = 0.8 + min(0.15, bullish_signals * 0.03)
            elif bullish_signals <= 1 and not ma_bullish and not st_bullish:
                regimes.append('bear')
                confidence = 0.8 + min(0.15, (5 - bullish_signals) * 0.03)
            elif abs(ma_fast[i] - ma_slow[i]) / close[i] < 0.02:  # Small MA separation
                regimes.append('sideways')
                confidence = 0.7
            else:
                # Mixed signals - use dominant
                if bullish_signals >= 3:
                    regimes.append('bull')
                    confidence = 0.4 + bullish_signals * 0.1
                elif bullish_signals <= 2:
                    regimes.append('bear')
                    confidence = 0.4 + (5 - bullish_signals) * 0.1
                else:
                    regimes.append('sideways')
                    confidence = 0.3
            
            # Agreement bonus
            if ma_bullish == st_bullish and regimes[-1] in ['bull', 'bear']:
                confidence += 0.1
            
            confidences.append(min(0.95, confidence))
        
        return regimes, confidences

class GridTradingSimulator:
    """Simulate grid trading based on regime detection"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.btc_position = 0.0
        self.trades = []
        self.grid_spacing = 0.005  # 0.5%
        self.position_size = 0.001  # BTC per trade
        
    def reset(self):
        """Reset simulator state"""
        self.current_balance = self.initial_balance
        self.btc_position = 0.0
        self.trades = []
    
    def simulate_trading(self, prices: np.ndarray, regimes: List[str], confidences: List[float]) -> Dict[str, Any]:
        """Simulate grid trading based on regime detection"""
        portfolio_values = []
        
        for i, (price, regime, confidence) in enumerate(zip(prices, regimes, confidences)):
            # Calculate current portfolio value
            portfolio_value = self.current_balance + self.btc_position * price
            portfolio_values.append(portfolio_value)
            
            # Skip if not enough confidence or unknown regime
            if confidence < 0.6 or regime == 'unknown':
                continue
            
            # Grid trading logic based on regime
            if regime == 'bull' and confidence > 0.7:
                # More aggressive buying in bull markets
                if self.current_balance > price * self.position_size * 1.5:
                    self._execute_buy(price, self.position_size * 1.5)
                    
            elif regime == 'bear' and confidence > 0.7:
                # More aggressive selling in bear markets
                if self.btc_position > self.position_size * 1.5:
                    self._execute_sell(price, self.position_size * 1.5)
                    
            elif regime == 'sideways' and confidence > 0.6:
                # Standard grid trading in sideways markets
                # Buy low, sell high within the range
                if i > 50:  # Need some history
                    recent_low = np.min(prices[max(0, i-20):i])
                    recent_high = np.max(prices[max(0, i-20):i])
                    mid_price = (recent_low + recent_high) / 2
                    
                    if price < mid_price * 0.98 and self.current_balance > price * self.position_size:
                        self._execute_buy(price, self.position_size)
                    elif price > mid_price * 1.02 and self.btc_position > self.position_size:
                        self._execute_sell(price, self.position_size)
        
        final_value = self.current_balance + self.btc_position * prices[-1]
        
        return {
            'portfolio_values': portfolio_values,
            'final_value': final_value,
            'total_return': (final_value - self.initial_balance) / self.initial_balance,
            'total_trades': len(self.trades),
            'trades': self.trades
        }
    
    def _execute_buy(self, price: float, quantity: float):
        """Execute buy order"""
        cost = price * quantity * 1.001  # 0.1% fee
        if self.current_balance >= cost:
            self.current_balance -= cost
            self.btc_position += quantity
            self.trades.append({
                'type': 'buy',
                'price': price,
                'quantity': quantity,
                'cost': cost
            })
    
    def _execute_sell(self, price: float, quantity: float):
        """Execute sell order"""
        if self.btc_position >= quantity:
            proceeds = price * quantity * 0.999  # 0.1% fee
            self.current_balance += proceeds
            self.btc_position -= quantity
            self.trades.append({
                'type': 'sell',
                'price': price,
                'quantity': quantity,
                'proceeds': proceeds
            })

class SupertrendBacktester:
    """Main backtesting class"""
    
    def __init__(self):
        self.detector = MarketRegimeDetector()
        self.simulator = GridTradingSimulator()
        
    def load_data(self) -> pd.DataFrame:
        """Load historical data"""
        try:
            df = pd.read_csv('data/btc_2021_2025_1h_combined.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Filter for 2022-2025 June
            start_date = '2022-01-01'
            end_date = '2025-06-30'
            df = df[start_date:end_date]
            
            logger.info(f"Loaded {len(df)} data points from {start_date} to {end_date}")
            return df
            
        except FileNotFoundError:
            logger.info("Creating synthetic data for demonstration")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic BTC data"""
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2025, 6, 30)
        timestamps = pd.date_range(start_date, end_date, freq='H')
        
        np.random.seed(42)
        n_points = len(timestamps)
        
        # Generate realistic price path
        initial_price = 47000
        returns = np.random.normal(0, 0.02, n_points)
        
        # Add trend and volatility clustering
        for i in range(1, n_points):
            returns[i] += -0.001 * returns[i-1]  # Mean reversion
            if abs(returns[i-1]) > 0.03:
                returns[i] *= 1.5  # Volatility clustering
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_points)
        }, index=timestamps)
        
        # Ensure OHLC consistency
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        
        return df
    
    def run_comparison_backtest(self) -> Dict[str, Any]:
        """Run comparison backtest between MA-only and MA+Supertrend"""
        logger.info("🚀 Starting Supertrend Enhanced Backtest Comparison")
        logger.info("="*60)
        
        # Load data
        df = self.load_data()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Detect regimes with both methods
        logger.info("Detecting regimes with MA-only method...")
        ma_regimes, ma_confidences = self.detector.detect_ma_only_regime(high, low, close)
        
        logger.info("Detecting regimes with MA+Supertrend method...")
        enhanced_regimes, enhanced_confidences = self.detector.detect_ma_supertrend_regime(high, low, close)
        
        # Simulate trading with both methods
        logger.info("Simulating trading with MA-only method...")
        self.simulator.reset()
        ma_results = self.simulator.simulate_trading(close, ma_regimes, ma_confidences)
        
        logger.info("Simulating trading with MA+Supertrend method...")
        self.simulator.reset()
        enhanced_results = self.simulator.simulate_trading(close, enhanced_regimes, enhanced_confidences)
        
        # Calculate metrics
        ma_metrics = self._calculate_metrics(ma_results, close)
        enhanced_metrics = self._calculate_metrics(enhanced_results, close)
        
        # Create comparison results
        comparison = {
            'data': df,
            'ma_only': {
                'regimes': ma_regimes,
                'confidences': ma_confidences,
                'results': ma_results,
                'metrics': ma_metrics
            },
            'ma_supertrend': {
                'regimes': enhanced_regimes,
                'confidences': enhanced_confidences,
                'results': enhanced_results,
                'metrics': enhanced_metrics
            }
        }
        
        # Print results
        self._print_comparison_results(comparison)
        
        # Create visualizations
        self._create_comparison_charts(comparison)
        
        return comparison
    
    def _calculate_metrics(self, results: Dict[str, Any], prices: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        portfolio_values = results['portfolio_values']
        
        if len(portfolio_values) < 2:
            return {}
        
        # Calculate returns
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        daily_returns = returns[::24] if len(returns) >= 24 else returns  # Assuming hourly data
        
        # Performance metrics
        total_return = results['total_return']
        days = len(portfolio_values) / 24  # Convert hours to days
        annual_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        volatility = np.std(daily_returns) * np.sqrt(365) if len(daily_returns) > 1 else 0
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # BTC comparison
        btc_return = (prices[-1] - prices[0]) / prices[0]
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'total_trades': results['total_trades'],
            'btc_return': btc_return,
            'excess_return': total_return - btc_return
        }
    
    def _print_comparison_results(self, comparison: Dict[str, Any]):
        """Print comparison results"""
        ma_metrics = comparison['ma_only']['metrics']
        enhanced_metrics = comparison['ma_supertrend']['metrics']
        
        print("\n" + "="*60)
        print("🔥 SUPERTREND ENHANCEMENT BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<25} {'MA Only':<15} {'MA+Supertrend':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics_to_compare = [
            ('Total Return', 'total_return', '%'),
            ('Annual Return', 'annual_return', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Total Trades', 'total_trades', ''),
            ('Excess vs BTC', 'excess_return', '%')
        ]
        
        for name, key, unit in metrics_to_compare:
            ma_val = ma_metrics.get(key, 0)
            enhanced_val = enhanced_metrics.get(key, 0)
            
            if unit == '%':
                ma_str = f"{ma_val:.1%}"
                enhanced_str = f"{enhanced_val:.1%}"
                improvement = enhanced_val - ma_val
                imp_str = f"{improvement:+.1%}"
            else:
                ma_str = f"{ma_val:.2f}"
                enhanced_str = f"{enhanced_val:.2f}"
                improvement = enhanced_val - ma_val
                imp_str = f"{improvement:+.2f}"
            
            print(f"{name:<25} {ma_str:<15} {enhanced_str:<15} {imp_str:<15}")
        
        # Performance summary
        print(f"\n🏆 ENHANCEMENT IMPACT:")
        return_improvement = enhanced_metrics.get('total_return', 0) - ma_metrics.get('total_return', 0)
        sharpe_improvement = enhanced_metrics.get('sharpe_ratio', 0) - ma_metrics.get('sharpe_ratio', 0)
        
        if return_improvement > 0.05:  # 5% improvement
            print("✅ SIGNIFICANT IMPROVEMENT in returns")
        elif return_improvement > 0:
            print("✅ POSITIVE IMPROVEMENT in returns")
        else:
            print("❌ No improvement in returns")
        
        if sharpe_improvement > 0.2:
            print("✅ SIGNIFICANT IMPROVEMENT in risk-adjusted returns")
        elif sharpe_improvement > 0:
            print("✅ POSITIVE IMPROVEMENT in risk-adjusted returns")
        else:
            print("❌ No improvement in risk-adjusted returns")
        
        print("\n" + "="*60)
    
    def _create_comparison_charts(self, comparison: Dict[str, Any]):
        """Create comparison visualization charts"""
        df = comparison['data']
        ma_results = comparison['ma_only']['results']
        enhanced_results = comparison['ma_supertrend']['results']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Portfolio Performance Comparison
        ax1 = axes[0, 0]
        ax1.plot(ma_results['portfolio_values'], label='MA Only', alpha=0.8)
        ax1.plot(enhanced_results['portfolio_values'], label='MA + Supertrend', alpha=0.8)
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_title('Portfolio Performance Comparison', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Comparison
        ax2 = axes[0, 1]
        ma_returns = [(v - 100000) / 100000 * 100 for v in ma_results['portfolio_values']]
        enhanced_returns = [(v - 100000) / 100000 * 100 for v in enhanced_results['portfolio_values']]
        
        ax2.plot(ma_returns, label='MA Only', alpha=0.8)
        ax2.plot(enhanced_returns, label='MA + Supertrend', alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Returns Comparison (%)', fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime Detection Accuracy
        ax3 = axes[1, 0]
        ma_regimes = comparison['ma_only']['regimes']
        enhanced_regimes = comparison['ma_supertrend']['regimes']
        
        # Count regime transitions
        ma_transitions = sum(1 for i in range(1, len(ma_regimes)) if ma_regimes[i] != ma_regimes[i-1])
        enhanced_transitions = sum(1 for i in range(1, len(enhanced_regimes)) if enhanced_regimes[i] != enhanced_regimes[i-1])
        
        ax3.bar(['MA Only', 'MA + Supertrend'], [ma_transitions, enhanced_transitions], 
               color=['blue', 'green'], alpha=0.7)
        ax3.set_title('Regime Transitions Count', fontweight='bold')
        ax3.set_ylabel('Number of Transitions')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence Distribution
        ax4 = axes[1, 1]
        ma_confidences = comparison['ma_only']['confidences']
        enhanced_confidences = comparison['ma_supertrend']['confidences']
        
        ax4.hist(ma_confidences, bins=20, alpha=0.6, label='MA Only', density=True)
        ax4.hist(enhanced_confidences, bins=20, alpha=0.6, label='MA + Supertrend', density=True)
        ax4.set_title('Confidence Distribution', fontweight='bold')
        ax4.set_xlabel('Confidence Level')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('supertrend_enhancement_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comparison charts saved as 'supertrend_enhancement_comparison.png'")

def main():
    """Run the Supertrend enhancement backtest"""
    backtester = SupertrendBacktester()
    results = backtester.run_comparison_backtest()
    return results

if __name__ == "__main__":
    results = main()