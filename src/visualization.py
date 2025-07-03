import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional
import logging

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestVisualizer:
    """
    Create comprehensive visualizations for backtest results
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_price_chart(self, price_data: pd.DataFrame, title: str = "BTC Price Chart 2021", 
                        save_path: str = None) -> str:
        """Plot BTC price chart with key statistics"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
            
            # Price chart
            ax1.plot(price_data['timestamp'], price_data['price'], linewidth=1, color='#1f77b4')
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add min/max annotations
            min_idx = price_data['price'].idxmin()
            max_idx = price_data['price'].idxmax()
            
            min_price = price_data.loc[min_idx, 'price']
            max_price = price_data.loc[max_idx, 'price']
            min_date = price_data.loc[min_idx, 'timestamp']
            max_date = price_data.loc[max_idx, 'timestamp']
            
            ax1.annotate(f'Min: ${min_price:,.0f}', 
                        xy=(min_date, min_price), xytext=(10, 20),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax1.annotate(f'Max: ${max_price:,.0f}', 
                        xy=(max_date, max_price), xytext=(10, -30),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Volume chart
            if 'volume' in price_data.columns:
                ax2.bar(price_data['timestamp'], price_data['volume'], alpha=0.6, color='gray')
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = f"charts/btc_price_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Price chart saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating price chart: {e}")
            return None
    
    def plot_strategy_comparison(self, results: Dict[str, Any], save_path: str = None) -> str:
        """Create comprehensive strategy comparison charts"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract data
            strategies = []
            returns = []
            drawdowns = []
            sharpe_ratios = []
            win_rates = []
            
            for name, data in results.items():
                if 'result' in data:
                    result = data['result']
                    strategies.append(name.replace('_', '\n'))
                    returns.append(result.total_return_pct)
                    drawdowns.append(result.max_drawdown_pct)
                    sharpe_ratios.append(result.sharpe_ratio)
                    win_rates.append(result.win_rate)
            
            # 1. Total Returns
            bars1 = ax1.bar(strategies, returns, color=self.colors[:len(strategies)])
            ax1.set_title('Total Returns by Strategy', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Return (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # 2. Max Drawdowns
            bars2 = ax2.bar(strategies, drawdowns, color='red', alpha=0.7)
            ax2.set_title('Maximum Drawdowns by Strategy', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # 3. Sharpe Ratios
            bars3 = ax3.bar(strategies, sharpe_ratios, color='green', alpha=0.7)
            ax3.set_title('Sharpe Ratios by Strategy', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Sharpe Ratio', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Win Rates
            bars4 = ax4.bar(strategies, win_rates, color='purple', alpha=0.7)
            ax4.set_title('Win Rates by Strategy', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Win Rate (%)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, bar in enumerate(bars4):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = f"charts/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Strategy comparison chart saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating strategy comparison chart: {e}")
            return None
    
    def plot_risk_return_scatter(self, results: Dict[str, Any], save_path: str = None) -> str:
        """Create risk-return scatter plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data
            returns = []
            risks = []
            labels = []
            
            for name, data in results.items():
                if 'result' in data:
                    result = data['result']
                    returns.append(result.total_return_pct)
                    risks.append(result.max_drawdown_pct)
                    labels.append(name)
            
            # Create scatter plot
            scatter = ax.scatter(risks, returns, s=100, alpha=0.7, 
                               c=range(len(returns)), cmap='viridis')
            
            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(label.replace('_', '\n'), (risks[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Add quadrant lines
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.mean(risks), color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
            ax.set_ylabel('Total Return (%)', fontsize=12)
            ax.set_title('Risk vs Return Analysis', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add quadrant labels
            ax.text(0.02, 0.98, 'Low Risk\nHigh Return', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                   verticalalignment='top')
            
            ax.text(0.98, 0.02, 'High Risk\nLow Return', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7),
                   verticalalignment='bottom', horizontalalignment='right')
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = f"charts/risk_return_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Risk-return scatter plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating risk-return scatter plot: {e}")
            return None
    
    def plot_monthly_performance(self, price_data: pd.DataFrame, save_path: str = None) -> str:
        """Plot monthly performance analysis"""
        try:
            # Calculate monthly returns
            df = price_data.copy()
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_data = df.groupby('month')['price'].agg(['first', 'last']).reset_index()
            monthly_data['return_pct'] = ((monthly_data['last'] - monthly_data['first']) / monthly_data['first']) * 100
            monthly_data['month_str'] = monthly_data['month'].astype(str)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Monthly returns bar chart
            colors = ['green' if x > 0 else 'red' for x in monthly_data['return_pct']]
            bars = ax1.bar(monthly_data['month_str'], monthly_data['return_pct'], 
                          color=colors, alpha=0.7)
            
            ax1.set_title('Monthly Returns - BTC 2021', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Return (%)', fontsize=12)
            ax1.set_xlabel('Month', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                        f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=9)
            
            # Cumulative returns
            monthly_data['cumulative_return'] = (1 + monthly_data['return_pct']/100).cumprod() - 1
            ax2.plot(monthly_data['month_str'], monthly_data['cumulative_return'] * 100, 
                    marker='o', linewidth=2, markersize=6)
            
            ax2.set_title('Cumulative Returns - BTC 2021', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax2.set_xlabel('Month', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = f"charts/monthly_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Monthly performance chart saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating monthly performance chart: {e}")
            return None
    
    def create_performance_dashboard(self, price_data: pd.DataFrame, results: Dict[str, Any], 
                                   save_path: str = None) -> str:
        """Create a comprehensive performance dashboard"""
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 3, height_ratios=[2, 1.5, 1.5, 1.5], width_ratios=[2, 1, 1])
            
            # 1. Main price chart (top row, spans 3 columns)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(price_data['timestamp'], price_data['price'], linewidth=1.5, color='#1f77b4')
            ax1.set_title('BTC Price Chart 2021', fontsize=18, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add price statistics
            start_price = price_data.iloc[0]['price']
            end_price = price_data.iloc[-1]['price']
            yearly_return = ((end_price - start_price) / start_price) * 100
            
            ax1.text(0.02, 0.98, f'Start: ${start_price:,.0f}\nEnd: ${end_price:,.0f}\nReturn: {yearly_return:.1f}%', 
                    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))
            
            # 2. Strategy returns comparison (2nd row, left)
            ax2 = fig.add_subplot(gs[1, 0])
            strategies = []
            returns = []
            for name, data in results.items():
                if 'result' in data:
                    strategies.append(name.replace('_', '\n'))
                    returns.append(data['result'].total_return_pct)
            
            bars = ax2.bar(strategies, returns, color=self.colors[:len(strategies)])
            ax2.set_title('Strategy Returns', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Return (%)', fontsize=10)
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Risk metrics (2nd row, middle)
            ax3 = fig.add_subplot(gs[1, 1])
            drawdowns = [data['result'].max_drawdown_pct for name, data in results.items() if 'result' in data]
            strategy_names = [name.replace('_', '\n') for name in results.keys() if 'result' in results[name]]
            
            ax3.bar(strategy_names, drawdowns, color='red', alpha=0.7)
            ax3.set_title('Max Drawdowns', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Drawdown (%)', fontsize=10)
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Sharpe ratios (2nd row, right)
            ax4 = fig.add_subplot(gs[1, 2])
            sharpe_ratios = [data['result'].sharpe_ratio for name, data in results.items() if 'result' in data]
            
            ax4.bar(strategy_names, sharpe_ratios, color='green', alpha=0.7)
            ax4.set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Sharpe Ratio', fontsize=10)
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 5. Risk-Return scatter (3rd row, left)
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.scatter(drawdowns, returns, s=80, alpha=0.7, c=range(len(returns)), cmap='viridis')
            ax5.set_xlabel('Max Drawdown (%)', fontsize=10)
            ax5.set_ylabel('Total Return (%)', fontsize=10)
            ax5.set_title('Risk vs Return', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # 6. Win rates (3rd row, middle)
            ax6 = fig.add_subplot(gs[2, 1])
            win_rates = [data['result'].win_rate for name, data in results.items() if 'result' in data]
            ax6.bar(strategy_names, win_rates, color='purple', alpha=0.7)
            ax6.set_title('Win Rates', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Win Rate (%)', fontsize=10)
            ax6.tick_params(axis='x', rotation=45, labelsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. Summary statistics (3rd row, right)
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.axis('off')
            
            # Calculate summary stats
            avg_return = np.mean(returns)
            avg_drawdown = np.mean(drawdowns)
            avg_sharpe = np.mean(sharpe_ratios)
            best_strategy = strategy_names[np.argmax(returns)]
            
            summary_text = f"""SUMMARY STATISTICS
            
Strategies Tested: {len(results)}
Average Return: {avg_return:.1f}%
Average Drawdown: {avg_drawdown:.1f}%
Average Sharpe: {avg_sharpe:.2f}

Best Strategy: {best_strategy}
Best Return: {max(returns):.1f}%
Lowest Drawdown: {min(drawdowns):.1f}%
Best Sharpe: {max(sharpe_ratios):.2f}"""
            
            ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))
            
            # 8. Monthly performance (bottom row, spans all columns)
            ax8 = fig.add_subplot(gs[3, :])
            df = price_data.copy()
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_data = df.groupby('month')['price'].agg(['first', 'last']).reset_index()
            monthly_data['return_pct'] = ((monthly_data['last'] - monthly_data['first']) / monthly_data['first']) * 100
            monthly_data['month_str'] = monthly_data['month'].astype(str)
            
            colors = ['green' if x > 0 else 'red' for x in monthly_data['return_pct']]
            ax8.bar(monthly_data['month_str'], monthly_data['return_pct'], color=colors, alpha=0.7)
            ax8.set_title('Monthly BTC Returns 2021', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Return (%)', fontsize=10)
            ax8.set_xlabel('Month', fontsize=10)
            ax8.grid(True, alpha=0.3)
            ax8.tick_params(axis='x', rotation=45)
            
            plt.suptitle('BTC 2021 Backtest Performance Dashboard', fontsize=24, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save plot
            if save_path is None:
                save_path = f"charts/performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Performance dashboard saved to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating performance dashboard: {e}")
            return None
    
    def generate_all_charts(self, price_data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all charts and return file paths"""
        chart_paths = {}
        
        try:
            # Create charts directory
            os.makedirs('charts', exist_ok=True)
            
            # Generate all charts
            chart_paths['price_chart'] = self.plot_price_chart(price_data)
            chart_paths['strategy_comparison'] = self.plot_strategy_comparison(results)
            chart_paths['risk_return_scatter'] = self.plot_risk_return_scatter(results)
            chart_paths['monthly_performance'] = self.plot_monthly_performance(price_data)
            chart_paths['performance_dashboard'] = self.create_performance_dashboard(price_data, results)
            
            # Remove None values
            chart_paths = {k: v for k, v in chart_paths.items() if v is not None}
            
            logging.info(f"Generated {len(chart_paths)} charts successfully")
            return chart_paths
            
        except Exception as e:
            logging.error(f"Error generating charts: {e}")
            return chart_paths

if __name__ == "__main__":
    # Example usage - this would normally be called from the main backtest script
    visualizer = BacktestVisualizer()
    print("Backtest visualizer initialized. Use with backtest results.")