import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import numpy as np

class TestPlotter:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.plot_dir = Path("results/test_plot")
        # Create base plot directory if it doesn't exist
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        # Create timestamped subdirectory
        self.timestamp_dir = self.plot_dir / self.timestamp
        self.timestamp_dir.mkdir(exist_ok=True)
        
    def plot_results(self, results: dict):
        """Plot results for each symbol and save to separate files"""
        for symbol, data in results.items():
            self._create_symbol_plot(symbol, data)
    
    def _create_symbol_plot(self, symbol: str, data: dict):
        """Create and save plot for a single symbol"""
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: Account Value Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_account_value(ax1, data)
        
        # Plot 2: Price and Positions
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_price_and_positions(ax2, data)
        
        # Plot 3: Actions Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_actions_distribution(ax3, data)
        
        # Plot 4: Cumulative Returns Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_cumulative_returns(ax4, data)
        
        # Add title with performance metrics
        plt.suptitle(f"Test Results for {symbol}\n" + 
                    f"Strategy Return: {data['returns']:.2%}, " +
                    f"Stock Return: {data['stock_returns']:.2%}\n" +
                    f"Utilization Rate: {data['utilization_rate']:.2%}, " +
                    f"Adjusted Outperformance: {data['utilization_adjusted_outperformance']:.2%}",
                    fontsize=12)
        
        # Save plot
        plot_path = self.timestamp_dir / f"{symbol}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def _plot_account_value(self, ax, data):
        """Plot account value over time"""
        ax.plot(data['dates'], data['account_values'], label='Account Value')
        ax.set_title('Account Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.grid(True)
        ax.legend()
        
    def _plot_price_and_positions(self, ax, data):
        """Plot price and positions"""
        # Plot price
        ax.plot(data['dates'], data['prices'], label='Price', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)', color='blue')
        ax.grid(True)
        
        # Plot positions on secondary y-axis
        ax2 = ax.twinx()
        # convert positions to percentage of account value
        data['positions'] = [x * 100 for x in data['positions']]
        ax2.plot(data['dates'], data['positions'], label=r'cash as % of account value', color='red', alpha=0.5)
        ax2.set_ylabel(r'cash as % of account value', color='red')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.set_title('Price and Positions')
        
    def _plot_actions_distribution(self, ax, data):
        """Plot distribution of actions"""
        actions = np.array(data['actions'])
        
        # Create a mapping for all possible actions (0: Hold, 1: Buy, 2: Sell)
        action_mapping = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        counts = np.zeros(3)  # Initialize counts for all three actions
        
        # Count occurrences of each action
        unique_actions, action_counts = np.unique(actions, return_counts=True)
        for action, count in zip(unique_actions, action_counts):
            counts[action] = count
        
        # Use action labels in the correct order (matching the action values)
        action_labels = ['Hold', 'Buy', 'Sell']
        ax.bar(action_labels, counts)
        ax.set_title('Action Distribution')
        ax.set_ylabel('Count')
        
        # Add count labels on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count, f'{int(count)}', ha='center', va='bottom')
            
    def _plot_cumulative_returns(self, ax, data):
        """Plot cumulative returns comparison"""
        # Calculate cumulative returns
        initial_value = data['account_values'][0]
        cum_strategy_returns = [(v - initial_value) / initial_value for v in data['account_values']]
        
        initial_price = data['prices'][0]
        cum_stock_returns = [(p - initial_price) / initial_price for p in data['prices']]
        
        # Plot both series
        ax.plot(data['dates'], cum_strategy_returns, label='Strategy Returns')
        ax.plot(data['dates'], cum_stock_returns, label='Stock Returns')
        ax.set_title('Cumulative Returns Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.grid(True)
        ax.legend() 