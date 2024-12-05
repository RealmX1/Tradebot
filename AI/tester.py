import numpy as np
from pathlib import Path
from typing import Dict, Any
from AI.model import DQNAgent
from data.data_loader import DataLoader
from sim.simulator import TradingSimulator
from sim.reward import RewardCalculator
import matplotlib.pyplot as plt
import pandas as pd

class Tester:
    def __init__(self, config: Dict[str, Any], use_fine_tuned: bool = True):
        self.config = config
        self.use_fine_tuned = use_fine_tuned
        self.data_loader = DataLoader(
            config['test_dir'],
            config['fred_data_path']
        )
        self.all_data = self.data_loader.load_data()
        print("Available test symbols:", self.all_data.keys())
        
        # Initialize components
        self.reward_calc = RewardCalculator()
        
        # Calculate state size using the first dataset's columns
        first_symbol = list(self.all_data.keys())[0]
        self.state_size = len(self.all_data[first_symbol]['bar_data'].columns) -1 + 2 # -1 for the raw close price, +2 for position info
        print(f"State size: {self.state_size} (features: {len(self.all_data[first_symbol]['bar_data'].columns)}, position info: 2)")
        
        # Don't load model in initialization if using fine-tuned models
        if not use_fine_tuned:
            self.model_path = Path(config['model_dir']) / "latest_model.pt"
            if not self.model_path.exists():
                raise FileNotFoundError(f"No trained model found at {self.model_path}")
                
            self.agent = DQNAgent(state_size=self.state_size)
            print(f"Loading model from {self.model_path}")
            self.agent.load(str(self.model_path))
            
            # Set epsilon to 0 for testing (no random actions)
            self.agent.epsilon = 0
        
    def test(self):
        results = {}
        test_metrics = []  # List to store metrics for each symbol
        
        for symbol in self.all_data.keys():
            print(f"\nTesting on {symbol}")
            
            bar_data = self.all_data[symbol]['bar_data']
            mock_trades = self.all_data[symbol]['mock_trades']
            
            simulator = TradingSimulator(
                self.config['initial_cash'],
                mock_trades,
                self.reward_calc,
                symbol
            )
            
            # Track performance metrics
            account_values = []
            actions = []
            positions = []
            raw_prices = []
            
            for timestamp, row in bar_data.iterrows():
                position = simulator.account.get_position(symbol)
                state, raw_close = DataLoader.prepare_state(row, position)
                
                action = self.agent.act(state)
                reward, info, done = simulator.step(timestamp, symbol, action, shares=-1)
                
                current_value = simulator.account.get_total_value({symbol: raw_close})
                account_values.append(current_value)
                actions.append(action)
                positions.append(position[1] if position else 0)
                raw_prices.append(raw_close)
                
                if done:
                    break
            
            # Calculate metrics
            initial_value = self.config['initial_cash']
            final_value = account_values[-1]
            account_value_pct_change = (final_value - initial_value) / initial_value
            
            # Calculate stock performance
            initial_price = raw_prices[0]
            final_price = raw_prices[-1]
            stock_pct_change = (final_price - initial_price) / initial_price
            
            # Calculate outperformance
            outperformance = account_value_pct_change - stock_pct_change
            
            # Store metrics in a dictionary
            symbol_metrics = {
                'symbol': symbol,
                'initial_value': initial_value,
                'final_value': final_value,
                'strategy_returns': account_value_pct_change,
                'stock_returns': stock_pct_change,
                'outperformance': outperformance,
                'num_trades': len([a for a in actions if a != 0])
            }
            test_metrics.append(symbol_metrics)
            
            # Store detailed results for plotting
            results[symbol] = {
                'initial_value': initial_value,
                'final_value': final_value,
                'returns': account_value_pct_change,
                'stock_returns': stock_pct_change,
                'outperformance': outperformance,
                'account_values': account_values,
                'actions': actions,
                'positions': positions,
                'prices': raw_prices,
                'dates': bar_data.index
            }
            
            print(f"Results for {symbol}:")
            print(f"Initial Value: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Strategy Return: {account_value_pct_change:.2%}")
            print(f"Stock Return: {stock_pct_change:.2%}")
            print(f"Strategy Outperformance: {outperformance:.2%}")
            print(f"Number of trades: {len([a for a in actions if a != 0])}")
            
        # Create DataFrame and calculate aggregated results
        results_df = pd.DataFrame(test_metrics)
        
        # Calculate aggregated metrics
        aggregated_metrics = {
            'symbol': 'AGGREGATE',
            'initial_value': results_df['initial_value'].mean(),
            'final_value': results_df['final_value'].mean(),
            'strategy_returns': results_df['strategy_returns'].mean(),
            'stock_returns': results_df['stock_returns'].mean(),
            'outperformance': results_df['outperformance'].mean(),
            'num_trades': results_df['num_trades'].mean()
        }
        
        # Append aggregated results
        results_df = pd.concat([results_df, pd.DataFrame([aggregated_metrics])], ignore_index=True)
        
        # Save results to CSV
        output_path = Path(self.config['model_dir']) / "test_results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nTest results saved to: {output_path}")
        
        # Print aggregated results
        print("\nAggregated Results:")
        print(f"Average Strategy Return: {aggregated_metrics['strategy_returns']:.2%}")
        print(f"Average Stock Return: {aggregated_metrics['stock_returns']:.2%}")
        print(f"Average Strategy Outperformance: {aggregated_metrics['outperformance']:.2%}")
        print(f"Average Number of Trades: {aggregated_metrics['num_trades']:.1f}")
        
        self._plot_results(results)
        return results
    
    def _plot_results(self, results):
        n_symbols = len(results)
        
        # Dimensions
        fig_height_per_symbol = 5
        max_visible_symbols = 5  # Max visible symbols in a single plot
        fig_width = 15
        
        # Check if the number of symbols exceeds the limit
        if n_symbols > max_visible_symbols:
            print(f"Too many symbols ({n_symbols}). Showing top {max_visible_symbols}.")
            n_symbols = max_visible_symbols

        # Set the figure size
        total_height = n_symbols * fig_height_per_symbol
        fig = plt.figure(figsize=(fig_width, total_height))

        # Create a GridSpec with spacing
        gs = plt.GridSpec(
            n_symbols, 2,
            figure=fig,
            hspace=0.4,  # Space between subplots
            wspace=0.3   # Space between plots
        )

        # Loop through results and plot
        for idx, (symbol, data) in enumerate(results.items()):
            if idx >= n_symbols:
                break  # Only plot up to the maximum visible symbols

            ax1 = fig.add_subplot(gs[idx, 0])
            ax2 = fig.add_subplot(gs[idx, 1])

            # Plot account value
            ax1.plot(data['dates'], data['account_values'], label='Account Value')
            ax1.set_title(f'{symbol} - Account Value Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value ($)')
            ax1.grid(True)

            # Plot price and positions
            ax2.plot(data['dates'], data['prices'], label='Price', color='blue')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)', color='blue')
            ax2.grid(True)

            ax3 = ax2.twinx()
            ax3.plot(data['dates'], data['positions'], label='Position', color='red', alpha=0.5)
            ax3.set_ylabel('Position Size', color='red')

            # Add legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax2.set_title(f'{symbol} - Price and Positions')

        plt.tight_layout()
        plt.show()

