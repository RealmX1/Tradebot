import numpy as np
from pathlib import Path
from typing import Dict, Any
from AI.model import DQNAgent
from data_util.data_loader import DataLoader
from sim.simulator import TradingSimulator
from sim.reward import RewardCalculator
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
from visualization.test_plot import TestPlotter

class Tester:
    def __init__(self, config: Dict[str, Any], use_fine_tuned: bool = False):
        self.config = config
        self.use_fine_tuned = use_fine_tuned
        self.data_loader = DataLoader(
            config['test_dir'],
            config['fred_data_path']
        )
        
        # Get list of available symbols
        self.test_symbols = self.data_loader.get_symbols()
        print("Available test symbols:", self.test_symbols)
        
        # Load data for first symbol to initialize state size
        if not self.test_symbols:
            raise FileNotFoundError("No test data files found")
            
        loaded_data = self.data_loader.load_symbol_data(self.test_symbols[0])
        self.sample_bar_data = loaded_data['bar_data']
        
        # Initialize components
        self.reward_calc = RewardCalculator()
        
        # Calculate state size using the first dataset's columns
        self.state_size = len(self.sample_bar_data.columns) - 1 + 2  # -1 for raw close price, +2 for position info
        print(f"State size: {self.state_size} (features: {len(self.sample_bar_data.columns)}, position info: 2)")
        
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
        
        self.plotter = TestPlotter()
        
    def test(self):
        results = {}
        test_metrics = []  # List to store metrics for each symbol
        
        for symbol in self.test_symbols:
            try:
                # Load data for current symbol
                loaded_data = self.data_loader.load_symbol_data(symbol)
                bar_data = loaded_data['bar_data']
                mock_trades = loaded_data['mock_trades']
                
                if self.use_fine_tuned:
                    model_path = Path(self.config['model_dir']) / f"latest_{symbol}.pt"
                    if not model_path.exists():
                        print(f"Fine-tuned model for {symbol} not found. Skip testing on {symbol}")
                        continue
                    else:
                        print(f"loading {model_path}")
                        self.agent = DQNAgent(state_size=self.state_size)
                        self.agent.load(str(model_path))
                        self.agent.epsilon = 0
                
                print(f"\nTesting on {symbol}")
                
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
                    positions.append(position[0] if position else 0)  # Use position[0] for shares instead of position[1]
                    raw_prices.append(raw_close)
                    
                    # Update utilization tracking after each step
                    simulator.account.update_utilization(symbol)
                    
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
                
                # Calculate outperformance metrics
                outperformance = account_value_pct_change - stock_pct_change
                relative_outperformance = account_value_pct_change / stock_pct_change - 1 if stock_pct_change != 0 else 0
                
                # Calculate utilization-adjusted metrics
                utilization_rate = simulator.account.utilization_rate
                utilization_adjusted_stock_pct_change = stock_pct_change * utilization_rate
                utilization_adjusted_outperformance = account_value_pct_change - utilization_adjusted_stock_pct_change
                utilization_adjusted_relative_outperformance = (
                    account_value_pct_change / utilization_adjusted_stock_pct_change - 1 
                    if utilization_adjusted_stock_pct_change != 0 else 0
                )
                
                # Store metrics in a dictionary
                symbol_metrics = {
                    'symbol': symbol,
                    'initial_value': initial_value,
                    'final_value': final_value,
                    'strategy_returns': account_value_pct_change,
                    'stock_returns': stock_pct_change,
                    'outperformance': outperformance,
                    'relative_outperformance': relative_outperformance,
                    'utilization_rate': utilization_rate,
                    'utilization_adjusted_outperformance': utilization_adjusted_outperformance,
                    'utilization_adjusted_relative_outperformance': utilization_adjusted_relative_outperformance,
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
                    'relative_outperformance': relative_outperformance,
                    'utilization_rate': utilization_rate,
                    'utilization_adjusted_outperformance': utilization_adjusted_outperformance,
                    'utilization_adjusted_relative_outperformance': utilization_adjusted_relative_outperformance,
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
                print(f"Fund utilization rate: {utilization_rate:.2%}")
                print(f"Raw strategy outperformance: {outperformance:.2%}; relative: {relative_outperformance:.2%}")
                print(f"Utilization-adjusted outperformance: {utilization_adjusted_outperformance:.2%}; relative: {utilization_adjusted_relative_outperformance:.2%}")
                print(f"Number of trades: {len([a for a in actions if a != 0])}")
                
            except FileNotFoundError:
                print(f"Data for {symbol} not found. Skipping testing on {symbol}")
                continue
        
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
            'relative_outperformance': results_df['relative_outperformance'].mean(),
            'utilization_rate': results_df['utilization_rate'].mean(),
            'utilization_adjusted_outperformance': results_df['utilization_adjusted_outperformance'].mean(),
            'utilization_adjusted_relative_outperformance': results_df['utilization_adjusted_relative_outperformance'].mean(),
            'num_trades': results_df['num_trades'].mean()
        }
        
        # Append aggregated results
        results_df = pd.concat([results_df, pd.DataFrame([aggregated_metrics])], ignore_index=True)
        
        # Save results to CSV
        output_path = Path('results') / "test_results.csv"
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nTest results saved to: {output_path}")
        
        # Print aggregated results
        print("\nAggregated Results:")
        print(f"Average Strategy Return: {aggregated_metrics['strategy_returns']:.2%}")
        print(f"Average Stock Return: {aggregated_metrics['stock_returns']:.2%}")
        print(f"Average Strategy Outperformance: {aggregated_metrics['outperformance']:.2%}")
        print(f"Average Relative Outperformance: {aggregated_metrics['relative_outperformance']:.2%}")
        print(f"Average Fund Utilization Rate: {aggregated_metrics['utilization_rate']:.2%}")
        print(f"Average Utilization-Adjusted Outperformance: {aggregated_metrics['utilization_adjusted_outperformance']:.2%}")
        print(f"Average Utilization-Adjusted Relative Outperformance: {aggregated_metrics['utilization_adjusted_relative_outperformance']:.2%}")
        print(f"Average Number of Trades: {aggregated_metrics['num_trades']:.1f}")
        
        self.plotter.plot_results(results)
        return results

