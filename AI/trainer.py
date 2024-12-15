import numpy as np
from pathlib import Path
from typing import Dict, Any
import signal
import sys
import time
from collections import defaultdict
from visualization.train_plot import TrainingPlotter
from sim.simulator import TradingSimulator
from sim.reward import RewardCalculator
from AI.model import DQNAgent
from data_util.data_loader import DataLoader
import pandas as pd
import glob, os
from util.timer import TimingStats

class Trainer:
    def __init__(self, config: Dict[str, Any], fine_tune_symbol: str = None):
        self.config = config
        self.fine_tune_symbol = fine_tune_symbol
        self.data_loader = DataLoader(
            config['train_dir'],
            config['fred_data_path']
        )
        
        self.symbols = self.data_loader.get_symbols()
        
        # Load data for measuring state size
        if fine_tune_symbol:
            print("this is a fine tuning session")
            symbol = fine_tune_symbol
        else:
            print("this is a training session")
            symbol = self.symbols[0]
            
        
        loaded_data = self.data_loader.load_symbol_data(symbol=symbol)
        self.sample_bar_data = loaded_data['bar_data']
        self.sample_mock_trades = loaded_data['mock_trades']
        
        
        # Initialize components
        self.reward_calc = RewardCalculator()
        
        # Calculate state size using the first dataset's columns
        self.state_size = len(self.sample_bar_data.columns) - 1 + 2  # -1 for the raw close price, +2 for position info
        print(f"State size: {self.state_size} (features: {len(self.sample_bar_data.columns)}, position info: 2)")
        
        self.agent = DQNAgent(state_size=self.state_size)
        
        # Modify model loading logic
        if fine_tune_symbol:
            self.model_path = Path(config['model_dir']) / f"latest_{fine_tune_symbol}.pt"
            base_model_path = Path(config['model_dir']) / "latest_model.pt"
            
            if not base_model_path.exists():
                raise FileNotFoundError(f"No base model found at {base_model_path}")
                
            print(f"Loading base model from {base_model_path} for fine-tuning")
            self.agent.load(str(base_model_path))
        else:
            self.model_path = Path(config['model_dir']) / "latest_model.pt"
            if self.model_path.exists():
                print(f"Loading existing model from {self.model_path}")
                self.agent.load(str(self.model_path))
        
        self.plotter = TrainingPlotter()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.timing_stats = TimingStats()
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C by saving the model before exit"""
        print('\nSaving model before exit...')
        self.agent.save(str(self.model_path))
        print(f'Model saved to {self.model_path}')
        sys.exit(0)
        
    def train(self):
        try:
            all_losses = []
            all_rewards = []
            all_actions = []
            all_account_values = []
            
            for episode in range(self.config['episodes']):
                # Symbol selection based on mode
                if self.fine_tune_symbol:
                    symbol = self.fine_tune_symbol
                    self.bar_data = self.sample_bar_data
                    self.mock_trades = self.sample_mock_trades
                else:
                    symbol = np.random.choice(list(self.symbols))
                    print(f"\nTraining on {symbol}")
                    
                    loaded_data = self.data_loader.load_symbol_data(symbol)
                    self.bar_data = loaded_data['bar_data']
                    self.mock_trades = loaded_data['mock_trades']
                
                # Initialize simulator with current symbol's data
                self.simulator = TradingSimulator(
                    self.config['initial_cash'],
                    self.mock_trades,
                    self.reward_calc,
                    symbol
                )
                
                episode_stats = self._run_episode(episode, all_losses, all_rewards, all_actions, all_account_values, symbol)
                self._handle_episode_end(episode, episode_stats, symbol)
                
            # Save final model after training completion
            print('\nTraining completed. Saving final model...')
            self.agent.save(str(self.model_path))
            print(f'Final model saved to {self.model_path}')
        except KeyboardInterrupt:
            print('\nSaving model before exit...')
            self.agent.save(str(self.model_path))
            print(f'Model saved to {self.model_path}')
            sys.exit(0)
        # finally:
        #     self.plotter.close()
    
    def _run_episode(self, episode: int, all_losses: list, all_rewards: list, all_actions: list, all_account_values: list, symbol: str) -> dict:
        t0 = time.time()
        print(f"\nEpisode {episode + 1}/{self.config['episodes']}")
        self.simulator.reset()
        self.timing_stats.reset()
        self.plotter.reset()
        self.timing_stats.update('episode_reset', time.time() - t0)
        
        episode_reward = 0
        losses = []
        rewards = []
        actions = []
        account_values = []
        
        
        all_stock_prices = self.bar_data['close'].tolist()
        counter = 0
        for timestamp, row in self.bar_data.iterrows():
            counter += 1
            # if counter % 100 == 0:
            #     print(f"Processing row {counter}, {timestamp}")
                
            action, loss, reward, done, account_value = self._training_step(timestamp, row, symbol)
            
            t1 = time.time()
            
            losses.append(loss)
            all_losses.append(loss)
            rewards.append(reward)
            actions.append(action)
            account_values.append(account_value)
            
            if len(all_losses) % 1000 == 0:
                # Print timing stats
                # self.timing_stats.print_stats()
                # self.simulator.timing_stats.print_stats()
                # Plotting
                t_plot = time.time()
                self.plotter.update_plot(losses, rewards, actions, account_values, all_stock_prices)
                self.timing_stats.update('plot_update', time.time() - t_plot)
                    
            episode_reward += reward
            if done:
                print("episode done")
                break
            
            # Update utilization tracking after each step
            self.simulator.account.update_utilization(symbol)
            
            self.timing_stats.update('episode_processing', time.time() - t1)
        
        # Extract stock prices (assuming they're in log form in bar_data)
        all_stock_prices = [10 ** x for x in self.bar_data['close'].tolist()]
        
        return {
            'reward': episode_reward,
            'losses': losses,
            'simulator': self.simulator
        }
    
    def _training_step(self, timestamp, row, symbol):
        t0 = time.time()
        
        position = self.simulator.account.get_position(symbol)
        state, raw_close = DataLoader.prepare_state(row, position)
        self.timing_stats.update('state_preparation', time.time() - t0)
        
        # Get action from agent
        t1 = time.time()
        action = self.agent.act(state)
        self.timing_stats.update('agent_action', time.time() - t1)
        
        # Execute action
        t2 = time.time()
        reward, info, done = self.simulator.step(timestamp, symbol, action, shares=-1)
        self.timing_stats.update('simulator_step', time.time() - t2)
        
        # Prepare next state
        t3 = time.time()
        next_position = self.simulator.account.get_position(symbol)
        next_state, _ = DataLoader.prepare_state(row, next_position)
        self.timing_stats.update('next_state_prep', time.time() - t3)
        
        # Store experience and train
        t4 = time.time()
        self.agent.remember(state, action, reward, next_state, done)
        self.timing_stats.update('memory_storage', time.time() - t4)
        
        t5 = time.time()
        loss = self.agent.train_step()
        self.timing_stats.update('training_step', time.time() - t5)
        
        account_value = self.simulator.account.get_total_value({symbol: raw_close})
        
        return action, loss, reward, done, account_value
        
    def _handle_episode_end(self, episode: int, stats: dict, symbol: str):
        # Update target network
        if (episode + 1) % self.config['target_update_freq'] == 0:
            self.agent.update_target_network()
            
        # Save model
        if (episode + 1) % self.config['save_freq'] == 0:
            self.agent.save(f"{self.config['model_dir']}/agent_episode_{episode+1}.pt")
            
        # Print episode statistics
        avg_loss = np.mean(stats['losses']) if stats['losses'] else 0
        final_account_value = stats['simulator'].account.get_total_value()
        account_value_pct_change = (final_account_value - self.config['initial_cash'])/self.config['initial_cash']
        
        initial_close = 10 ** self.bar_data.iloc[0]['close']
        final_close = 10 ** self.bar_data.iloc[-1]['close']
        background_stock_pct_change = (final_close - initial_close)/initial_close
        
        # Calculate regular and utilization-adjusted outperformance
        outperformance = account_value_pct_change - background_stock_pct_change
        utilization_rate = stats['simulator'].account.utilization_rate
        utilization_adjusted_background_stock_pct_change = background_stock_pct_change * utilization_rate
        utilization_adjusted_outperformance = account_value_pct_change - utilization_adjusted_background_stock_pct_change
        
        relative_outperformance = account_value_pct_change / background_stock_pct_change - 1
        utilization_adjusted_relative_outperformance = account_value_pct_change / utilization_adjusted_background_stock_pct_change -1
        
        # Create log entry with new metrics
        log_entry = {
            'episode': episode + 1,
            'symbol': symbol,
            'reward': stats['reward'],
            'avg_loss': avg_loss,
            'epsilon': self.agent.epsilon,
            'final_account_value': final_account_value,
            'account_value_pct_change': account_value_pct_change,
            'stock_pct_change': background_stock_pct_change,
            'outperformance': outperformance,
            'relative_outperformance': relative_outperformance,
            'utilization_rate': utilization_rate,
            'utilization_adjusted_outperformance': utilization_adjusted_outperformance,
            'utilization_adjusted_relative_outperformance': utilization_adjusted_relative_outperformance,
            'filled_orders': len(stats['simulator'].filled_orders),
            'unfilled_orders': len(stats['simulator'].unfilled_orders),
            'total_sec_fees': stats['simulator'].account.total_sec_fees,
            'total_taf_fees': stats['simulator'].account.total_taf_fees,
            'total_fees': stats['simulator'].account.total_sec_fees + stats['simulator'].account.total_taf_fees
        }
        
        # Save to CSV
        log_path = Path('results') / 'training_log.csv'
        if not log_path.exists():
            pd.DataFrame([log_entry]).to_csv(log_path, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=False, index=False)
        
        # Print enhanced statistics
        print(f"Episode {episode + 1} ({symbol}): Reward = {stats['reward']:.2f}, Avg Loss = {avg_loss:.4f}, Epsilon = {self.agent.epsilon:.4f}")
        print(f"Account Value: {final_account_value:.2f}, pct change from init: {account_value_pct_change:.2%}")
        print(f"Background stock pct change: {background_stock_pct_change:.2%}, start price: {initial_close:.2f}, end price: {final_close:.2f}")
        print(f"Fund utilization rate: {utilization_rate:.2%}")
        print(f"Raw strategy outperformance: {outperformance:.2%}; relative: {relative_outperformance:.2%}")
        print(f"Utilization-adjusted outperformance: {utilization_adjusted_outperformance:.2%}; relative: {utilization_adjusted_relative_outperformance:.2%}")
        # print(f"Trading Fees - SEC: ${stats['simulator'].account.total_sec_fees:.2f}, TAF: ${stats['simulator'].account.total_taf_fees:.2f}")
        # print(f"Total Fees: ${(stats['simulator'].account.total_sec_fees + stats['simulator'].account.total_taf_fees):.2f}")
        print(f"Filled/Unfilled Orders: {len(stats['simulator'].filled_orders)}/{len(stats['simulator'].unfilled_orders)}")
        