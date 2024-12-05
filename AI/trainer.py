import numpy as np
from pathlib import Path
from typing import Dict, Any
import signal
import sys
from visualization.train_plot import TrainingPlotter
from sim.simulator import TradingSimulator
from sim.reward import RewardCalculator
from AI.model import DQNAgent
from data.data_loader import DataLoader
import pandas as pd

class Trainer:
    def __init__(self, config: Dict[str, Any], fine_tune_symbol: str = None):
        self.config = config
        self.fine_tune_symbol = fine_tune_symbol
        self.data_loader = DataLoader(
            config['train_dir'],
            config['fred_data_path']
        )
        self.all_data = self.data_loader.load_data()
        
        if fine_tune_symbol and fine_tune_symbol not in self.all_data:
            raise ValueError(f"Symbol {fine_tune_symbol} not found in training data")
            
        print(self.all_data.keys())
        
        # Initialize components
        self.reward_calc = RewardCalculator()
        
        # Calculate state size using the first dataset's columns
        first_symbol = list(self.all_data.keys())[0]
        self.state_size = len(self.all_data[first_symbol]['bar_data'].columns) - 1 + 2  # -1 for the raw close price, +2 for position info
        print(f"State size: {self.state_size} (features: {len(self.all_data[first_symbol]['bar_data'].columns)}, position info: 2)")
        
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
        
        self.plotter = TrainingPlotter(window_size=500)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
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
                else:
                    symbol = np.random.choice(list(self.all_data.keys()))
                    
                print(f"\nTraining on {symbol}")
                
                self.bar_data = self.all_data[symbol]['bar_data']
                self.mock_trades = self.all_data[symbol]['mock_trades']
                
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
        print(f"\nEpisode {episode + 1}/{self.config['episodes']}")
        self.simulator.reset()
        episode_reward = 0
        losses = []
        episode_account_values = []  # Track values for this episode
        
        counter = 0
        for timestamp, row in self.bar_data.iterrows():
            counter += 1
            # if counter % 100 == 0:
            #     print(f"Processing row {counter}, {timestamp}")
                
            action, loss, reward, done, account_value = self._training_step(timestamp, row, symbol)
            
            
            losses.append(loss)
            all_losses.append(loss)
            all_rewards.append(reward)
            all_actions.append(action)
            all_account_values.append(account_value)
            # if len(all_losses) % 50 == 0:
            #     self.plotter.update_plot(all_losses, all_rewards, all_actions, all_account_values)
                    
            episode_reward += reward
            if done:
                print("episode done")
                break
                
        return {
            'reward': episode_reward,
            'losses': losses,
            'simulator': self.simulator
        }
    
    def _training_step(self, timestamp, row, symbol):
        position = self.simulator.account.get_position(symbol)
        state, raw_close = DataLoader.prepare_state(row, position)
        
        # Get action from agent
        action = self.agent.act(state)
        
        # Execute action
        reward, info, done = self.simulator.step(timestamp, symbol, action, shares=-1)
        
        # Prepare next state
        next_position = self.simulator.account.get_position(symbol)
        next_state, _ = DataLoader.prepare_state(row, next_position)
        
        # Store experience and train
        self.agent.remember(state, action, reward, next_state, done)
        loss = self.agent.train_step()
        
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
        
        close_initial = 10 ** self.bar_data.iloc[0]['close']
        close_final = 10 ** self.bar_data.iloc[-1]['close']
        background_stock_pct_change = (close_final - close_initial)/close_initial
        outperformance = account_value_pct_change - background_stock_pct_change
        
        # Create log entry
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
            'filled_orders': len(stats['simulator'].filled_orders),
            'unfilled_orders': len(stats['simulator'].unfilled_orders)
        }
        
        # Save to CSV
        log_path = Path(self.config['model_dir']) / 'training_log.csv'
        if not log_path.exists():
            pd.DataFrame([log_entry]).to_csv(log_path, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=False, index=False)
        
        # Print existing statistics
        print(f"Episode {episode + 1} ({symbol}): Reward = {stats['reward']:.2f}, Avg Loss = {avg_loss:.4f}, Epsilon = {self.agent.epsilon:.4f}")
        print(f"Account Value: {final_account_value:.2f}, pct change from init: {account_value_pct_change:.2%}")
        print(f"Background stock pct change: {background_stock_pct_change:.2%}, start price: {close_initial:.2f}, end price: {close_final:.2f}")
        print(f"Strategy outperformance: {outperformance:.2%}")
        print(f"Filled/Unfilled Orders: {len(stats['simulator'].filled_orders)}/{len(stats['simulator'].unfilled_orders)}")
        