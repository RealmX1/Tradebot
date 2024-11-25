import pandas as pd
import numpy as np
from typing import List, Tuple
import torch
from pathlib import Path

from AI.model import DQNAgent
from sim.simulator import TradingSimulator
from sim.reward import RewardCalculator
from sim.account import Account

def prepare_state(row: pd.Series, position: Tuple[int, float]) -> np.ndarray:
    """Convert raw data row and position info into state vector"""
    # Get only numeric columns for features
    numeric_features = row.values.astype(np.float32)
    
    # Add position information
    position_features = np.array([
        float(position[0]),  # current shares
        float(position[1]),  # average price
    ], dtype=np.float32)
    
    # Concatenate and ensure the result is float32
    return np.concatenate([numeric_features, position_features]).astype(np.float32)

def train_agent(
    bar_data_path: str,
    mock_trade_path: str,
    initial_cash: float = 100000,
    episodes: int = 100,
    target_update_freq: int = 10,
    save_freq: int = 10,
    model_dir: str = "models"
):
    # Load data
    bar_data = pd.read_csv(bar_data_path, index_col=['timestamp', 'symbol'], parse_dates=True)
    mock_trades = pd.read_csv(mock_trade_path, index_col=['timestamp', 'symbol'], parse_dates=True)
    
    # Get only numeric columns and convert to float32
    numeric_columns = bar_data.select_dtypes(include=[np.number]).columns
    bar_data = bar_data[numeric_columns].astype(np.float32)
    
    # Initialize components
    reward_calc = RewardCalculator(profit_factor=1.0)
    simulator = TradingSimulator(initial_cash, mock_trades, reward_calc)
    
    # Calculate state size (numeric features + position info)
    state_size = len(numeric_columns) + 2  # +2 for position info
    print(f"State size: {state_size} (features: {len(numeric_columns)}, position info: 2)")
    agent = DQNAgent(state_size=state_size)
    
    # Create model directory
    Path(model_dir).mkdir(exist_ok=True)
    
    # Training loop
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        simulator.reset()
        episode_reward = 0
        losses = []
        
        # Episode loop
        counter = 0
        for timestamp, row in bar_data.iterrows():
            counter += 1
            if counter % 100 == 0:
                print(f"Processing row {counter}, {timestamp}")
            # Prepare state
            position = simulator.account.get_position('NVDA')
            state = prepare_state(row, position)
            
            # Get action from agent
            action = agent.act(state)
            
            # Execute action
            # For simplicity, use fixed number of shares
            shares = 100 if action != 0 else 0
            reward, info, done = simulator.step(timestamp, 'NVDA', action, shares)
            
            # Prepare next state
            next_position = simulator.account.get_position('NVDA')
            next_state = prepare_state(row, next_position)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            
            episode_reward += reward
            
            if done:
                break
                
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
            
        # Save model
        if (episode + 1) % save_freq == 0:
            agent.save(f"{model_dir}/agent_episode_{episode+1}.pt")
            
        # Print episode statistics
        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Avg Loss = {avg_loss:.4f}, "
              f"Epsilon = {agent.epsilon:.4f}")
        print(f"Account Value: {simulator.account.get_total_value({'NVDA': bar_data.iloc[-1]['close']}):.2f}")
        print(f"Filled/Unfilled Orders: {len(simulator.filled_orders)}/{len(simulator.unfilled_orders)}")

if __name__ == "__main__":
    bar_data_path = "data/train/bar_set_20200101_20230101_NVDA_1Min_15feature0_IEX.csv"
    mock_trade_path = "data/train/mock_trade/20200101_20230101_NVDA.csv"
    
    train_agent(bar_data_path, mock_trade_path) 