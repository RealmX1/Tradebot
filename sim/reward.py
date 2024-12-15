from typing import Dict
import pandas as pd
import numpy as np
from .account import Account
import logging
import matplotlib.pyplot as plt

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class RewardCalculator:
    """Calculates rewards for RL training using portfolio-based metrics"""
    
    def __init__(self,
                 returns_multiplier: float = 5.0,
                 sharpe_factor: float = 0.1,
                 volatility_penalty: float = 0.1,
                 transaction_penalty: float = 0.01,
                 plot_reward_components: bool = False):
        self.returns_multiplier = returns_multiplier
        self.sharpe_factor = sharpe_factor
        self.volatility_penalty = volatility_penalty
        self.transaction_penalty = transaction_penalty
        self.unfilled_order_penalty = transaction_penalty
        self.invalid_order_penalty = 10 * transaction_penalty
        self.plot_reward_components = plot_reward_components
        
        self.returns_history = []
        
        # Add component tracking
        self.component_history = {
            'returns': [],
            'sharpe': [],
            'volatility_penalty': [],
            'transaction_penalty': [],
            'unfilled_penalty': [],
            'invalid_penalty': []
        }
        self.reward_count = 0
        
        logger.debug(f"Initialized RewardCalculator with sharpe_factor={sharpe_factor}, "
                    f"volatility_penalty={volatility_penalty}, "
                    f"transaction_penalty={transaction_penalty}")
    
    def calculate_reward(self,
                        account: Account,
                        symbol: str,
                        current_price: float,
                        order_id: str,
                        order_invalid: bool = False,
                        order_unfilled: bool = False) -> float:
        """Calculate reward based on portfolio performance metrics"""
        # Get current portfolio value
        portfolio_value = account.get_total_value({symbol: current_price})
        
        # Calculate return
        if len(self.returns_history) > 0:
            portfolio_return = (portfolio_value - self.returns_history[-1]) / self.returns_history[-1]
        else:
            portfolio_return = 0
        portfolio_return *= self.returns_multiplier
            
        self.returns_history.append(portfolio_value)
        
        # Calculate Sharpe ratio component (if we have enough history)
        sharpe_component = 0
        volatility_penalty = 0
        # if len(self.returns_history) > 1:
        #     returns_array = np.diff(self.returns_history) / self.returns_history[:-1]
        #     sharpe_ratio = 0
        #     if np.std(returns_array) != 0:
        #         sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
        #     sharpe_component = sharpe_ratio * self.sharpe_factor
        
        # # Calculate volatility penalty
        # if len(self.returns_history) > 1:
        #     returns_std = np.std(np.diff(self.returns_history) / self.returns_history[:-1])
        #     volatility_penalty = returns_std * self.volatility_penalty
        
        # Calculate transaction cost penalty
        has_order = order_id is not None
        transaction_penalty = self.transaction_penalty * has_order
        
        unfilled_penalty = order_unfilled * self.unfilled_order_penalty
        invalid_penalty = order_invalid * self.invalid_order_penalty
        
        # Track individual components
        if self.plot_reward_components:
            self.component_history['returns'].append(portfolio_return)
            self.component_history['sharpe'].append(sharpe_component)
            self.component_history['volatility_penalty'].append(volatility_penalty)
            self.component_history['transaction_penalty'].append(transaction_penalty)
            self.component_history['unfilled_penalty'].append(unfilled_penalty)
            self.component_history['invalid_penalty'].append(invalid_penalty)
            
            self.reward_count += 1
        
            # Plot components every 1000 calculations
            if self.reward_count % 1000 == 0:
                self.plot_component_distribution()
                
        # Combine components
        reward = portfolio_return + sharpe_component - volatility_penalty - transaction_penalty \
            - unfilled_penalty \
            - invalid_penalty
        
        # logger.debug(f"Reward calculation for {order_id}:")
        # logger.debug(f"  Portfolio value: {portfolio_value:.2f}")
        # logger.debug(f"  Return: {portfolio_return:.4f}")
        # logger.debug(f"  Sharpe component: {sharpe_component:.4f}")
        # logger.debug(f"  Volatility penalty: {volatility_penalty:.4f}")
        # logger.debug(f"  Transaction penalty: {transaction_penalty:.4f}")
        # logger.debug(f"  Final reward: {reward:.4f}")
        
        return reward
    
    def plot_component_distribution(self):
        """Plot the average contribution of each reward component"""
        components = {}
        for component, values in self.component_history.items():
            if values:  # Check if we have any values
                components[component] = np.mean(np.abs(values))
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = ['g', 'b', 'r', 'orange', 'purple', 'brown']
        plt.bar(components.keys(), components.values(), color=colors)
        plt.title('Average Absolute Contribution of Reward Components')
        plt.xticks(rotation=45)
        plt.ylabel('Absolute Magnitude')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('reward_components.png')
        plt.close()
        
        # Log the component averages
        logger.info("Average reward component contributions:")
        for component, value in components.items():
            logger.info(f"  {component}: {value:.4f}")
    
    def reset(self):
        """Reset the reward calculator state"""
        self.returns_history.clear()
        # Clear component history as well
        for component in self.component_history.values():
            component.clear()
        self.reward_count = 0
