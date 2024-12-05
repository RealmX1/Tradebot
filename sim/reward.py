from typing import Dict
import pandas as pd
import numpy as np
from .account import Account
import logging

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
                 transaction_penalty: float = 0.01):
        self.returns_multiplier = returns_multiplier
        self.sharpe_factor = sharpe_factor
        self.volatility_penalty = volatility_penalty
        self.transaction_penalty = transaction_penalty
        self.unfilled_order_penalty = transaction_penalty
        self.invalid_order_penalty = 10 * transaction_penalty
        self.returns_history = []
        
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
        if len(self.returns_history) > 1:
            returns_array = np.diff(self.returns_history) / self.returns_history[:-1]
            sharpe_ratio = 0
            if np.std(returns_array) != 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
            sharpe_component = sharpe_ratio * self.sharpe_factor
        
        # Calculate volatility penalty
        volatility_penalty = 0
        if len(self.returns_history) > 1:
            returns_std = np.std(np.diff(self.returns_history) / self.returns_history[:-1])
            volatility_penalty = returns_std * self.volatility_penalty
        
        # Calculate transaction cost penalty
        has_order = order_id is not None
        transaction_penalty = self.transaction_penalty * has_order
        
        # Combine components
        reward = portfolio_return + sharpe_component - volatility_penalty - transaction_penalty \
            - order_unfilled*self.unfilled_order_penalty \
            - order_invalid*self.invalid_order_penalty
        
        logger.debug(f"Reward calculation for {order_id}:")
        logger.debug(f"  Portfolio value: {portfolio_value:.2f}")
        logger.debug(f"  Return: {portfolio_return:.4f}")
        logger.debug(f"  Sharpe component: {sharpe_component:.4f}")
        logger.debug(f"  Volatility penalty: {volatility_penalty:.4f}")
        logger.debug(f"  Transaction penalty: {transaction_penalty:.4f}")
        logger.debug(f"  Final reward: {reward:.4f}")
        
        return reward
    
    def reset(self):
        """Reset the reward calculator state"""
        self.returns_history.clear()