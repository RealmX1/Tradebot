from typing import Dict
import pandas as pd
from .account import Account

class RewardCalculator:
    """Calculates rewards for RL training"""
    
    def __init__(self,
                 profit_factor: float = 1.0,
                 unfilled_penalty: float = -0.1,
                 invalid_penalty: float = -0.2):
        self.profit_factor = profit_factor
        self.unfilled_penalty = unfilled_penalty
        self.invalid_penalty = invalid_penalty
        
    def calculate_filled_order_reward(self,
                                    account: Account,
                                    order_id: str,
                                    trades: pd.DataFrame) -> float:
        """Calculate reward for a successfully filled order"""
        symbol = order_id.split('_')[2]
        position = account.get_position(symbol)
        
        if order_id.startswith('buy'):
            # For buys, small positive reward to encourage trading
            return 0.1
            
        else:  # Sell order
            # Calculate profit
            profit = (trades.iloc[0]['price'] - position[1]) * position[0]
            return profit * self.profit_factor
            
    def calculate_unfilled_order_penalty(self) -> float:
        """Return penalty for unfilled orders"""
        return self.unfilled_penalty
        
    def calculate_invalid_order_penalty(self) -> float:
        """Return penalty for invalid orders (insufficient funds/shares)"""
        return self.invalid_penalty