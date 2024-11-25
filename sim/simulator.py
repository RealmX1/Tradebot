from typing import Dict, List, Optional, Tuple
import pandas as pd
from .account import Account
from .reward import RewardCalculator

class TradingSimulator:
    """Simulates trading with an RL agent"""
    
    def __init__(self, 
                 initial_cash: float,
                 trade_data: pd.DataFrame,
                 reward_calculator: RewardCalculator):
        """
        trade_data: DataFrame with columns [timestamp, price, volume]
        """
        self.account = Account(initial_cash)
        self.trade_data = trade_data
        self.reward_calc = reward_calculator
        
        # Track metrics
        self.filled_orders: List[str] = []
        self.unfilled_orders: List[str] = []
        self.rewards: List[float] = []
        
    def step(self, 
            timestamp: str,
            symbol: str, 
            action: int,  # 0: hold, 1: buy, 2: sell
            shares: int
            ) -> Tuple[float, Dict, bool]:
        """
        Executes one step of trading simulation
        Returns: (reward, info_dict, done)
        """
        # Get trade data for this timestamp
        trades = self.trade_data[self.trade_data.index == timestamp]
        if trades.empty:
            return 0.0, {}, True
            
        # Execute action
        order_id = None
        reward = 0.0
        
        if action == 1:  # Buy
            current_price = trades.iloc[0]['price']
            order_id = self.account.place_buy_order(
                symbol, shares, current_price, timestamp
            )
            
        elif action == 2:  # Sell
            current_price = trades.iloc[0]['price']
            order_id = self.account.place_sell_order(
                symbol, shares, current_price, timestamp
            )
            
        # Check if order was placed successfully
        if order_id:
            # Try to fill the order
            order_filled = self._try_fill_order(order_id, trades)
            if order_filled:
                self.filled_orders.append(order_id)
                # Calculate reward only for filled orders
                reward = self.reward_calc.calculate_filled_order_reward(
                    self.account,
                    order_id,
                    trades
                )
            else:
                self.unfilled_orders.append(order_id)
                self.account.cancel_order(order_id)
                reward = self.reward_calc.calculate_unfilled_order_penalty()
        else:
            # Order couldn't be placed (insufficient funds/shares)
            reward = self.reward_calc.calculate_invalid_order_penalty()
                
        # Prepare info dict
        info = {
            'filled_orders': len(self.filled_orders),
            'unfilled_orders': len(self.unfilled_orders),
            'account_value': self.account.get_total_value(
                {symbol: trades.iloc[-1]['price']}
            )
        }
        
        return reward, info, False
        
    def _try_fill_order(self, 
                       order_id: str, 
                       trades: pd.DataFrame) -> bool:
        """
        Attempts to fill order using trade data
        Returns True if filled, False if unfilled
        """
        order = self.account.pending_orders[order_id]
        
        if order_id.startswith('buy'):
            # Buy order fills if price falls to or below order price
            fills = trades[trades['price'] <= order.price]
            if not fills.empty:
                self.account.execute_buy(order_id)
                return True
                
        else:  # Sell order
            # Sell order fills if price rises to or above order price
            fills = trades[trades['price'] >= order.price]
            if not fills.empty:
                self.account.execute_sell(order_id)
                return True
                
        return False
        
    def reset(self) -> None:
        """Resets the simulation state"""
        self.account.reset()
        self.filled_orders.clear()
        self.unfilled_orders.clear()
        self.rewards.clear() 