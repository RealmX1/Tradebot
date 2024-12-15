from typing import Dict, List, Optional, Tuple
import pandas as pd
import time
from .account import Account
from .reward import RewardCalculator
from util.timer import TimingStats

class TradingSimulator:
    """Simulates trading with an RL agent"""
    
    def __init__(self, 
                 initial_cash: float,
                 mock_trades: pd.DataFrame,
                 reward_calculator: RewardCalculator,
                 symbol: str):
        """
        mock_trades: DataFrame with columns [timestamp, price, volume]
        """
        self.initial_cash = initial_cash
        self.mock_trades = mock_trades
        price = mock_trades.iloc[0]['price']
        price_lookup = {symbol: price}
        self.account = Account(initial_cash, price_lookup)
        self.reward_calc = reward_calculator
        self.timing_stats = TimingStats()
        
        # Track metrics
        self.filled_orders: List[str] = []
        self.unfilled_orders: List[str] = []
        self.rewards: List[float] = []
        
    def step(self, 
            timestamp: str,
            symbol: str, 
            action: int,  # 0: hold, 1: buy, 2: sell
            shares: int = -1
            ) -> Tuple[float, Dict, bool]:
        """
        Executes one step of trading simulation
        Returns: (reward, info_dict, done)
        """
        t0 = time.time()
        # Get trade data for this timestamp
        trades = self.mock_trades[self.mock_trades.index == timestamp]
        if trades.empty:
            raise ValueError(f"No trades found for timestamp: {timestamp}")
            return 0.0, {}, True
            
        self.timing_stats.update('data_lookup', time.time() - t0)
        
        # Execute action
        t1 = time.time()
        order_id = None
        order_invalid = False
        order_unfilled = False
        current_price = trades.iloc[0]['price']
        
        if action == 1:  # Buy
            order_id = self.account.place_buy_order(
                symbol, shares, current_price, timestamp
            )
            if not order_id:
                order_invalid = True
            
        elif action == 2:  # Sell
            order_id = self.account.place_sell_order(
                symbol, shares, current_price, timestamp
            )
            if not order_id:
                order_invalid = True
                
        self.timing_stats.update('order_placement', time.time() - t1)
        
        t2 = time.time()
        # Check if order was placed successfully
        if order_id: # has madeorder
            # Try to fill the order
            order_unfilled = not self._try_fill_order(order_id, trades)
            if not order_unfilled:
                self.filled_orders.append(order_id)
            else:
                self.unfilled_orders.append(order_id)
                self.account.cancel_order(order_id)
                
        self.timing_stats.update('order_processing', time.time() - t2)
        
        t3 = time.time()
        reward = self.reward_calc.calculate_reward(
            self.account, symbol, current_price,
            order_id, order_invalid, order_unfilled
        )
        self.timing_stats.update('reward_calculation', time.time() - t3)
            
        t4 = time.time()    
        # Prepare info dict
        info = {
            'filled_orders': len(self.filled_orders),
            'unfilled_orders': len(self.unfilled_orders),
            'account_value': self.account.get_total_value(
                {symbol: trades.iloc[-1]['price']}
            )
        }
        self.timing_stats.update('info_preparation', time.time() - t4)
        
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
        self.reward_calc.reset()