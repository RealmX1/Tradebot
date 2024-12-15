from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np

@dataclass
class Position:
    shares: int = 0
    avg_price: float = 0.0
    utilization_periods: int = 0  # New field to track periods with position

@dataclass 
class Order:
    shares: int
    price: float
    timestamp: str

class Account:
    """Simulates a trading account with cash and positions"""
    
    def __init__(self, initial_cash: float, price_lookup: Dict[str, float]):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.pending_orders: Dict[str, Order] = {}  # order_id -> Order
        
        # Constants for fees
        self.SEC_FEE_RATE = 8.00 / 1000000  # $22.90 per million dollars sold
        self.TAF_FEE_RATE = 0.000145  # $0.000145 per share
        self.TAF_FEE_CAP = 7.27
        
        self.price_lookup = price_lookup
        
        # Add tracking for utilization
        self.total_periods = 0
        self.utilized_periods = 0
        self.utilization_rate = 0.0
        
        # Add tracking for fees
        self.total_sec_fees = 0.0
        self.total_taf_fees = 0.0
    
    def update_utilization(self, symbol: str) -> None:
        """Update utilization metrics for each time period"""
        self.total_periods += 1
        if symbol in self.positions and self.positions[symbol].shares > 0:
            self.utilized_periods += 1
        self.utilization_rate = self.utilized_periods / self.total_periods if self.total_periods > 0 else 0
    
    def reset(self) -> None:
        """Resets the account state"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.pending_orders.clear()
        self.total_periods = 0
        self.utilized_periods = 0
        self.utilization_rate = 0.0
        self.total_sec_fees = 0.0
        self.total_taf_fees = 0.0
        
    def place_buy_order(self, symbol: str, shares: int, price: float, 
                       timestamp: str) -> str:
        """
        Attempts to place a buy order
        Returns order_id if successful, None if insufficient funds
        """
        if shares == -1:
            shares = int(self.get_total_value() / price)
        
        shares = min(shares, int(self.cash / price))
        if shares == 0:
            return None
        
        
        cost = shares * price
            
        order_id = f"buy_{timestamp}_{symbol}"
        self.pending_orders[order_id] = Order(shares, price, timestamp)
        self.cash -= cost
        
        if symbol not in self.positions:
            self.positions[symbol] = Position()
            
        return order_id
        
    def place_sell_order(self, symbol: str, shares: int, price: float, 
                        timestamp: str) -> str:
        """
        Attempts to place a sell order
        Returns order_id if successful, None if insufficient shares
        """
        # have the actual shares count be the smaller of the shares requested and the shares available
        if shares == -1:
            shares = int(self.get_total_value() / price)
            
        if symbol in self.positions:
            shares = min(shares, self.positions[symbol].shares)
        else:
            shares = 0
        
        if shares == 0:
            return None
            
        order_id = f"sell_{timestamp}_{symbol}"
        self.pending_orders[order_id] = Order(shares, price, timestamp)
        self.positions[symbol].shares -= shares
        
        return order_id
        
    def execute_buy(self, order_id: str) -> None:
        """Executes a filled buy order"""
        order = self.pending_orders[order_id]
        symbol = order_id.split('_')[2]
        
        position = self.positions[symbol]
        total_cost = position.shares * position.avg_price
        new_shares = position.shares + order.shares
        new_cost = total_cost + (order.shares * order.price)
        
        position.shares = new_shares
        position.avg_price = new_cost / new_shares
        
        del self.pending_orders[order_id]
        
    def execute_sell(self, order_id: str) -> None:
        """Executes a filled sell order"""
        order = self.pending_orders[order_id]
        
        # Calculate fees
        value = order.shares * order.price
        sec_fee = math.ceil(value * self.SEC_FEE_RATE * 100) / 100
        taf_fee = min(
            self.TAF_FEE_CAP,
            math.ceil(order.shares * self.TAF_FEE_RATE * 100) / 100
        )
        
        # Track fees
        self.total_sec_fees += sec_fee
        self.total_taf_fees += taf_fee
        
        self.cash += value - sec_fee - taf_fee
        del self.pending_orders[order_id]
        
    def cancel_order(self, order_id: str) -> None:
        """Cancels a pending order"""
        order = self.pending_orders[order_id]
        if order_id.startswith('buy'):
            self.cash += order.shares * order.price
        else:  # sell order
            symbol = order_id.split('_')[2]
            self.positions[symbol].shares += order.shares
        del self.pending_orders[order_id]
        
    def get_position(self, symbol: str) -> Tuple[float, float]: # returns (cash as percentage of portfolio value, log10(avg_price))
        """Returns (percentage of account value in cash, log10(avg_price)) for given symbol"""
        if symbol not in self.positions:
            return (0, 0.0)
        cash_ratio = self.cash / self.get_total_value()
        return (cash_ratio, np.log10(self.positions[symbol].avg_price))
        
    def get_total_value(self, price_lookup: Dict[str, float] = None, log: bool = False) -> float:
        """
        Calculate total account value using provided prices
        price_lookup: Dict[symbol -> current_price]
        """
        value = self.cash
        if not price_lookup:
            price_lookup = self.price_lookup
        self.price_lookup = price_lookup
            
        if log:
            print("calculating total value")
        for symbol, pos in self.positions.items():
            assert symbol in price_lookup
            value += pos.shares * price_lookup[symbol]
            if log:
                print(f"added {pos.shares} shares of {symbol} at {price_lookup[symbol]}, total value is {value}")
        return value 

    def get_state_normalized(self, price_lookup: Dict[str, float]) -> np.ndarray:
        """
        Returns normalized state representation of the account
        [holdings_value_1/total_value, ..., holdings_value_n/total_value, cash/total_value]
        """
        total_value = self.get_total_value(price_lookup)
        if total_value == 0:
            return np.zeros(len(price_lookup) + 1)
        
        # Calculate normalized holdings values
        normalized_holdings = []
        for symbol, price in price_lookup.items():
            position = self.positions.get(symbol, Position())
            holding_value = position.shares * price
            normalized_holdings.append(holding_value / total_value)
        
        # Add normalized cash
        normalized_holdings.append(self.cash / total_value)
        
        return np.array(normalized_holdings)

    def get_state_raw(self, price_lookup: Dict[str, float]) -> np.ndarray:
        """
        Returns raw state representation of the account
        [holdings_1, ..., holdings_n, cash]
        """
        holdings = []
        for symbol in price_lookup.keys():
            position = self.positions.get(symbol, Position())
            holdings.append(position.shares)
        
        holdings.append(self.cash)
        return np.array(holdings)

    def get_portfolio_metrics(self, price_lookup: Dict[str, float]) -> Dict:
        """
        Calculate various portfolio metrics
        """
        total_value = self.get_total_value(price_lookup)
        metrics = {
            'total_value': total_value,
            'cash_ratio': self.cash / total_value if total_value > 0 else 0,
            'positions': {}
        }
        
        for symbol, price in price_lookup.items():
            position = self.positions.get(symbol, Position())
            position_value = position.shares * price
            metrics['positions'][symbol] = {
                'shares': position.shares,
                'value': position_value,
                'weight': position_value / total_value if total_value > 0 else 0
            }
        
        return metrics