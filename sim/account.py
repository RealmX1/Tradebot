from dataclasses import dataclass
from typing import Dict, Tuple
import math

@dataclass
class Position:
    shares: int = 0
    avg_price: float = 0.0

@dataclass 
class Order:
    shares: int
    price: float
    timestamp: str

class Account:
    """Simulates a trading account with cash and positions"""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.pending_orders: Dict[str, Order] = {}  # order_id -> Order
        
        # Constants for fees
        self.SEC_FEE_RATE = 8.00 / 1000000  # $22.90 per million dollars sold
        self.TAF_FEE_RATE = 0.000145  # $0.000145 per share
        self.TAF_FEE_CAP = 7.27
    
    def reset(self) -> None:
        """Resets the account state"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.pending_orders.clear()
        
    def place_buy_order(self, symbol: str, shares: int, price: float, 
                       timestamp: str) -> str:
        """
        Attempts to place a buy order
        Returns order_id if successful, None if insufficient funds
        """
        cost = shares * price
        if cost > self.cash:
            return None
            
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
        if symbol not in self.positions or self.positions[symbol].shares < shares:
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
        
    def get_position(self, symbol: str) -> Tuple[int, float]:
        """Returns (shares, avg_price) for given symbol"""
        if symbol not in self.positions:
            return (0, 0.0)
        pos = self.positions[symbol]
        return (pos.shares, pos.avg_price)
        
    def get_total_value(self, price_lookup: Dict[str, float]) -> float:
        """
        Calculate total account value using provided prices
        price_lookup: Dict[symbol -> current_price]
        """
        value = self.cash
        for symbol, pos in self.positions.items():
            if symbol in price_lookup:
                value += pos.shares * price_lookup[symbol]
        return value 