from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from alpaca_api_param import *


def create_market_order(symbol, qty, stop, take_profit, order_side, tif):
    stop_loss_request = StopLossRequest(
    stop_price=stop
    )
    take_profit_request = TakeProfitRequest(
    limit_price=take_profit
    )
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=tif,
        order_class = OrderClass.BRACKET,
        stop_loss=stop_loss_request,
        take_profit=take_profit_request
    )
    return market_order_data

def create_limit_order(symbol, qty, price, order_side, tif, stop = None):
    # stop_loss_request = StopLossRequest(
    # stop_price=stop
    # )
    # take_profit_request = TakeProfitRequest(
    # limit_price=stop
    # )
    limit_order_request = LimitOrderRequest(
        symbol=symbol,
        limit_price = price,
        qty=qty,
        side=order_side,
        time_in_force=tif,
        order_class = OrderClass.SIMPLE
        # stop_loss=stop_loss_request, # might be useful; but how to utilize it; how to determine whether it is useful?
        # take_profit=take_profit_request
    )
    return limit_order_request

# Submit an order and print the returned object function
def submit_order(trading_client, order_data):
    order = trading_client.submit_order(order_data)
    for property_name, value in order:
        print(f'"{property_name}": {value}')


def main():
    trading_client = TradingClient(PAPER_API_KEY, PAPER_SECRET_KEY, paper = True)
    price = 100
    # Request a Bracket market order (calling Bracket market order function)
    market_order_data = create_limit_order(symbol = "AAPL", price = price, qty = 1, order_side = OrderSide.BUY, tif = TimeInForce.DAY)
    # Submit market order
    submit_order(trading_client, market_order_data)


if __name__ == "__main__":
    main()