from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

API_KEY = "PKGNSI31E7XI9ACCSSVZ"
SECRET_KEY =  "yhupKUckY5vAbP7UOrkB26v4X4Gb9cdffo39V4OM"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def bracket_order(symbol, qty, stop, take_profit, order_side, tif):
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

# Submit an order and print the returned object function
def submit_order(order_data):
    order = trading_client.submit_order(order_data)
    for property_name, value in order:
        print(f'"{property_name}": {value}')


def main():

    # Request a Bracket market order (calling Bracket market order function)
    market_order_data = bracket_order(symbol = "AAPL", qty = 1, stop = 100, take_profit = 200, order_side = OrderSide.BUY, tif = TimeInForce.GTC)
    # Submit market order
    submit_order(market_order_data)


if __name__ == "__main__":
    main()