class Account:
    def __init__(self, balance, symbols):
        self.balance = balance
        self.account_value = balance

        self.symbols = symbols
        self.holding = {symbol:[0,0] for symbol in symbols}
        
        self.orders = {symbol:{} for symbol in symbols}


    # buy order functions
    def place_buy_order(self, symbol, price, shares, order_id):
        self.orders[symbol][order_id] = (shares, price)
        self.holding[symbol][1] = price
        self.balance -= price * shares
    
    def place_buy_max_order(self, symbol, price, order_id):
        shares = self.balance // price
        self.orders[symbol][order_id] = (shares, price)
        self.holding[symbol][1] = price
        self.balance -= price * shares
        return shares
    
    def cancel_buy_order(self, symbol, order_id):
        self.balance += self.orders[symbol][order_id][0] * self.orders[symbol][order_id][1]
        del self.orders[symbol][order_id]
    
    def complete_buy_order(self, symbol, order_id):
        self.holding[symbol][0] += self.orders[symbol][order_id][0]
        del self.orders[symbol][order_id]

    # sell order functions
    def place_sell_order(self, symbol, price, shares, order_id):
        self.orders[symbol][order_id] = (shares, price)
        self.holding[symbol][1] = price
        self.holding[symbol][0] -= shares

    def place_sell_max_order(self, symbol, price, order_id):
        shares = self.holding[symbol][0]
        self.holding[symbol][1] = price
        self.orders[symbol][order_id] = (shares, price)
        self.holding[symbol][0] = 0
        return shares
    
    def cancel_sell_order(self, symbol, order_id):
        self.holding[symbol][0] += self.orders[symbol][order_id][0]
        del self.orders[symbol][order_id]

    def complete_sell_order(self, symbol, order_id):
        self.balance += self.orders[symbol][order_id][0] * self.orders[symbol][order_id][1]
        del self.orders[symbol][order_id]
    
    # short order functions
    def place_short_order(self, symbol, price, shares, order_id):
        self.orders[symbol][order_id] = (-shares, price)
        self.holding[symbol][1] = price
        self.balance += price * shares

    def place_short_max_order(self, symbol, price, order_id):
        shares = self.balance // price
        self.orders[symbol][order_id] = (-shares, price)
        self.holding[symbol][1] = price
        self.balance += price * shares
        return shares
    


    def get_account_balance(self):
        return self.balance
    
    def evaluate(self):
        self.account_value = self.balance
        for symbol in self.symbols:
            self.account_value += self.holding[symbol][0] * self.holding[symbol][1]
        return self.account_value

if __name__ == '__main__':
    acc = Account(100000, ['BABA'])
    print(acc.holding)
    print(acc.balance)
    acc.place_buy_max_order('BABA', 100, 0)
    print(acc.holding)
    print(acc.orders)
    print(acc.balance)
    acc.complete_buy_order('BABA', 0)
    print(acc.holding)
    print(acc.orders)
    print(acc.balance)
    print(acc.holding)

    print(acc.evaluate())
