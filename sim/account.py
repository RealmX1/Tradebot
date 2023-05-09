import math
just_one_more_constant = 0.375 # https://www.youtube.com/watch?v=_FuuYSM7yOo&list=TLPQMjkwNDIwMjOTFJtp1wN2Pg&index=3

#https://alpaca.markets/support/regulatory-fees
SEC = 8.00 / 1000000 # SEC regulation fee is applied at 22.90 dollars per million dollars in sale.
# which is 0.000008 dollars per dollar, or 0.0008% of each trade
# IT NEED TO BE ROUNDED TO NEAREST CENT

TAF = 0.000145 # TAF Trading Activity Fee (TAF) from FINRA is applied at 0.000145 dollars per share. no greater than $7.27.
TAF_CEIL = 7.27
# for a 500$ share it is 0.000145 / 500 = 2.9E-7 which is 0.000029%
# for a 1$ share it is 0.000145 / 1 = 0.000145 which is 0.0145%
# IT NEED TO BE ROUNDED TO NEAREST CENT
# It can be no greater than $5.95


class Account:
    def __init__(self, balance, symbols):
        self.balance = balance
        self.frozen_balance = 0 # for shorting.
        self.account_value = balance

        self.symbols = symbols
        self.holding = {symbol:[0,0] for symbol in symbols}
        
        self.orders = {symbol:{} for symbol in symbols}


    # buy order functions
    def place_buy_order(self, symbol, price, shares, order_id):
        affordable_shares = (self.balance + self.frozen_balance) // price
        if affordable_shares < shares:
            return 0
        else:
            self.orders[symbol][order_id] = (shares, price)
            self.holding[symbol][1] = price
            self.balance -= price * shares
            return 1
    
    def place_buy_max_order(self, symbol, price, order_id, optimal = True):
        shares = (self.balance + self.frozen_balance) // price
        
        if optimal:
            current_holding = self.holding[symbol][0]
            if current_holding >= 0:
                shares = int(shares*just_one_more_constant)
            else: # fill short position, then purchase as much asneeded
                shares = -current_holding + int((shares+current_holding) * just_one_more_constant)

        if shares <= 0:
            return 0
        else:
            self.place_buy_order(symbol, price, shares, order_id)
        
        return shares
    
    def cancel_buy_order(self, symbol, order_id):
        self.balance += self.orders[symbol][order_id][0] * self.orders[symbol][order_id][1]
        del self.orders[symbol][order_id]
    
    def complete_buy_order(self, symbol, order_id):
        shares = self.orders[symbol][order_id][0]
        self.holding[symbol][0] += shares
        if self.holding[symbol][0] >= 0:
            self.balance += self.frozen_balance
            self.frozen_balance = 0
        del self.orders[symbol][order_id]
        return shares

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
        shares = self.orders[symbol][order_id][0]
        price = self.orders[symbol][order_id][1]
        value = shares * price
        SEC_FEE = math.ceil(value * SEC * 100) / 100
        TAF_FEE = math.ceil(shares * TAF * 100) / 100
        TAF_FEE = min (TAF_CEIL, TAF_FEE)
        print(f'TAF_FEE: {TAF_FEE}, SEC_FEE: {SEC_FEE}')
        self.balance += value - TAF_FEE - SEC_FEE
        del self.orders[symbol][order_id]
        return shares
    
    # short order functions
    def place_short_order(self, symbol, price, shares, order_id):
        affordable_shares = self.balance // price
        if affordable_shares < shares:
            return 0
        else:
            self.orders[symbol][order_id] = (-shares, price)
            self.holding[symbol][1] = price
            cost = price * shares
            self.balance -= cost
            self.frozen_balance += 2 * cost
            return 1

    def place_short_max_order(self, symbol, price, order_id, optimal = True):
        shares = self.balance // price

        if optimal:
            shares = int(shares*just_one_more_constant)
        
        self.place_short_order(symbol, price, shares, order_id)

        if shares <= 0:
            return 0

        return shares
    
    def cancel_short_order(self, symbol, order_id):
        cost = self.orders[symbol][order_id][0] * self.orders[symbol][order_id][1]
        self.balance -= cost
        self.frozen_balance += 2 * cost
        del self.orders[symbol][order_id]
    
    def complete_short_order(self, symbol, order_id):
        self.holding[symbol][0] += self.orders[symbol][order_id][0]
        
        shares = -self.orders[symbol][order_id][0]
        price = self.orders[symbol][order_id][1]
        value = shares * price
        TAF_FEE = math.ceil(shares * TAF * 100) / 100
        TAF_FEE = min (TAF_CEIL, TAF_FEE)
        SEC_FEE = math.ceil(value * SEC * 100) / 100
        self.balance = self.balance - TAF_FEE - SEC_FEE
        del self.orders[symbol][order_id]
        return shares # positive shares



    def place_reverse_short_order(self, symbol, price, order_id):
        shares = self.holding[symbol][0]
        if shares >= 0:
            return 0
        else:
            has_money = self.place_buy_order(symbol, price, -shares, order_id)

            if has_money:
                return -shares
            else:
                print('not enough to close short position!!!')
                return -1
    
    def complete_reverse_short_order(self, symbol, order_id):
        return self.complete_buy_order(symbol, order_id)

    def cancel_reverse_short_order(self, symbol, order_id):
        self.cancel_buy_order(symbol, order_id)

    def get_free_account_balance(self):
        return self.balance

    def get_total_account_balance(self):
        return self.balance + self.frozen_balance
    
    def evaluate(self):
        self.account_value = self.balance + self.frozen_balance
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
