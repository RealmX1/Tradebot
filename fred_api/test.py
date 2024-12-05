from fredapi import Fred
import os

fred_key = open(os.path.join(os.path.dirname(__file__), 'fred.key'), 'r').read()
fred = Fred(api_key=fred_key)
data = fred.get_series('UMCSENT')
print(data)
data.to_csv('consumer_sentiment.csv')

