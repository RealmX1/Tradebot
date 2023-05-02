import sys
print(sys.path)
sys.path.append('../AI')
sys.path.append('../stock_data')
print(sys.path)

from alpaca_history_bars import *
from indicators import *


def 