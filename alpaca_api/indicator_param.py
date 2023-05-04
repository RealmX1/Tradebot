import pandas_ta as ta
from enum import Enum
class IndicatorParam(Enum):
    ADX = {'length':14}
    RSI = (1,1)
    BBANDS = 20

for parm in IndicatorParam:
    print(parm.name, parm.value)