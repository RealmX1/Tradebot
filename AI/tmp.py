import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

raw_start = pd.to_datetime('2022-01-05')
raw_end = pd.to_datetime('2023-01-01')

def get_next_time_session(start, raw_end, type = 'bars'):
    if type == 'bars':
        end = start.replace(day=1) + relativedelta(years=+1)
    elif type == 'trades':
        if (start.day == 1):
            end = start.replace(day=16)
        else:
            end = start.replace(day=1) + relativedelta(months=+1)
    
    if end > raw_end:
        end = raw_end
    return end

start = raw_start

while start < raw_end:
    end = get_next_time_session(start, raw_end, type = 'trades')
    print(start, end)
    start = end