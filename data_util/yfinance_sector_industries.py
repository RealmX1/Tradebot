import yfinance as yf
import pandas as pd
from data_util.symbols import pre_2000_snp_stock_symbols

def get_sector_and_industry(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A")
        }
    except Exception as e:
        return {"Error": str(e)}

industries = []
sectors = []
df = pd.DataFrame(columns=['symbol', 'Industry', 'Sector'])
for symbol in pre_2000_snp_stock_symbols:
    info = get_sector_and_industry(symbol)
    industry = info['Industry']
    sector = info['Sector']
    industries.append(industry)
    sectors.append(sector)
    df.loc[len(df)] = [symbol, industry, sector]

df.to_csv('symbols_industries_sectors.csv', index=False)
print(set(industries))
print(set(sectors))
print(len(set(industries)))
print(len(set(sectors)))