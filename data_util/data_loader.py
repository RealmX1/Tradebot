import pandas as pd
import numpy as np
from typing import Tuple
import glob
import os

class DataLoader:
    def __init__(self, data_dir: str, 
                 fred_data_path: str = "data/fred/normalized_fred_data.csv",
                 normalization_path: str = "feature_engineering/indicator_normalization.csv"):
        self.data_dir = data_dir
        self.fred_data_path = fred_data_path
        self.normalization_path = normalization_path
        self.fred_data = self.load_fred_data()
        self.symbols_industries_sectors_df = pd.read_csv('symbols_industries_sectors.csv')
        # sort first by sector, then by industry
        self.symbols_industries_sectors_df = self.symbols_industries_sectors_df.sort_values(by=['Sector', 'Industry'])
        self.sectors = self.symbols_industries_sectors_df['Sector'].unique()
        self.industries = self.symbols_industries_sectors_df['Industry'].unique()

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data according to the normalization rules"""
        # Load normalization rules
        norm_rules = pd.read_csv(self.normalization_path)
        
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Store raw close price before normalization
        normalized_df['raw_close'] = normalized_df['close'].copy()
        
        # First handle price normalization
        price_indicators = norm_rules[norm_rules['norm_type'] == 'price']['indicator'].tolist()
        
        if price_indicators:
            # Apply log10 to all price columns
            for col in price_indicators:
                if col in normalized_df.columns:
                    normalized_df[col] = np.log10(normalized_df[col])
            
            # Subtract close from other price indicators
            close_series = normalized_df['close']
            for col in price_indicators:
                if col in normalized_df.columns and col != 'close':
                    normalized_df[col] = normalized_df[col] - close_series
        
        return normalized_df

    def load_fred_data(self) -> pd.DataFrame:
        """Load and process FRED data"""
        # Load normalized FRED data
        fred_data = pd.read_csv(self.fred_data_path, parse_dates=['date'])
        fred_data['date'] = pd.to_datetime(fred_data['date'], utc=True)
        fred_data.set_index('date', inplace=True)
        
        # Resample to daily frequency and forward fill
        # This ensures FRED data aligns with market data timestamps
        fred_data = fred_data.resample('D').ffill()
        
        return fred_data
    
    def get_symbols(self):
        """Get all symbols from the data directory"""
        bar_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        return [os.path.basename(file).split('_')[2] for file in bar_files]

    def get_symbol_sector_industry(self, symbol: str) -> Tuple[str, str]:
        """Get sector and industry for a given symbol"""
        symbol_info = self.symbols_industries_sectors_df[
            self.symbols_industries_sectors_df['Symbol'] == symbol
        ]
        if len(symbol_info) == 0:
            raise ValueError(f"Symbol {symbol} not found in sector/industry data")
        
        return symbol_info.iloc[0]['Sector'], symbol_info.iloc[0]['Industry']
    
    def create_one_hot_features(self, sector: str, industry: str) -> pd.DataFrame:
        """Create one-hot encoded features for sector and industry"""
        # Create one-hot vectors
        sector_one_hot = pd.get_dummies([sector], prefix='sector')[0]
        industry_one_hot = pd.get_dummies([industry], prefix='industry')[0]
        
        # Ensure all sectors and industries are represented
        for s in self.sectors:
            if f'sector_{s}' not in sector_one_hot:
                sector_one_hot[f'sector_{s}'] = 0
        
        for i in self.industries:
            if f'industry_{i}' not in industry_one_hot:
                industry_one_hot[f'industry_{i}'] = 0
                
        # Combine sector and industry features
        one_hot_features = pd.concat([sector_one_hot, industry_one_hot])
        
        return one_hot_features.astype(np.float32)

    def load_symbol_data(self, symbol: str) -> dict:
        """Load and preprocess bar data and mock trades for a specific symbol"""
        print(f"Loading data for symbol: {symbol}")
        
        # Get all bar data files in data directory
        bar_files = glob.glob(os.path.join(self.data_dir, f"*_{symbol}_1Day_*.csv"))
        
        if not bar_files:
            raise FileNotFoundError(f"No data file found for symbol {symbol}")
        
        # Use the first matching file (should only be one)
        assert len(bar_files) == 1, f"Expected 1 bar file for symbol {symbol}, but found {len(bar_files)}"
        bar_file = bar_files[0]
        
        # Extract filename components
        filename = os.path.basename(bar_file)
        start_date, end_date, symbol, timeframe, feature_str, exchange = filename.split('_')
        
        # Construct corresponding mock trade filename
        mock_file = os.path.join(self.data_dir, "mock_trade", start_date + "_" + end_date + "_" + symbol + "_" + timeframe + ".csv")
        
        if not os.path.exists(mock_file):
            raise FileNotFoundError(f"Mock trade file not found for {symbol}")
        
        # Load market data
        bar_data = pd.read_csv(bar_file, index_col='timestamp', parse_dates=True)
        mock_trades = pd.read_csv(mock_file, index_col='timestamp', parse_dates=True)
        bar_data.index = pd.to_datetime(bar_data.index, utc=True)
        mock_trades.index = pd.to_datetime(mock_trades.index, utc=True)
        
        bar_data.index = bar_data.index.normalize()
        mock_trades.index = mock_trades.index.normalize()
        
        # drop the "symbol" column
        bar_data = bar_data.drop(columns=['symbol'])
        mock_trades = mock_trades.drop(columns=['symbol'])
        
        # Apply normalization to market data
        bar_data = self.normalize_data(bar_data)
        
        print(f"type of index: {bar_data.index[0]}")
        
        # Get only numeric columns from bar data and convert to float32
        numeric_columns = bar_data.select_dtypes(include=[np.number]).columns
        bar_data = bar_data[numeric_columns].astype(np.float32)
        
        # # Get sector and industry information
        # sector, industry = self.get_symbol_sector_industry(symbol)
        # one_hot_features = self.create_one_hot_features(sector, industry)
        # print(f"number of one-hot features: {len(one_hot_features.columns)}")
        
        # # Create a DataFrame with the one-hot features repeated for each date
        # one_hot_df = pd.DataFrame(
        #     np.tile(one_hot_features.values, (len(bar_data), 1)),
        #     index=bar_data.index,
        #     columns=one_hot_features.index
        # )
        
        # Concatenate bar_data, fred_data, and one-hot features
        bar_data = pd.concat([bar_data, self.fred_data], axis=1).dropna()
        
        data = {
            'bar_data': bar_data,
            'mock_trades': mock_trades
        }
        
        return data
        
    @staticmethod
    def prepare_state(row: pd.Series, position: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """Convert raw data row and position info into state vector and return raw close price"""
        # Get market data features excluding raw_close
        numeric_features = row[:-1].values.astype(np.float32)  # Exclude the last column (raw_close)
        raw_close = float(row['raw_close'])
        
        # Add position information
        position_features = np.array([
            float(position[0]),  # current shares
            float(position[1]),  # average price
        ], dtype=np.float32)
        
        # Concatenate features and ensure the result is float32
        state = np.concatenate([numeric_features, position_features]).astype(np.float32)
        
        return state, raw_close