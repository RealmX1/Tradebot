import pandas as pd
import numpy as np
from typing import Tuple
import glob
import os

class DataLoader:
    def __init__(self, train_dir: str, 
                 fred_data_path: str = "data/fred/normalized_fred_data.csv",
                 normalization_path: str = "feature_engineering/indicator_normalization.csv"):
        self.train_dir = train_dir
        self.fred_data_path = fred_data_path
        self.normalization_path = normalization_path
        
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

    def load_data(self) -> dict:
        """Load and preprocess all bar data and mock trades from the train directory"""
        # Dictionary to store data for each symbol
        all_data = {}
        
        # Get all bar data files in train directory
        bar_files = glob.glob(os.path.join(self.train_dir, "*.csv"))
        
        for bar_file in bar_files:
            # Extract symbol from filename
            filename = os.path.basename(bar_file)
            # Assuming filename format: 20200101_20230101_SYMBOL_1Day_15feature0_IEX.csv
            
            # Construct corresponding mock trade filename
            start_date, end_date, symbol, timeframe, feature_str, exchange = filename.split('_')
            mock_file = os.path.join(self.train_dir, "mock_trade", start_date + "_" + end_date + "_" + symbol + "_" + timeframe + ".csv")
            
            if not os.path.exists(mock_file):
                print(f"Warning: Mock trade file not found for {symbol}, skipping...")
                continue
                
            # Load market data
            bar_data = pd.read_csv(bar_file, index_col=['timestamp', 'symbol'], parse_dates=True)
            mock_trades = pd.read_csv(mock_file, index_col=['timestamp', 'symbol'], parse_dates=True)
            
            # Load FRED data
            fred_data = self.load_fred_data()
            
            # Get only numeric columns from bar data and convert to float32
            numeric_columns = bar_data.select_dtypes(include=[np.number]).columns
            bar_data = bar_data[numeric_columns].astype(np.float32)
            
            # Apply normalization to market data
            bar_data = self.normalize_data(bar_data)
            
            # Align timestamps
            bar_data.index = bar_data.index.map(lambda x: x[0].replace(hour=0, minute=0, second=0))
            mock_trades.index = mock_trades.index.map(lambda x: x[0].replace(hour=0, minute=0, second=0))
            
            # Concatenate bar_data and fred_data
            bar_data = pd.concat([bar_data, fred_data], axis=1).dropna()
            
            # block the raw close column when returning
            
            all_data[symbol] = {
                'bar_data': bar_data,
                'mock_trades': mock_trades
            }
            
        return all_data
        
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