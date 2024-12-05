import pandas as pd
import numpy as np
from fredapi import Fred
import os
import ast
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_path = os.path.join('data', 'fred')
# create the data path if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

class FredDataProcessor:
    def __init__(self):
        # Initialize FRED API
        fred_key_path = os.path.join(os.path.dirname(__file__), 'fred.key')
        with open(fred_key_path, 'r') as f:
            fred_key = f.read().strip()
        self.fred = Fred(api_key=fred_key)
        
    def load_series_config(self):
        """Load the series configuration from CSV"""
        config_path = os.path.join(os.path.dirname(__file__), 'fred_series.csv')
        df = pd.read_csv(config_path)
        df['normalize'] = df['normalize'].apply(ast.literal_eval)
        return df
    
    def calculate_pct_changes(self, series):
        """Calculate percentage changes for different time periods"""
        changes = pd.DataFrame(index=series.index)
        # Monthly change
        changes['pct_1m'] = series.pct_change()
        # Quarterly change
        changes['pct_3m'] = series.pct_change(periods=3)
        # Yearly change
        changes['pct_12m'] = series.pct_change(periods=12)
        return changes
    
    def normalize_series(self, series, normalize_config):
        """Apply normalization steps to the series"""
        normalized_series = {}
        
        for norm_step in normalize_config:
            if isinstance(norm_step, tuple):
                print(norm_step)
                method, params = norm_step
                if method == 'center':
                    min_val, max_val = params
                    normalized = (series - min_val) / (max_val - min_val)
                    normalized_series[f'normalized_center'] = normalized
                elif method == 'pct_changes':
                    changes = self.calculate_pct_changes(series)
                    normalized_series.update(changes)
                    
        return pd.DataFrame(normalized_series)
    
    def process_series(self):
        """Process all series from the configuration"""
        config_df = self.load_series_config()
        raw_series = {}
        normalized_series = {}
        
        for _, row in config_df.iterrows():
            logger.info(f"Processing series: {row['series_id']} - {row['name']}")
            
            # Get the raw series data
            series = self.fred.get_series(row['series_id'])
            
            # Fill missing values with forward fill then backward fill
            series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Store the original series
            raw_series[f"{row['series_id']}_raw"] = series
            
            # Apply normalizations
            normalized = self.normalize_series(series, row['normalize'])
            print(normalized.head())
            
            # Add normalized columns to the result
            for col in normalized.columns:
                normalized_series[f"{row['series_id']}_{col}"] = normalized[col]
        
        # Combine all series into a single DataFrame
        raw_series_df = pd.DataFrame(raw_series)
        result_df = pd.DataFrame(normalized_series)
        # Remove any row that has a NaN value
        result_df = result_df.dropna()
        raw_series_df = raw_series_df.dropna()
        
        # Save the processed data
        raw_path = os.path.join(data_path, 'raw_fred_data.csv')
        raw_series_df.to_csv(raw_path)
        
        normalized_path = os.path.join(data_path, 'normalized_fred_data.csv')
        result_df.to_csv(normalized_path)
        
        logger.info(f"Processed data saved to {normalized_path}")
        
        return result_df

if __name__ == "__main__":
    processor = FredDataProcessor()
    processed_data = processor.process_series()
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
    print("\nColumns in processed data:")
    print(processed_data.columns.tolist()) 