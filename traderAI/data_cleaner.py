import os
import pandas as pd

# Define the path to the folder containing the stock files
data_folder = '/data/'

# Step 1: Read all names of the stock files in data folder that end with "_1min.csv", and combine them to a list
file_list = [f for f in os.listdir(data_folder) if f.endswith('_1min.csv')]

# Step 2: Iterate through the list to fix the issue of missing minutes
for file_name in file_list:
    # Load the CSV file into a Pandas dataframe
    df = pd.read_csv(data_folder + file_name)

    # Convert the UTFtime column to a datetime object
    df['Datetime'] = pd.to_datetime(df['UTFtime'], unit='s')
    df = df.set_index('Datetime')

    # Resample the data to fill in missing minutes with forward filling
    df = df.resample('1min').ffill()

    # Save the fixed data to a new CSV file
    fixed_file_name = data_folder + file_name[:-4] + '_fixed.csv'
    df.to_csv(fixed_file_name, index=False)