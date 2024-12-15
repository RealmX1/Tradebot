import feature_engineering.indicators as indicators

# training_data = indicators.indicate(training=True)
test_data = indicators.indicate(training=False)

from fred_api.prepare_fred_data import FredDataPreper
fred_data_preparer = FredDataPreper()
fred_data_preparer.process_series()
