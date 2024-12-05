import feature_engineering.indicators as indicators



training_data = indicators.indicate(training=True)
test_data = indicators.indicate(training=False)
