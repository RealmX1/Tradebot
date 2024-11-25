n = 3  # number of top symbols to return

info_dict = {
    'AAPL': {'weighted_prediction': 0.2, 'price': 100},
    'BABA': {'weighted_prediction': 0.3, 'price': 100},
    'TSLA': {'weighted_prediction': 0.4, 'price': 100},
    'GOOG': {'weighted_prediction': 0.5, 'price': 100},
}
# sort the info_dict by weighted_prediction value of each info dict
sorted_info = sorted(info_dict.items(), key=lambda x: x[1]['weighted_prediction'], reverse=True)

# get the top n symbols with their corresponding info
top_symbols = [(symbol, info) for symbol, info in sorted_info[:n]]
for symbol, info in top_symbols:
    print(symbol, info)