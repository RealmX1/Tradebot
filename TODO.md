prepare better normalization; the state printed out at data_loader "prepare_state" doesn't seem very well normalized; The close price from the stock
re-check whether the time in the fred data is the update time/observation time, or the time of the month being recorded; The latter might result in look-ahead bias.
add the industry of stock as onehot input; also, upscale the model and check if it's better.
