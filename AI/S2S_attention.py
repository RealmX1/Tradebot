import torch.nn as nn # PyTorch neural network module

bidirectional = False
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        # self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        # self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x): # assumes that x is of shape (batch_size,time_steps, features) 
        encoder_states, (hidden, cell) = self.lstm(x) #.float() 
        # print("hidden.shape: ",hidden.shape)
        # print("cell.shape: ",cell.shape)
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size) # hidden size *2 because we are using bidirectional model

        # hidden = self.fc_hidden(torch)

    def forward(self, x, hidden, cell): # assumes that x is of shape (batch_size,1 (time_step), output_features) 
        output, (hidden, cell) = self.lstm(x, (hidden, cell)) #.float()
        # output: (N, 1, hidden size)
        prediction = self.fc(output)
        return prediction,  hidden, cell