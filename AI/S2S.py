import torch
import torch.nn as nn # PyTorch neural network module
import random

bidirectional = False
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        # self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        # self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x): # assumes that x is of shape (batch_size,time_steps, features) 
        _, (hidden, cell) = self.lstm(x) #.float() 
        # print("hidden.shape: ",hidden.shape)
        # print("cell.shape: ",cell.shape)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ouptut_size = output_size

        if num_layers == 1:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size) # hidden size *2 because we are using bidirectional model

        # hidden = self.fc_hidden(torch)

    def forward(self, x, hidden, cell): # assumes that x is of shape (batch_size,1 (time_step), output_features) 
        output, (hidden, cell) = self.lstm(x, (hidden, cell)) #.float()
        # output: (N, 1, hidden size)
        prediction = self.fc(output)
        return prediction,  hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_window, dropout, device):
        super().__init__()
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout).to(device)
        self.decoder = Decoder(output_size, hidden_size, num_layers, output_size, dropout).to(device)
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.prediction_window = prediction_window
        # self.bn1 = nn.BatchNorm1d(1)
        
        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    # use teacher forcing ratio to balance between using predicted result vs. true result in generating next prediction
    def forward(self, input, target, teacher_forcing_ratio = 0.5):
        batch_size = input.shape[0]

        hidden, cell = self.encoder(input)
        # print("hidden.shape: ",hidden.shape)
        # expected: ?????(batch_size, hidden_size)

        outputs = torch.zeros(batch_size, self.prediction_window, self.output_size).to(self.device)
        x = target[:,0:1,None] # x at timestamp 

        for t in range (self.prediction_window):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:,t:t+1,:] = output
            x = target[:,t:t+1,None] if random.random() < teacher_forcing_ratio else output
        
        return outputs.squeeze(2) # note that squeeze is used since y_batch is 2d, yet y_pred is 3d. (if output size sin't 1, then y_batch will be 3d.)            
