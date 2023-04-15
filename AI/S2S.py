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
            self.dropout = 0
        else:
            self.dropout = dropout
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional,dropout = self.dropout)
        # self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        # self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x): # assumes that x is of shape (batch_size,time_steps, features) 
        encoder_states, (hidden, cell) = self.lstm(x) #.float() 
        # print("hidden.shape: ",hidden.shape)
        # print("cell.shape: ",cell.shape)
        # print("encoder_states.shape: ",encoder_states.shape)
        # encoder_states: (batch_size, seq_length, hidden_size)
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = input_size
        self.hidden_size = hidden_size

        if bidirectional:
            self.lstm_input_size = self.hidden_size*2 + self.embedding_size
        else:
            self.lstm_input_size = self.hidden_size + self.embedding_size

        self.num_layers = num_layers
        self.ouptut_size = output_size

        self.relu = nn.ReLU()
        self.energy = nn.Linear(hidden_size*2,1)

        if num_layers == 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional, dropout = self.dropout)

        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size) # hidden size *2 because we are using bidirectional model

        # hidden = self.fc_hidden(torch)

    def forward(self, x, encoder_states, hidden, cell): # assumes that x is of shape (batch_size,1 (time_step), output_features) 
        seq_len = encoder_states.shape[1]
        # print("hidden: ", hidden.shape)
        # hidden: (num_layers, batch_size, hidden_size)
        hidden_4 = hidden.unsqueeze(2)
        h_reshaped = hidden_4.repeat(1, 1, seq_len, 1)
        # print("h_reshaped: ", h_reshaped.shape)
        # h_reshaped:       (num_layers, batch_size, sig_len, hidden_size)

        # torch.cat([h_reshaped[-1,:,:,:], encoder_states], dim=2)
        # (batch_size, seq_length, hidden_size*2)
        energy = self.relu(self.energy(torch.cat([h_reshaped[-1,:,:,:], encoder_states], dim=2))) # only taking the last layer of h_reshaped; is this a good idea?
        # encoder_states:   (batch_size, seq_length, hidden_size)
        # h_reshaped:       (num_layers, batch_size, sig_len, hidden_size)
        # energy:           (batch_size, seq_length, 1)
        # print("energy: ", energy.shape)

        attention = torch.softmax(energy, dim = 1)
        # print("attention: ", attention.shape)
        # attention should be of shape (batch_size, seq_len, 1)

        # print(encoder_states.shape)
        context_vector = torch.bmm(attention.transpose(2,1), encoder_states)
        # encoder_states: (batch_size, seq_len, hidden_size)
        # attention:      (batch_size, seq_len, 1)
        # context_vector: (batch_size, 1, hidden_size)
        # print("context_vector: ", context_vector.shape)

        # v*tanh(hencoder*w1+hdecoder*w2)

        lstm_input = torch.cat([context_vector, x], dim=2)
        # lstm_input: (batch_size, 1, embedding_size + hidden_size)
        # print("lstm_input: ", lstm_input.shape)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell)) #.float()
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

        encoder_states, hidden, cell = self.encoder(input)
        # print("hidden.shape: ",hidden.shape)
        # expected: ?????(batch_size, hidden_size)

        outputs = torch.zeros(batch_size, self.prediction_window, self.output_size).to(self.device)
        x = target[:,0:1,None] # x at timestamp 

        for t in range (self.prediction_window):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[:,t:t+1,:] = output
            x = target[:,t:t+1,None] if random.random() < teacher_forcing_ratio else output
        
        return outputs.squeeze(2) # note that squeeze is used since y_batch is 2d, yet y_pred is 3d. (if output size sin't 1, then y_batch will be 3d.)            
