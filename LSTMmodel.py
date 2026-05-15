import torch
from torch import nn


class LSTMmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 3
        self.n_hidden = 64
        self.n_classes = 5
        self.drop_prob = 0.5
        self.n_input = 1
        self.batch_size=64

        self.lstm=nn.LSTM(self.n_input, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.fc=nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        x=x.permute(0,2,1)
        x, _=self.lstm(x)
        out=x[:, -1, :]
        out=out.contiguous().view(-1, self.n_hidden)
        out=self.fc(out)

        return out
    
        
