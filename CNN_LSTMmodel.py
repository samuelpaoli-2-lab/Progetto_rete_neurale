import torch
from torch import nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 4
        self.n_hidden = 64
        self.n_classes = 5
        self.drop_prob = 0.5
        self.n_input = 64
        self.batch_size=64

        self.CNN=nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm1=nn.LSTM(self.n_input, int(self.n_hidden/2), self.n_layers, bidirectional=True, dropout=self.drop_prob, batch_first=True)
        
        self.fc=nn.Linear(self.n_hidden, self.n_classes)
    def forward(self, x):
        x=self.CNN(x)
        x=x.permute(0,2,1)
        x, (hn, cn)=self.lstm1(x)
        out_forward=hn[-2, :, :]
        out_backward=hn[-1, :, :]
        out=torch.cat((out_forward, out_backward), dim=1)
        out=self.fc(out)
        return out
        