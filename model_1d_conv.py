import torch.nn as nn
import torch.nn.functional as F

class conv_1d_net(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()

        self.features=nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(2752, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x=self.features(x)
        x=x.view(x.size(0), -1)
        out=self.classifier(x)

        return out