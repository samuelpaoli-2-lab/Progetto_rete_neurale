from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model_1d_conv import conv_1d_net 
from LSTMmodel import LSTMmodel 
from Bidir_LSTMmodel import Bidir_LSTM 
from mitbih_dataset import Mitbih_dataloader, preprocess_data

base_path=Path(__file__).resolve().parent
training_filepath=base_path/"mitbih_train.csv"
test_filepath=base_path/"mitbih_test.csv"

mitbih_dataloader=Mitbih_dataloader(training_filepath, test_filepath)
(x_train, y_train), (x_test, y_test)=mitbih_dataloader.load_data()

train_dataset=preprocess_data(x_train, y_train)
test_dataset=preprocess_data(x_test, y_test)

train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Bidir_LSTM().to(device)

classi, conteggi=np.unique(y_train, return_counts=True)
pesi=np.sqrt(len(y_train)/(len(classi)*conteggi))

print("Weights: ", pesi, "\n")
pesi[1]+=0.5
pesi[2]+=0.2

class_weights=torch.tensor(pesi, dtype=torch.float32).to(device)

loss_function=nn.CrossEntropyLoss(weight=class_weights)
optimizer=optim.Adam(model.parameters(), lr=0.001)

epochs=5
print("Training on", device)

training_start=time()
min_loss=10000
model_filepath = base_path / "mitbih_LSTM.pth"

for epoch in range(epochs):
    epoch_start=time()
    model.train()
    epoch_loss=0.0
    for x, y in train_loader:
        x, y=x.to(device), y.to(device)
        outputs=model(x)
        loss=loss_function(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    epoch_end=time()
    if (epoch_loss/len(train_loader))<min_loss:
        torch.save(model.state_dict(), model_filepath)
        min_loss=epoch_loss
    epoch_time=epoch_end-epoch_start
    print("Epoch: ", epoch+1, " Loss: ", epoch_loss/len(train_loader), " Time: ", int(epoch_time//60), ":", str(int(epoch_time%60)).zfill(2))

training_end=time()
training_time=training_end-training_start


print(f"Model saved to {model_filepath}")




