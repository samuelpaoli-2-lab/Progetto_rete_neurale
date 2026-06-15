from pathlib import Path
from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model_1d_conv import conv_1d_net 
from LSTMmodel import LSTMmodel 
from Bidir_LSTMmodel import Bidir_LSTM 
from CNN_LSTMmodel import CNN_LSTM
from mitbih_dataset import Mitbih_dataloader, preprocess_data

base_path=Path(__file__).resolve().parent
training_filepath=base_path/"mitbih_train.csv"
test_filepath=base_path/"mitbih_test.csv"

mitbih_dataloader=Mitbih_dataloader(training_filepath, test_filepath)
(x_train, y_train), (x_test, y_test)=mitbih_dataloader.load_data()

x_tr, x_val, y_tr, y_val=train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

train_dataset=preprocess_data(x_tr, y_tr)
val_dataset=preprocess_data(x_val, y_val)

train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=64, shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=conv_1d_net().to(device)

classi, conteggi=np.unique(y_tr, return_counts=True)
pesi=len(y_tr)/(len(classi)*conteggi)

class_weights=torch.tensor(pesi, dtype=torch.float32).to(device)

loss_function=nn.CrossEntropyLoss(weight=class_weights)
optimizer=optim.Adam(model.parameters(), lr=0.001)

lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

epochs=150
print("Training on", device)

storia_train_loss=[]
storia_val_loss=[]
storia_lr=[]
epoca_migl=1

training_start=time()
min_loss=10000
model_filepath = base_path / "mitbih_lr_CNN_weights.pth"

count=0
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

    avg_train_loss = epoch_loss / len(train_loader)
    model.eval() 
    epoch_val_loss = 0.0

    with torch.no_grad():
        for x,y in val_loader:
            x, y=x.to(device), y.to(device)
            outputs=model(x)
            loss_val=loss_function(outputs, y)
            epoch_val_loss+=loss_val.item()

    msg_save=""
    avg_val_loss = epoch_val_loss / len(val_loader)

    lr_scheduler.step(avg_val_loss)
    storia_lr.append(optimizer.param_groups[0]['lr'])

    if min_loss>avg_val_loss:
        torch.save(model.state_dict(), model_filepath)
        min_loss=avg_val_loss
        msg_save="Modello salvato"
        epoca_migl=epoch+1
        count=0

    count+=1
    storia_train_loss.append(avg_train_loss)
    storia_val_loss.append(avg_val_loss)

    epoch_end=time()
    epoch_time=epoch_end-epoch_start
    print("Epoch: ", epoch+1, " Train loss: ", avg_train_loss, "Validation loss: ", avg_val_loss, " Time: ", int(epoch_time//60), ":", str(int(epoch_time%60)).zfill(2), msg_save)
    if count>10:
        print("Nessun miglioramento nelle ultime 10 epoche")
        print("Addestramento interrotto")
        break

training_end=time()
training_time=training_end-training_start


print(f"Model saved to {model_filepath}")


plt.figure(figsize=(10, 6))

epoche_reali=len(storia_train_loss)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epoche Reali Eseguite')
ax1.set_ylabel('Loss (Cross Entropy)', color='black')
linea_train, = ax1.plot(range(1, epoche_reali + 1), storia_train_loss, label='Train Loss', marker='o', color='tab:blue', linewidth=2)
linea_val, = ax1.plot(range(1, epoche_reali + 1), storia_val_loss, label='Validation Loss', marker='o', color='tab:orange', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(range(1, epoche_reali + 1, 5)) 


ax2 = ax1.twinx() 
color_lr = 'tab:green'
ax2.set_ylabel('Learning Rate', color=color_lr)  
linea_lr, = ax2.plot(range(1, epoche_reali + 1), storia_lr, label='Learning Rate', color=color_lr, linestyle='-.', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_lr)

ax2.set_yscale('log') 

linea_stop = ax1.axvline(x=epoca_migl, color='red', linestyle='--', label=f'Miglior Modello (Epoca {epoca_migl})')

linee = [linea_train, linea_val, linea_lr, linea_stop]
etichette = [l.get_label() for l in linee]
ax1.legend(linee, etichette, loc='upper center')

plt.title('Andamento Loss e Modulazione del Learning Rate\n(CNN con pesi)')
ax1.grid(True, linestyle=':', alpha=0.7)

percorso_grafico = base_path / "loss_lr_CNN_weights.png"
plt.savefig(percorso_grafico, dpi=300, bbox_inches='tight')

print(f"\nGrafico salvato in {percorso_grafico}")


