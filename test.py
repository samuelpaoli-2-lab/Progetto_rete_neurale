from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model_1d_conv import conv_1d_net 
from LSTMmodel import LSTMmodel 
from mitbih_dataset import Mitbih_dataloader, preprocess_data
from sklearn.metrics import classification_report

import os

base_path=Path(__file__).resolve().parent
training_filepath=base_path/"mitbih_train.csv"
test_filepath=base_path/"mitbih_test.csv"

mitbih_dataloader=Mitbih_dataloader(training_filepath, test_filepath)
_, (x_test, y_test)=mitbih_dataloader.load_data()


test_dataset=preprocess_data(x_test, y_test)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=LSTMmodel().to(device)
model_filepath=base_path/"mitbih_LSTM.pth"#"mitbih_weights.pth"
model.load_state_dict(torch.load(model_filepath, map_location=device))
model.eval()

correct=0
total=0
all_preds=[]
all_labs=[]

with torch.no_grad():
    for x, y in test_loader:
        x,y=x.to(device), y.to(device)
        outputs=model(x)
        _, predicted=torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labs.extend(y.cpu().numpy())


print("Accuracy", 100*correct/total)

classi_ecg = ['N', 'S', 'V', 'F', 'Q']

cm = confusion_matrix(all_labs, all_preds)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi_ecg)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f'Confusion Matrix\nAccuracy: {100 * correct / total:.2f}%')
plt.savefig(os.path.join('Confusion_matrix', 'cm_LSTM.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

cm_percent = confusion_matrix(all_labs, all_preds, normalize='true')


# 2. Genera il display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=classi_ecg)

# 3. Disegna il plot. values_format='.2%' trasforma i decimali (es. 0.98) in 98.00%
disp.plot(cmap=plt.cm.Blues, values_format='.2%')
plt.title('Matrice di Confusione Normalizzata (Recall)')
plt.savefig(os.path.join('Confusion_matrix', 'cm_LSTM_norm.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n--- REPORT DI CLASSIFICAZIONE ---")
# Passiamo le tue etichette personalizzate per rendere la tabella leggibile
print(classification_report(all_labs, all_preds, target_names=classi_ecg))
