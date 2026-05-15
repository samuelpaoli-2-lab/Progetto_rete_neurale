import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from typing import Tuple

class Mitbih_dataloader(object):
    def __init__(self, training_path: str, test_path: str):
        self.train_filepath=training_path
        self.test_filepath=test_path

    def read_csv_data(self, file_path: str):
        df=pd.read_csv(file_path, header=None)

        x_data=df.iloc[:, :-1].values.tolist()

        y_data=df.iloc[:,-1].values.tolist()

        return x_data, y_data
    
    def load_data(self)  -> Tuple[Tuple[list,list], Tuple[list, list]]:
        x_train, y_train=self.read_csv_data(self.train_filepath)
        x_test, y_test=self.read_csv_data(self.test_filepath)

        return (x_train, y_train), (x_test, y_test)
    
def preprocess_data(x, y):
    x_tensor=torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(1)
    y_tensor=torch.tensor(np.array(y), dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)