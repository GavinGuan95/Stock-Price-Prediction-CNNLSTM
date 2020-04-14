"""

"""

import os
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader, Dataset, Sampler

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# the parameters from the json file will be fed into here.
# class StockDataset(Dataset):
#     def __init__(self, csv_file, window=10):
#         self.csv_file = csv_file
#         self.df = pd.read_csv(self.csv_file)
#         self.df = self.df.dropna()  # calculation of MA and EMA will result in some columns missing
#
#         # columns = ['Adj Close', 'Close']
#         columns = ['MA_3', 'EMA_3']
#         total_len = len(self.df[columns[0]])
#         self.tensor_list = []
#         for column in columns:
#             self.tensor_list.append(torch.tensor(self.df[column].to_numpy(), dtype=torch.float))
#         self.stacked_series = torch.stack(self.tensor_list, dim=0)
#
#         self.window = window
#         self.data = []
#         for i in range(0, total_len-self.window):
#             self.data.append((self.stacked_series[:,i:i+self.window], self.stacked_series[:,i+self.window]))
#         dummy = 1
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]

class StockDataset(Dataset):
    def __init__(self, torch_matrix, window=7):
        total_len = list(torch_matrix.shape)[1]

        self.window = window
        self.data = []
        for i in range(0, total_len-self.window):
            self.data.append((torch_matrix[:,i:i+self.window], torch_matrix[:,i+self.window]))
        dummy = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StockDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        self.transformer = None
        torch_matrix = self.normalization(data_dir)
        self.dataset = StockDataset(torch_matrix)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def normalization(self, csv_file):
        df = pd.read_csv(csv_file).dropna()
        # needs to add one more entry ROC_1
        columns = ['MA_2_3', 'EMA_2_3']
        np_array_list = []
        for column in columns:
            np_array_list.append(df[column].to_numpy())
        np_matrix = np.stack(np_array_list, axis=1)
        self.transformer = StandardScaler()
        np_matrix_normalized = self.transformer.fit_transform(np_matrix)
        # save the mean and variance to file

        np.savez("norm_para", mean=self.transformer.mean_, std=np.sqrt(self.transformer.var_))

        np_matrix_original = self.transformer.inverse_transform(np_matrix_normalized)
        torch_matrix = torch.tensor(np_matrix_normalized, dtype=torch.float).t()
        return torch_matrix


if __name__ == "__main__":
    # quick tester
    csv_file = "/home/guanyush/Pictures/CSC2516/CNNLSTM/data/STOCK/SPY.csv"
    # df = pd.read_csv(csv_file)
    # tt = torch.tensor(df['Adj Close'])
    # dummy = 1

    sd = StockDataset(csv_file)
    print(sd[-1])
