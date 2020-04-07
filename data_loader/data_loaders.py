"""

"""

import os
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader, Dataset, Sampler

import torch
import numpy as np
import pandas as pd

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
    def __init__(self, csv_file, window=10):
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        self.series = torch.tensor(self.df['Adj Close'])
        self.data = []
        self.window = window
        for i in range(0, len(self.series)-self.window):
            self.data.append((self.series[i:i+self.window], self.series[i+self.window]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StockDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = StockDataset("/home/guanyush/Pictures/CSC2516/CNNLSTM/data/STOCK/SPY.csv")
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == "__main__":
    # quick tester
    csv_file = "/home/guanyush/Pictures/CSC2516/CNNLSTM/data/STOCK/SPY.csv"
    # df = pd.read_csv(csv_file)
    # tt = torch.tensor(df['Adj Close'])
    # dummy = 1

    sd = StockDataset(csv_file)
    print(sd[-1])
