"""

"""

import os
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader, Dataset, Sampler
from data_preprocessor import extract_features

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

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
    def __init__(self, input_torch_matrix, target_torch_matrix, window):
        assert(list(input_torch_matrix.shape)[1] == list(target_torch_matrix.shape)[1])
        total_len = list(input_torch_matrix.shape)[1]

        self.window = window
        self.data = []
        for i in range(0, total_len-self.window):
            # append a tuple: (input, target)
            self.data.append((input_torch_matrix[:, i:i + self.window],
                              target_torch_matrix[:, i + self.window]))
        dummy = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StockDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, input_columns, target_columns, window, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        # the transformer may be needed for transforming later
        self.input_transformer = StandardScaler()
        self.output_transformer = StandardScaler()

        # input_columns = ['MA_2_3', 'EMA_2_3', "ROC_1"]
        # input_columns = ["SMA_2_3", "EMA_2_3"]
        # input_columns = ["MA_2_3", "EMA_2_3",
        #                  "AAPL","AMZN","GE","JNJ","JPM","MSFT","WFC","XOM",
        #                  "AUD=X","CAD=X","CHF=X","CNY=X","EUR=X","GBP=X","JPY=X","NZD=X","usd index",
        #                  "^DJI","SS","^FCHI","^FTSE","^GDAXI","^GSPC","^HSI","^IXIC","^NYA","^RUT"]
        # input_columns = ["ROC_1"]
        # target_columns = ["ROC_1"]

        # call the data preprocessor in here - saved to spy_processed.csv
        extract_features(features_col=input_columns + target_columns,economic=False)
        # saved the input and output columns to

        # change some of the columns if a technical indicator function returns more than one value

        input_torch_matrix = self.normalization(data_dir, input_columns, self.input_transformer, "input", normalization=True)
        target_torch_matrix = self.normalization(data_dir, target_columns, self.output_transformer, "target", normalization=True)
        self.dataset = StockDataset(input_torch_matrix, target_torch_matrix,window)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



    def normalization(self, csv_file, columns, transformer, tag, normalization=True):
        df = pd.read_csv(csv_file).dropna()
        np_array_list = []
        for column in columns:
            np_array_list.append(df[column].to_numpy())
        np_matrix = np.stack(np_array_list, axis=1)
        if normalization:
            np_matrix_normalized = transformer.fit_transform(np_matrix)
        else:
            np_matrix_normalized = np_matrix
        # demonstrate that unnormalization can be done
        # np_matrix_original = self.transformer.inverse_transform(np_matrix_normalized)

        # save the mean and variance to file
        np.savez(tag+"_norm_para", mean=transformer.mean_, std=np.sqrt(transformer.var_))

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
