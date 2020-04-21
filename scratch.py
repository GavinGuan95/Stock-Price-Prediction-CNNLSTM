import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("/home/guanyush/Pictures/CSC2516/CNNLSTM/data_loader/processed_data/SPY_processed.csv", index_col="Date")
# df = df.cumsum()
# # plt.figure()
# df.plot()
# plt.savefig("/home/guanyush/Pictures/CSC2516/CNNLSTM/testplot")
# # plt.show()



class ReturnPlotter:
    def __init__(self, index_array, list_of_np_array, list_of_name, title=None):

        self.index_array = index_array
        self.list_of_pd_series = []
        for np_array in list_of_np_array:
            assert(len(self.index_array) == len(np_array))
            self.list_of_pd_series.append(pd.Series(np_array, index=index_array))
        self.list_of_name = list_of_name
        self.title = title

        self.df = pd.concat(self.list_of_pd_series, axis=1)
        self.return_plot()
        dummy = 1

    def return_plot(self, save_path=None):
        self.df.plot()
        plt.title(self.title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


data_index = df.index
data_1 = df["EMA_2_3"].to_numpy()
data_2 = df["MA_2_3"].to_numpy()
ReturnPlotter(data_index, [data_1, data_2], ["Date", "EMA_2_3", "MA_2_3"], "random title")
dummy = 1