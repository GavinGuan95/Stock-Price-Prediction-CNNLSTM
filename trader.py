import torch
import numpy as np
from data_loader.unnormalization import unnormalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure

class Trader:
    tot_earned = 0.0
    reg_ret_list = []

    def __init__(self, config, data_loader):
        self.original_csv_path = config['data_loader']['args']['data_dir']
        self.input_window = data_loader.dataset[0][0].shape[0]  # if there are other ways to find the input window, that is good too

        self.original_df = pd.read_csv(self.original_csv_path).dropna()

        self.trade_period_cur_day = data_loader.sampler.data_source + self.input_window - 1
        self.cur_day_value = self.original_df["Close"].to_numpy()[self.trade_period_cur_day]

        self.trade_period_nxt_day = data_loader.sampler.data_source + self.input_window
        self.nxt_day_value = self.original_df["Close"].to_numpy()[self.trade_period_nxt_day]
        self.date_indices = self.original_df["Date"][self.trade_period_nxt_day]

    def calc_return(self, torch_output_list):
        self.torch_output_list = torch_output_list
        all_output = torch.cat(torch_output_list, dim=0)
        all_output = all_output.cpu().numpy()

        # unnormalize output to their original value
        all_output = unnormalize(all_output)

        self.decision_output = all_output[:, 0]
        assert(len(self.cur_day_value) == len(self.nxt_day_value) == len(self.decision_output))
        self.buy_and_hold_return, self.buy_and_hold_ret_list = self.buy_and_hold()
        reg_return, reg_ret_list = self.trade_with_regression_result()

    def plot_ret(self):
        ReturnPlotter(self.date_indices, [self.buy_and_hold_ret_list, self.reg_ret_list], ["Buy&Hold", "CNNLSTM"], title="Comparison of Return", path="./return_plot.png")

    def buy_and_hold(self):
        ret_list = []
        beginning_price = self.cur_day_value[0]
        for nxt_day in self.nxt_day_value:
            ret_list.append(nxt_day/beginning_price)
        buy_and_hold_return = (self.nxt_day_value[-1] - beginning_price) / beginning_price
        print("buy_and_hold return: {}".format(buy_and_hold_return))
        return buy_and_hold_return, ret_list

    def trade_with_regression_result(self, short=False):
        total_earned = 1.0
        scaling = 100.0
        ret_list = []
        for cur_day, nxt_day, decision in zip(self.cur_day_value, self.nxt_day_value, self.decision_output):
            actual_pct_change = (nxt_day - cur_day)/cur_day
            if short:
                bounded_decision = max(-1.0, min(1.0, decision * scaling))
            else:
                bounded_decision = max(0.0, min(1.0, decision * scaling))
            earned_pct = bounded_decision * actual_pct_change
            total_earned = total_earned * (1 + earned_pct)
            ret_list.append(total_earned)
            # print("total_earned: {}, decision: {}, original_decision: {}, actual: {}, earned_pct: {}".format(total_earned, bounded_decision, decision, actual_pct_change, earned_pct))
        total_earned = total_earned - 1.0
        print("trade_with_regression_result return: {}".format(total_earned))
        if total_earned > self.tot_earned:
            self.tot_earned = total_earned
            self.reg_ret_list = ret_list
        return total_earned, ret_list


class ReturnPlotter:
    def __init__(self, index_array, list_of_np_array, list_of_name, title=None, path=None):

        self.index_array = index_array
        self.list_of_pd_series = []
        for np_array in list_of_np_array:
            assert(len(self.index_array) == len(np_array))
            self.list_of_pd_series.append(pd.Series(np_array, index=index_array))
        self.list_of_name = list_of_name
        self.title = title
        self.path = path

        self.df = pd.concat(self.list_of_pd_series, axis=1)
        self.return_plot(self.path)

    def return_plot(self, save_path=None):
        self.df.plot(figsize=(20, 10))
        plt.title(self.title, fontsize=20)
        plt.legend(self.list_of_name, fontsize=20)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()