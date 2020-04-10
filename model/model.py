import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNLSTMModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        #### Data Dependent ####
        # input
        self.input_dim = input_dim
        # output
        self.output_dim = output_dim
        ########################

        #### Architecture Dependent ####
        # convolution
        self.conv_channels = [20, 20, 20]
        self.dilation = [1, 2, 4]
        self.kernel_sizes = [3, 3, 3]
        self.num_conv = len(self.conv_channels)
        assert(len(self.conv_channels) == len(self.dilation) == len(self.kernel_sizes)) # ensure right amount of conv channel size is supplied

        self.conv_channels.insert(0, self.input_dim)
        self.conv_channels = list(zip(self.conv_channels, self.conv_channels[1:])) # convert conv_channels into a list of tuples for easier use

        # rnn
        self.rnn1_input = self.conv_channels[-1][-1]
        self.rnn1_hidden = 20
        self.rnn1_num_layers = 2
        ################################

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_c, out_channels=out_c, dilation=dilation, kernel_size=k_size)
                                    for (in_c, out_c), dilation, k_size in zip(self.conv_channels, self.dilation, self.kernel_sizes)])
        self.rnn1 = nn.LSTM(input_size=self.rnn1_input, hidden_size=self.rnn1_hidden, num_layers=self.rnn1_num_layers)
        self.fc = nn.Linear(self.rnn1_hidden, self.output_dim)

        # figure out how to use LSTM later
        # self.lstm1 = nn.LSTM(input_size=1, hidden_size=3, num_layers=2)
        # self.h0 = torch.zeros(2, 1, 3)
        # self.c0 = torch.zeros(2, 1, 3)

    def forward(self, x):
        # print("input x.shape: {}".format(x.shape))
        # if x.dim() == 2: # currently only works if x is single series input (can be batched)
        #     x = x.view(x.shape[0], 1, x.shape[1]) # batch, channel, sequence_length
        # print("after view x.shape: {}".format(x.shape))
        for conv in self.convs:
            x = F.relu(conv(x))
            # print("after conv x.shape: {}".format(x.shape))
        x = x.permute(2, 0, 1) # RNN expects seq_len, batch, channel
        x, _ = self.rnn1(x) # RNN outputs seq_len, batch, channel
        # print("after rnn x.shape: {}".format(x.shape))
        x = self.fc(x[-1, :, :]) # only use the output of the last time step to make our prediction
        # print("after fc x.shape: {}".format(x.shape))
        return x

if __name__ == "__main__":
    # quick tester
    conv1 = nn.Conv1d(1, 1, kernel_size=3)
    rnn1 = nn.RNN(1,3,1)
    fc = nn.Linear(3, 1)
    t = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float)
    t = t.view(1, 1, -1)
    o = conv1(t)
    print(o)
    print(o.shape)
    o = o.view(1, -1, 1)
    o2, hidden = rnn1(o)
    print(o2)
    print(o2.shape)
    print(hidden)
    print(hidden.shape)
    o3 = fc(o2[-1,-1])
    print(o3)
    print(o3.shape)