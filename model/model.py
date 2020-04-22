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
        self.use_conv = True

        # convolution
        self.conv_channels = [20, 20]
        self.dilation = [2, 2]
        self.kernel_sizes = [3, 3]
        self.num_conv = len(self.conv_channels)
        assert(len(self.conv_channels) == len(self.dilation) == len(self.kernel_sizes)) # ensure right amount of conv channel size is supplied

        self.conv_channels.insert(0, self.input_dim)
        self.conv_channels = list(zip(self.conv_channels, self.conv_channels[1:])) # convert conv_channels into a list of tuples for easier use

        # lstm
        if self.use_conv:
            self.lstm1_input = self.conv_channels[-1][-1]
        else:
            self.lstm1_input = input_dim
        self.lstm1_hidden = 20
        self.lstm1_num_layers = 2
        self.lstm1_dropout = 0.1
        ################################

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_c, out_channels=out_c, dilation=dilation, kernel_size=k_size)
                                    for (in_c, out_c), dilation, k_size in zip(self.conv_channels, self.dilation, self.kernel_sizes)])
        self.lstm1 = nn.LSTM(input_size=self.lstm1_input, hidden_size=self.lstm1_hidden, num_layers=self.lstm1_num_layers, dropout=self.lstm1_dropout)
        self.fc = nn.Linear(self.lstm1_hidden, self.output_dim)


    def forward(self, x):
        # print("input x.shape: {}".format(x.shape))
        # if x.dim() == 2: # currently only works if x is single series input (can be batched)
        #     x = x.view(x.shape[0], 1, x.shape[1]) # batch, channel, sequence_length
        # print("after view x.shape: {}".format(x.shape))
        if self.use_conv:
            for conv in self.convs:
                x = F.relu(conv(x))
            # print("after conv x.shape: {}".format(x.shape))

        x = x.permute(2, 0, 1) # LSTM expects seq_len, batch, channel
        x, _ = self.lstm1(x) # LSTM outputs seq_len, batch, channel
        # print("after lstm x.shape: {}".format(x.shape))
        x = self.fc(x[-1, :, :]) # only use the output of the last time step to make our prediction

        # print("after fc x.shape: {}".format(x.shape))
        return x

