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
    def __init__(self):
        super().__init__()
        k = 3
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k) # no padding
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k)
        self.rnn1 = nn.RNN(1, 3, 1)
        self.fc = nn.Linear(3, 1)
        # figure out how to use LSTM later
        # self.lstm1 = nn.LSTM(input_size=1, hidden_size=3, num_layers=2)
        # self.h0 = torch.zeros(2, 1, 3)
        # self.c0 = torch.zeros(2, 1, 3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, _ = self.rnn1(x.view(1, -1, 1))
        x = self.fc(x[-1, -1])
        # x, _ = self.lstm1(x, (self.h0, self.c0))
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