import torch
from torch import nn
# torch.manual_seed(0)
# i = 3
# h = 2
# l = 1 # irrelavent for the output size
# batch = 2
# seq_len = 5
# rnn = nn.RNN(i, h, l)
# input = torch.randn(seq_len, batch, i) # seq_len, batch, input channel
# h0 = torch.randn(l, batch, h) # layer, channel, hidden
# output, _ = rnn(input, h0)
# print("first output:")
# print(output.shape) # batch, channel, hidden dim
# print(output)
#
# print("\noriginal input:")
# print(input)
#
# print("\nmodified input:")
# input[2][0][0] = 1000.0
# print(input)
#
# print("second output")
# output, _ = rnn(input, h0)
# print(output)
# torch.manual_seed(0)
# rnn1_hidden = 11
# output_size = 1
# fc = nn.Linear(rnn1_hidden, output_size)
# x = torch.randn(6, 2, 11)
# print(x.shape)
# print(x)
# o = fc(x[-1, :, :])
# print(o.shape)
# print(o)
#
# print("modify x")
# x[5][1][0] = 1000.0
# x[5][1][1] = 1000.0
# o = fc(x[-1, :, :])
# print(o.shape)
# print(o)

from sklearn import preprocessing
import numpy as np
x = np.array([1,2,3,4,5])
x2 = np.array([-2,4,6,8,10])
k = np.sign(x) == np.sign(x2)
print(k)
s = np.sum(k)/np.size(k)
print(s)
# k = [x, x2]
# s = np.stack(k, axis=1)
# print(s)
# x = np.stack((x, x2), axis=1)
# print(x)
# # x = x.reshape(-1, 1)
# print(x)
# transformer = preprocessing.StandardScaler()
# x_std = transformer.fit_transform(x)
# print(x_std)
#
# x_ori = transformer.inverse_transform(x_std)
# print(x_ori)

# std_x = preprocessing.scale(x)
# print(std_x)

from sklearn.preprocessing import StandardScaler
import numpy as np
data = np.array([[1,1],[2,2],[3,3]])
print(data)
transformer = StandardScaler()
data_norm = transformer.fit_transform(data)
print(data_norm)

print(transformer.mean_,transformer.var_)

print(data_norm* np.sqrt(transformer.var_)+transformer.mean_)