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
torch.manual_seed(0)
rnn1_hidden = 11
output_size = 1
fc = nn.Linear(rnn1_hidden, output_size)
x = torch.randn(6, 2, 11)
print(x.shape)
print(x)
o = fc(x[-1, :, :])
print(o.shape)
print(o)

print("modify x")
x[5][1][0] = 1000.0
x[5][1][1] = 1000.0
o = fc(x[-1, :, :])
print(o.shape)
print(o)