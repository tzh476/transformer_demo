import torch

import torch.nn as nn

import math

from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    # d_model词嵌入的维度，dropout置0的比率(让神经网络中的神经失效)，max_len:每个句子的最大长度
    def __init__(self, d_model, dropout,max_len=5000):
        super(PositionalEncoding, self).__init__()

        #
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,d_model)

        position = torch.arange(0,max_len).unsqueeze(1)


    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embeddings(d_model,vocab)
embr = emb(x)
print("embr:", embr)
print(embr.shape)