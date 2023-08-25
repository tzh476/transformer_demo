import torch

import torch.nn as nn

import math

from torch.autograd import Variable


class Embeddings(nn.Module):
    # d_model词嵌入的维度，vocab词表的大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)

        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embeddings(d_model,vocab)
embr = emb(x)
print("embr:", embr)
print(embr.shape)