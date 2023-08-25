import torch

import torch.nn as nn

import math

class Embeddings(nn.Module):
    # d_model词嵌入的维度，vocab词表的大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)

        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

embedding = nn.Embedding(10, 3)
x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(embedding(x))