import torch.nn as nn
from data import *

d_model = 512   # 字 Embedding 的维度

class Transformer(nn.Module):
    def __int__(self):
        super(Transformer, self).__int__()
        self.Encoder = Encoder.cuda()
        self.Decoer = Encoder.cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
