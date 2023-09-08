from datasets import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

from transformer import PositionalEncoding


def maintest(model, enc_input, start_symbol):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    res = [0] * tgt_len
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
        res[i] = next_symbol
    print(res)
    return res


# enc_inputs 对应['我 是 学 生 P'] ['我 喜 欢 学 习'] ['我 是 男 生 P']
# [1, 2, 3, 4, 0],\n[1, 5, 6, 3, 7],\n[1, 2, 8, 4, 0]

# dec_inputs 对应 [' S I  am  a student']       [S I like learning P]      [S I am a boy]
# [[1, 3, 4, 5, 6],\n        [1, 3, 7, 8, 0],\n        [1, 3, 4, 5, 9]]

# dec_outputs 对应 ['I am a student E']        ['I like learning P E']       [I  am a boy E]
# [[3, 4, 5, 6, 2],\n        [3, 7, 8, 0, 2],\n        [3, 4, 5, 9, 2]]
enc_inputs, dec_inputs, dec_outputs = make_data()
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 3, True)
enc_inputs, _, _ = next(iter(loader))
model = torch.load('model.pth')
predict_dec_input = maintest(model, enc_inputs[0].view(1, -1).cuda(), start_symbol=tgt_vocab["S"])


print([src_idx2word[int(i)] for i in enc_inputs[0]], '->',
      [idx2word[n] for n in predict_dec_input])
