# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer

if __name__ == "__main__":
    # enc_inputs 对应['我 是 学 生 P'] ['我 喜 欢 学 习'] ['我 是 男 生 P']
    # [1, 2, 3, 4, 0],\n[1, 5, 6, 3, 7],\n[1, 2, 8, 4, 0]
    # dec_inputs 对应 ' S I  am  a student'       S I like learning P      S I am a boy
    # [[1, 3, 4, 5, 6],\n        [1, 3, 7, 8, 0],\n        [1, 3, 4, 5, 9]]
    # dec_outputs 对应 'I am a student E'        'I like learning P E'       I  am a boy E
    # [[3, 4, 5, 6, 2],\n        [3, 7, 8, 0, 2],\n        [3, 4, 5, 9, 2]]
    enc_inputs, dec_inputs, dec_outputs = make_data()
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'model.pth')
    print("保存模型")
