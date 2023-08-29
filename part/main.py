from data import *
import torch.utils.data as Data
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    enc_inputs, dec_inputs, dec_outputs = make_data()
    # 把数据转换成batch大小为2的分组数据，3句话一共可以分成两组，一组2句话、一组1句话
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    # SGD(Stochastic Gradient Descent)随机梯度下降，lr:学习率,momentum:动量因子
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1)) # 损失函数（loss function）是用来估量模型的预测值f(x)与真实值Y的不一致程度
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'model.pth')
    print("保存模型")
