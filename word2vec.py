import collections
import numpy as np
import torch as t
import torch.nn as nn
from torch import LongTensor as LT
from torch.autograd import Variable as Var


class Word2Vec(nn.Module):
    '''
    count: 单词和其频率,按照从高到低排列
    data: 语料中的单词替换为对应的数字
    '''
    def __init__(self, count = None, data = None, vocab_size = 50000, emb_dim = 100):
        super(Word2Vec,self).__init__()
        self.vcab_size = vocab_size
        self.emb_dim = emb_dim
        self.iemb = nn.Embedding(vocab_size, emb_dim,sparse=True)
        self.oemb = nn.Embedding(vocab_size, emb_dim, sparse=True)
        # init weight
        initrange = 0.5 / self.emb_dim
        self.iemb.weight.data.uniform_(-initrange,initrange)
        self.oemb.weight.data.uniform_(0,0)
        self.count = count
        self.sample_table = self.init_sample_table()
        self.data = data
        self.num_negs = 10
        self.batch_size = 100
        self.data_idx = 0

    def forward(self, iwords, owords):
        logsigmoid = nn.LogSigmoid()
        iembs = self.iemb(Var(LT(iwords)))
        oembs = self.oemb(Var(LT(owords)))
        # 计算得到的向量的得分，用向量相似度来衡量，也就是相乘
        score = t.mul(iembs,oembs).squeeze()
        score = t.sum(score, dim = 1)
        score = logsigmoid(score).squeeze()
        fakewords = np.random.choice(self.sample_table, size=(self.batch_size,self.num_negs))
        neg_oemb = self.oemb(Var(LT(fakewords)))
        # print(neg_oemb,iembs.unsqueeze(2))
        neg_score = t.bmm(neg_oemb,iembs.unsqueeze(2)).squeeze()
        neg_score = t.sum(neg_score, dim = 1)
        neg_score = logsigmoid(-1 * neg_score).squeeze()
        losses = score + neg_score
        return -1*losses.sum()
    
    def init_sample_table(self):
        freq_count = [ele[1] for ele in self.count]
        pow_frequency = np.array(freq_count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency/ power
        table_size = 1e8
        count = np.round(ratio*table_size)
        sample_table = []
        for idx, x in enumerate(freq_count):
            sample_table += [idx]*int(x)
        return np.array(sample_table)

    def gen_batchs(self,batch_size, window):
        '''
        window表示skip-gram的窗口大小
        window=1表示前后各取样一个单词
        '''
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_idx])
            self.data_idx = (self.data_idx + 1) % len(self.data)
        num_skips = 2 * window
        for i in range(batch_size // num_skips):
            for j in range(num_skips):
                batch[i * num_skips + j] = buffer[window]
            idx = 0
            for j in range(span):
                if j == window:
                    continue
                labels[i * num_skips + idx] = buffer[j]
                idx += 1
            buffer.append(self.data[self.data_idx])
            self.data_idx = (self.data_idx + 1) % len(self.data)
        return batch, labels

