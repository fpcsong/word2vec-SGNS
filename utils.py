import os
import zipfile
from urllib.request import urlretrieve

import collections
import tensorflow as tf
import torch.optim as optim
import matplotlib.pyplot as plt
import torch as t
vocab_size = 50000

# 使用tensorflow实战一书中使用的数据集
url = 'http://mattmahoney.net/dc/'
def downloader(filename):
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    print('downloaded file with size: ',statinfo.st_size)


# 留着这个只是保证代码容易使用，我使用uget下载的
#downloader('text8.zip')

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return  data


def build_dataset():
    words = read_data('text8.zip')
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocab_size-1))
    dicts = dict()
    for word,_ in count:
        dicts[word] = len(dicts)
    data = list()
    unk_cnt = 0
    for word in words:
        if dicts.__contains__(word):
            idx = dicts[word]
        else:
            idx = 0
            unk_cnt+=1
        data.append(idx)
    count[0][1] = unk_cnt
    rev_dicts = dict(zip(dicts.values(),dicts.keys()))
    return  data, count, dicts, rev_dicts

def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    plt.figure(figsize=(20,20))
    for i, label in enumerate(labels):
        x,y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                    xy = (x,y),
                    xytext=(5,2),
                    textcoords = 'offset points',
                    ha = 'right',
                    va = 'bottom')
    plt.savefig(filename)

def train(model, savepath = None):
    batch_size = model.batch_size
    optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.5)
    for epoch in range(vocab_size * 2 // batch_size * 20):
        batch, labels = model.gen_batchs(batch_size, window=1)
        # print(batch.size, labels.size)
        model.zero_grad()
        optimizer.zero_grad()
        loss = model(batch, labels)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch, loss)

    if not savepath == None:
        t.save(model.state_dict(), savepath)

