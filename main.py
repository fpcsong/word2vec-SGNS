import utils
from word2vec import Word2Vec
import os
import torch as t
import numpy as np
from torch.autograd import Variable as Var
from torch import LongTensor as LT
from sklearn.manifold import TSNE


data, count, dicts, rev_dicts = utils.build_dataset()
model = Word2Vec(count,data)

if not os.path.exists('model.st'):
    utils.train(model,'model.st')

model.load_state_dict(t.load('model.st'))

# 画图展示
plot_num = 300
idxs = np.arange(0,plot_num,1)
results = model.iemb(Var(LT(idxs))).data.numpy()
labels = [rev_dicts[i] for i in range(plot_num)]
tsne = TSNE(n_components=2,init = 'pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(results[:plot_num,:])
utils.plot_with_labels(low_dim_embs,labels)

