from utils import sparse_to_adjlist
import scipy.sparse as sp
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch

graph, label_dict = load_graphs('../data/tfinance')
tfinance = graph[0].adj()

row = (tfinance._indices()[0]).numpy()
col = (tfinance._indices()[1]).numpy()
data = tfinance._values().numpy()
t1 = sp.coo_matrix((data,(row,col)),shape=(39357,39357))
t1 = t1.tocsc()
sparse_to_adjlist(t1,'../data/tfinance.pickle')

# graph.ndata['label'] = graph.ndata['label'].argmax(1)
# graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
# graph.ndata['feature'] = graph.ndata['feature'].float()
# print(graph[0].formats())
# print(graph[0].ndata)
# print(graph[0].ndata['label'].argmax(1))

# print(graph[0].edges())
# print(graph[0].adj().to_dense())

# class Dataset:
#     def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
#         self.name = name
#         graph = None
#         if name == 'tfinance':
#             graph, label_dict = load_graphs('../data/tfinance')
#             graph = graph[0]
#             graph.ndata['label'] = graph.ndata['label'].argmax(1)
#             print(g)
#
#             if anomaly_std:
#                 graph, label_dict = load_graphs('../data/tfinance')
#                 graph = graph[0]
#                 feat = graph.ndata['feature'].numpy()
#                 anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
#                 feat = (feat-np.average(feat,0)) / np.std(feat,0)
#                 feat[anomaly_id] = anomaly_std * feat[anomaly_id]
#                 graph.ndata['feature'] = torch.tensor(feat)
#                 graph.ndata['label'] = graph.ndata['label'].argmax(1)
#
#             if anomaly_alpha:
#                 graph, label_dict = load_graphs('../data/tfinance')
#                 graph = graph[0]
#                 feat = graph.ndata['feature'].numpy()
#                 anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
#                 normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
#                 label = graph.ndata['label'].argmax(1)
#                 diff = anomaly_alpha * len(label) - len(anomaly_id)
#                 import random
#                 new_id = random.sample(normal_id, int(diff))
#                 # new_id = random.sample(anomaly_id, int(diff))
#                 for idx in new_id:
#                     aid = random.choice(anomaly_id)
#                     # aid = random.choice(normal_id)
#                     feat[idx] = feat[aid]
#                     label[idx] = 1  # 0


        # elif name == 'tsocial':
        #     graph, label_dict = load_graphs('dataset/tsocial')
        #     graph = graph[0]
        #
        # elif name == 'yelp':
        #     dataset = FraudYelpDataset()
        #     graph = dataset[0]
        #     if homo:
        #         graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        #         graph = dgl.add_self_loop(graph)
        # elif name == 'amazon':
        #     dataset = FraudAmazonDataset()
        #     graph = dataset[0]
        #     if homo:
        #         graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        #         graph = dgl.add_self_loop(graph)
        # else:
        #     print('no such dataset')
        #     exit(1)



        # self.graph = graph