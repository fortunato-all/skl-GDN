import torch
import pickle
import random
import os.path as osp
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as c
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from collections import defaultdict
import logging
from dgl.data.utils import load_graphs, save_graphs
import dgl
from sklearn.model_selection import train_test_split
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.distributed import DistGraph, DistDataLoader, node_split
from deeprobust.graph.global_attack import BaseAttack
from dgl import node_subgraph
# import sys
# sys.path.append("/home/lijh/wsdm_GDN_attack/utils")
from utils import sparse_to_adjlist
# from torch_geometric.utils import from_scipy_sparse_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class DICE(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cuda'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure,
                                   attack_features=attack_features, device=device)

        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_adj, labels, n_perturbations, index_target, **kwargs):
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        indices = sp.triu(modified_adj).nonzero()
        # Remove edges of the same label
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]] == labels[x[1]] and x[0] in index_target or x[1] in index_target]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T

            # filter out existing edges, and pairs with the different labels
            # source node or target node are in the target_index set
            candidate_edges = set([(u, v) for u, v in candidate_edges if labels[u] != labels[v]
                                   and modified_adj[u, v] == 0 and modified_adj[v, u] == 0 and (
                                               u in index_target or v in index_target)])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj

prefix = 'data/' 
data_name = 'amazon'
#yelp amazon
attack_ratio = 0.1
# -0.4
# -0.8
# -1.2
# -1.6 so large for yelp

# For amazon
dataset = FraudAmazonDataset()
graph = dataset[0]
labels = graph.ndata['label'].numpy()
net_upu = graph[('user', 'net_upu', 'user')]
net_usu = graph[('user', 'net_usu', 'user')]
net_uvu = graph[('user', 'net_uvu', 'user')]

# For yelp
# dataset = FraudYelpDataset()
# graph = dataset[0]
# labels = graph.ndata['label'].numpy()
# net_rsr = graph[('review', 'net_rsr', 'review')]
# net_rtr = graph[('review', 'net_rtr', 'review')]
# net_rur = graph[('review', 'net_rur', 'review')]

sub_graph = net_upu # change for different relations
sub_adj = sub_graph.adj()

row = (sub_adj._indices()[0]).numpy()
col = (sub_adj._indices()[1]).numpy()
data = (sub_adj._values()).numpy()
adj = sp.coo_matrix((data,(row,col)),shape=(graph.number_of_nodes(),graph.number_of_nodes()))
adj = adj.tocsr()

idx_test = np.load('data/amazon_idx_test.npy') #save the test idx of amazon or yelp at the modle_handler.py
idx_test = idx_test.tolist()

num_edge = sub_graph.number_of_edges() // 2
n_edge_mod = int(attack_ratio * num_edge)

adj_name = "_".join(["DICE", data_name, str(sub_graph.etypes), str(int(attack_ratio * 100))]) + ".pt"
adjlist_name = "_".join([data_name, str(sub_graph.etypes), str(int(attack_ratio * 100)), "adjlists"]) + ".pickle"
adj_path = osp.join("data/", adj_name)
adjlist_path = osp.join("data/", adjlist_name)

atk_model = DICE()
atk_model.attack(adj, labels, n_perturbations=n_edge_mod, index_target=idx_test)
modified_adj = atk_model.modified_adj
torch.save(modified_adj, adj_path)
modified_adj = modified_adj.tocsc()
sparse_to_adjlist(modified_adj, adjlist_path)
