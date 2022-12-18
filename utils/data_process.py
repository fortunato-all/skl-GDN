from utils.utils import sparse_to_adjlist
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from utils import sparse_to_adjlist
import scipy.sparse as sp
from dgl.data.utils import load_graphs, save_graphs
import torch
import numpy as np

"""
	Read data and save the adjacency matrices to adjacency lists
"""


if __name__ == "__main__":

	prefix = 'data/'

	# 处理tfinance
	# graph, label_dict = load_graphs(r'{prefix}tfinance')
	# tfinance = graph[0].adj()
	# row = (tfinance._indices()[0]).numpy()
	# col = (tfinance._indices()[1]).numpy()
	# data = tfinance._values().numpy()
	# t1 = sp.coo_matrix((data, (row, col)), shape=(39357, 39357))
	# t1 = t1.tocsc()
	# sparse_to_adjlist(t1, r'{prefix}tfinance.pickle')

	# yelp = loadmat('../data/YelpChi.mat')
	# net_rur = yelp['net_rur']
	# net_rtr = yelp['net_rtr']
	# net_rsr = yelp['net_rsr']
	# yelp_homo = yelp['homo']

	# sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	# sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	# sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	# sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

	# amz = loadmat('data/Amazon.mat')
	# net_upu = amz['net_upu']
	# net_usu = amz['net_usu']
	# net_uvu = amz['net_uvu']
	# amz_homo = amz['homo']

	# sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	# sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	# sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	# sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')


	# values = a.data
	# indices = np.vstack((a.row, a.col))
	# i = torch.LongTensor(a.indices)
	# i = torch.LongTensor(a.indptr)
	# v = torch.FloatTensor(values)
	# shape = a.shape
	# a = torch.sparse.FloatTensor(i, v, torch.Size(shape))
