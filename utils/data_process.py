from utils import sparse_to_adjlist
from scipy.io import loadmat
import scipy.sparse as sp
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch

"""
	Read data and save the adjacency matrices to adjacency lists
"""


if __name__ == "__main__":

	prefix = 'data/'
	
	#For Yelp
	yelp = loadmat('data/YelpChi.mat')
	net_rur = yelp['net_rur']
	net_rtr = yelp['net_rtr']
	net_rsr = yelp['net_rsr']
	yelp_homo = yelp['homo']
	
	sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')
	
	# For Amazon
	# amz = loadmat('data/Amazon.mat')
	# net_upu = amz['net_upu']
	# net_usu = amz['net_usu']
	# net_uvu = amz['net_uvu']
	# amz_homo = amz['homo']

	# sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	# sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	# sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	# sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')

	# For T-Finance and T-Social
	# graph, label_dict = load_graphs('data/tfinance')
	# tfinance = graph[0].adj()
	# row = (tfinance._indices()[0]).numpy()
	# col = (tfinance._indices()[1]).numpy()
	# data = tfinance._values().numpy()
	# t = sp.coo_matrix((data, (row, col)), shape=(39357,39357)) #shape is 5781065 for T-Social
	# t = t.tocsc()
	# sparse_to_adjlist(t, prefix + 'tfinance.pickle')
