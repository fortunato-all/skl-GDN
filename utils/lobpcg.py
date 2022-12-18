import torch
import numpy as np
from utils import sparse_to_adjlist
from scipy.io import loadmat
from utils import sparse_to_adjlist
import scipy.sparse as sp
from dgl.data.utils import load_graphs, save_graphs
import gc
import time
import pandas as pd
torch.cuda.set_device(2)



dataset = 'Amaon_net_upu'
# Amaon: net_upu net_usu net_uvu
item_size = 11944
# tfinance
# item_size = 39357
# Yelp: net_rur net_rsr net_rtr
# item_size = 45954

smooth_ratio = 0.001
rough_ratio = 0



def cal_spectral_feature(Adj, size, largest=True, niter=2):
	# params for the function lobpcg
	# k: the number of required features
	# largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
	# niter: maximum number of iterations
	# for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html

	value,vector=torch.lobpcg(Adj,k=size,largest=largest,niter=niter)


	if largest==True:
		feature_file_name=dataset+'_smooth_features.npy'
		value_file_name=dataset+'_smooth_values.npy'

	else:
		feature_file_name=dataset+'_rough_features.npy'
		value_file_name=dataset+'_rough_values.npy'


	np.save(r'./data/'+value_file_name,value.cpu().numpy())
	np.save(r'./data/'+feature_file_name,vector.cpu().numpy())


prefix = 'data/'

# graph, label_dict = load_graphs(r'data/tfinance')
# tfinance = graph[0].adj() #生成tensor矩阵，到这儿就可以用lobpcg了

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
#
# sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
# sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
# sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
# sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

amz = loadmat('data/Amazon.mat')
net_upu = amz['net_upu']
# a = net_upu
# _add_sparse(a, a.t())
# a = net_upu+(torch.t(net_upu))
# net_usu = amz['net_usu']
# net_uvu = amz['net_uvu']
# amz_homo = amz['homo']

# sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
# sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
# sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
# sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')


a = net_upu
a = torch.from_numpy(a.toarray())

#smoothed feautes for item-item relations
# cal_spectral_feature(tfinance, int(smooth_ratio*item_size), largest=True)
cal_spectral_feature(a,int(smooth_ratio*item_size), largest=True)

#rough feautes for item-item relations
if rough_ratio!=0:
    # cal_spectral_feature(tfinance, int(rough_ratio*item_size), largest=False)
	cal_spectral_feature(a,int(smooth_ratio*item_size), largest=False)

