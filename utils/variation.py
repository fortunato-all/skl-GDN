import os
import torch
import dgl.function as fn
from dgl.data import FraudYelpDataset, FraudAmazonDataset
import dgl
import numpy as np
import argparse
from dgl.data.utils import load_graphs, save_graphs

class Dataset:
    def __init__(self, name, homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'tsocial':
            graph, label_dict = load_graphs('data/tsocial')
            graph = graph[0]
        elif name == 'tfinance':
            graph, label_dict = load_graphs('/data/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        self.graph = graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelp",help="yelp/amazon/tfinance/tsocial")
    args = parser.parse_args()
    dataset_name = args.dataset
    homo = args.homo
    feat = torch.Tensor(np.load(r'data/'+dataset_name+ '_feat_s4.npy')) # save the top or else feature at model_handler.py
    graph = Dataset(dataset_name, homo).graph
    
    D_invsqrt = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5).unsqueeze(-1)
    graph.ndata['h'] = feat * D_invsqrt
    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    variation = feat - graph.ndata.pop('h') * D_invsqrt
    variation1 = np.linalg.norm(variation)
    print(variation)
    print(variation1)
