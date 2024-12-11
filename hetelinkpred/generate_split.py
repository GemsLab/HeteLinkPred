import torch
import torch_geometric

import os
import dgl
from tqdm import tqdm
from dgl.data.utils import save_graphs
import numpy as np


def split_graph_new_50(g, path='./data', name="", ratio=0.1, neg_per_edge=50, val_test_subratio=None):
    u, v = g.edges()
    edges = torch.stack((u, v), dim=1)
    sorted_edges = torch.sort(edges, dim=1).values
    unique_edges = torch.unique(sorted_edges, dim=0)
    u, v = unique_edges[:, 0], unique_edges[:, 1]

    eids = np.arange(unique_edges.shape[0])
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * ratio)
    val_size = test_size
    trainval_size = unique_edges.shape[0] - test_size

    test_eids = eids[:test_size]
    train_eids = eids[test_size:trainval_size]
    val_eids = eids[trainval_size:]

    if val_test_subratio is not None:
        val_eids = np.random.choice(val_eids, int(val_eids.shape[0] * val_test_subratio), replace=False)
        test_eids = np.random.choice(test_eids, int(test_eids.shape[0] * val_test_subratio), replace=False)
    
    train_pos_u, train_pos_v = torch.cat((u[train_eids], v[train_eids]), dim=0), torch.cat(
        (v[train_eids], u[train_eids]), dim=0)
    val_pos_u, val_pos_v = torch.cat((u[val_eids], v[val_eids]), dim=0), torch.cat(
        (v[val_eids], u[val_eids]), dim=0)
    test_pos_u, test_pos_v = torch.cat((u[test_eids], v[test_eids]), dim=0), torch.cat(
        (v[test_eids], u[test_eids]), dim=0)

    num_negative_edges = neg_per_edge * 100
    sampled_negatives = []
    num_nodes = g.number_of_nodes()
    nodes_to_sample = torch.arange(0, g.number_of_nodes())
    for i in tqdm(range(val_pos_u.shape[0])):
        u = val_pos_u[i]
        v = val_pos_v[i]
        if num_negative_edges > num_nodes:
            num_negatives = nodes_to_sample
        else:
            num_negatives = np.random.choice(
                num_nodes, num_negative_edges, replace=False)
            # num_negatives = np.unique(num_negatives)
            num_negatives = torch.as_tensor(num_negatives, dtype=torch.int64)
        num_negative_edges_filtered = num_negatives.shape[0]
        sampled_negative = nodes_to_sample[num_negatives]
        repeat_u = torch.full((num_negative_edges_filtered, 1), u)[:, 0]
        check_valid = g.has_edges_between(
            repeat_u, sampled_negative)
        choose_valid = check_valid == False
        choose_negative = sampled_negative[choose_valid]
        if torch.sum(choose_valid) >= neg_per_edge:
            tmp_final_neg = np.random.choice(
                choose_negative.numpy(), neg_per_edge, replace=False)
            sampled_negatives.append(torch.as_tensor(tmp_final_neg))
            # sampled_negatives.append(choose_negative[:50])
        else:
            raise RuntimeError("insufficient negatives sampled")
        
    neg_valid_edges = torch.stack((sampled_negatives))
    sampled_negatives = []
    for i in tqdm(range(test_pos_u.shape[0])):
        u = test_pos_u[i]
        v = test_pos_v[i]
        if num_negative_edges > num_nodes:
            num_negatives = nodes_to_sample
        else:
            num_negatives = np.random.choice(
                num_nodes, num_negative_edges, replace=False)
            # num_negatives = np.unique(num_negatives)
            num_negatives = torch.as_tensor(num_negatives, dtype=torch.int64)
        num_negative_edges_filtered = num_negatives.shape[0]
        sampled_negative = nodes_to_sample[num_negatives]
        repeat_u = torch.full((num_negative_edges_filtered, 1), u)[:, 0]
        check_valid = g.has_edges_between(
            repeat_u, sampled_negative)
        choose_valid = check_valid == False
        choose_negative = sampled_negative[choose_valid]
        if torch.sum(choose_valid) >= neg_per_edge:
            tmp_final_neg = np.random.choice(
                choose_negative.numpy(), neg_per_edge, replace=False)
            sampled_negatives.append(torch.as_tensor(tmp_final_neg))
            # sampled_negatives.append(choose_negative[:50])
        else:
            raise RuntimeError("insufficient negatives sampled")
        
    neg_test_edges = torch.stack((sampled_negatives))

    edge_split = {}
    edge_split['train'] = {}
    edge_split['train']['source_node'] = train_pos_u.long()
    edge_split['train']['target_node'] = train_pos_v.long()
    edge_split['valid'] = {}
    edge_split['valid']['source_node'] = val_pos_u.long()
    edge_split['valid']['target_node'] = val_pos_v.long()
    edge_split['valid']['target_node_neg'] = neg_valid_edges.long()
    edge_split['test'] = {}
    edge_split['test']['source_node'] = test_pos_u.long()
    edge_split['test']['target_node'] = test_pos_v.long()
    edge_split['test']['target_node_neg'] = neg_test_edges.long()

    save_path = os.path.join(path, f"{name}-train.dgl")
    g_sub = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    g_sub.ndata['feat'] = g.ndata['feat']
    save_graphs(save_path, [g_sub])
    torch.save(edge_split, os.path.join(
        path, f"{name}-train-split.pt"))


dataset = torch.load("./dataset/attributed_PPI/ppi/processed/data.pt")[0]
dataset = torch_geometric.data.Data(x=dataset['x'], edge_index=dataset['edge_index'])
g = torch_geometric.utils.to_dgl(dataset)

g.ndata['feat'] = g.ndata.pop('x')
split_graph_new_50(g, name=f"AttributedGraphTWeibo", ratio=0.1, neg_per_edge=200)