import dgl
import torch
import numpy as np

dataset, label_dict = dgl.load_graphs('esci.dgl')
g = dataset[0]
# isolated nodes will affect the negative sampler a lot and hurts performance
# need to remove it during preprocessing
isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
g.remove_nodes(isolated_nodes)
print(g)
# generate train, valid, test mask
train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2
int_edges = g.number_of_edges()
num_train_exps = int(int_edges * train_ratio)
num_valid_exps = int(int_edges * valid_ratio)
perturb_edges = torch.randperm(int(int_edges))
train_edges = perturb_edges[:num_train_exps] # train ids
valid_edges = perturb_edges[num_train_exps:num_valid_exps+num_train_exps] # valid ids
test_edges = perturb_edges[num_valid_exps+num_train_exps:] # test ids
u, v = g.edges()
train_u = u[train_edges]
train_v = v[train_edges]
train_pos = torch.cat((train_u.unsqueeze(1), train_v.unsqueeze(1)), dim=1)
valid_u = u[valid_edges]
valid_v = v[valid_edges]
valid_pos = torch.cat((valid_u.unsqueeze(1), valid_v.unsqueeze(1)), dim=1)
test_u = u[test_edges]
test_v = v[test_edges]
test_pos = torch.cat((test_u.unsqueeze(1), test_v.unsqueeze(1)), dim=1)
# delete valid and test edges in the graph
valid_eids = g.edge_ids(valid_u, valid_v)
test_eids = g.edge_ids(test_u, test_v)
delete_eids = torch.cat((valid_eids,  test_eids), dim=0)
train_g = dgl.remove_edges(g, delete_eids, store_ids=False)
isolated_nodes = ((train_g.in_degrees() == 0) & (train_g.out_degrees() == 0)).nonzero().squeeze(1)
import pdb
pdb.set_trace()
assert isolated_nodes.shape[0] == 0
print(train_g)
labels = {'train_pos': train_pos, 'valid_pos': valid_pos, 'test_pos': test_pos}
np.savez_compressed('esci.npz', current=labels, allow_pickle=True)
# save g again
dgl.save_graphs('esci.dgl', [g])
dgl.save_graphs('esci_train.dgl', [train_g])
print("successfully saved graph")