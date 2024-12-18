import torch
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import dgl
import os
import pandas as pd
import warnings
from tqdm import tqdm

def debug_attach(port=5678):
    import debugpy
    debugpy.listen(port)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    print("Debugger attached")
    breakpoint()         

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    u, v = g.edges()
    if (u == v).any():
        warnings.warn('Self-loop detected. Will remove self-loops.')
        g = dgl.remove_self_loop(g)
    
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


def load_usair_dataset(device, dataset_root):
    dataset, label_dict = dgl.load_graphs(os.path.join(dataset_root, "USAir.dgl"))
    g = dataset[0]
    g = g.to('cpu')
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    load_split = np.load(os.path.join(dataset_root, "USAir.npz"), allow_pickle=True)['current'].item()
    test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(load_split['orginal_A'] == 0, k=1))
    test_neg_i = torch.Tensor(test_neg_i).unsqueeze(1)
    test_neg_j = torch.Tensor(test_neg_j).unsqueeze(1)
    negatives = torch.cat((test_neg_i, test_neg_j), dim=1)
    edge_split = {}
    train_idx = torch.cat((torch.Tensor(load_split['train_pos'][0]).unsqueeze(1),
                           torch.Tensor(load_split['train_pos'][1]).unsqueeze(1)), dim=1)
    valid = torch.cat((torch.Tensor(load_split['valid_pos'][0]).unsqueeze(1),
                       torch.Tensor(load_split['valid_pos'][1]).unsqueeze(1)), dim=1)
    test = torch.cat((torch.Tensor(load_split['test_pos'][0]).unsqueeze(1),
                      torch.Tensor(load_split['test_pos'][1]).unsqueeze(1)), dim=1)
    edge_split['train'] = {}
    edge_split['train']['edge'] = train_idx.long()
    edge_split['valid'] = {}
    edge_split['valid']['edge'] = valid.long()
    edge_split['valid']['edge_neg'] = negatives.long()
    edge_split['test'] = {}
    edge_split['test']['edge'] = test.long()
    edge_split['test']['edge_neg'] = negatives.long()
    g.ndata['feat'] = F.one_hot(torch.arange(0, g.number_of_nodes()))
    return g, reverse_eids, seed_edges, edge_split

def load_grid_dataset(device, dataset_root, graph_idx):

    assort_dict = {
                    "0":"0.639485",
                    "1":"0.299142",
                    "2":"0.235534",
                    "3":"0.189669",
                    "4":"0.151556",
                    "5":"0.120896",
                    "6":"0.095887",
                    "7":"0.075932",
                    "8":"0.059205",
                    "9":"0.044877",
                    "10":"0.031907",
                    "11":"0.020253",
                    "12":"0.009716",
                    "13":"-0.000675",
                    "14":"-0.011686",
                    "15":"-0.023202",
                    "16":"-0.034559",
                    "17":"-0.046585",
                    "18":"-0.059712",
                    "19":"-0.073061",
                    "20":"-0.087600",
                    "21":"-0.104724",
                    "22":"-0.123835",
                    "23":"-0.146388",
                    "24":"-0.177851"
                }
    
    g = dgl.load_graphs(dataset_root+"/grid4#"+str(graph_idx)+"#"+assort_dict[str(graph_idx)]+assort_dict[str(graph_idx)]+"-train.dgl")[0][0]

    g = g.to('cpu')
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    
    edge_split = torch.load(dataset_root+"/grid4#"+str(graph_idx)+"#"+assort_dict[str(graph_idx)]+assort_dict[str(graph_idx)]+"-train-split.pt")
    return g, reverse_eids, seed_edges, edge_split

def load_dense_grid_dataset(device, dataset_root, graph_idx):

    # assort_dict = {
    #                 "0":"-0.216436",
    #                 "1":"-0.100126",
    #                 "2":"-0.014610",
    #                 "3":"0.080349",
    #                 "4":"0.264352",
    #             }
    
    base_filename = dataset_root + "/dense_jiong_new-" + str(graph_idx)
    g = dgl.load_graphs(base_filename+"-train.dgl")[0][0]

    g = g.to('cpu')
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    
    edge_split = torch.load(base_filename + "-train-split.pt")
    # debug_attach()

    assert edge_split["valid"]["source_node"].shape[0] == edge_split["valid"]["target_node_neg"].shape[0]
    assert edge_split["test"]["source_node"].shape[0] == edge_split["test"]["target_node_neg"].shape[0]
    return g, reverse_eids, seed_edges, edge_split

def load_custom_dgl_dataset(device, dataset_root, graph_name):

    # assort_dict = {
    #                 "0":"-0.216436",
    #                 "1":"-0.100126",
    #                 "2":"-0.014610",
    #                 "3":"0.080349",
    #                 "4":"0.264352",
    #             }
    
    base_filename = os.path.join(dataset_root, graph_name)
    g = dgl.load_graphs(base_filename+"-train.dgl")[0][0]
    
    g = g.to('cpu')
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    
    edge_split = torch.load(base_filename + "-train-split.pt")
    # debug_attach()

    assert edge_split["valid"]["source_node"].shape[0] == edge_split["valid"]["target_node_neg"].shape[0]
    assert edge_split["test"]["source_node"].shape[0] == edge_split["test"]["target_node_neg"].shape[0]
    return g, reverse_eids, seed_edges, edge_split


def load_esci_dataset(device, dataset_root, val_num_negs=1000):
    dataset, label_dict = dgl.load_graphs(os.path.join(dataset_root, "esci_train.dgl"))
    g = dataset[0]
    g = g.to('cpu')
    # since negative sampler regards reverse edges as non-exists and might sample them
    # it's essential to add reverse edges to the graph first
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    load_split = np.load(os.path.join(dataset_root, "esci.npz"), allow_pickle=True)['current'].item()
    # generate validation and test negatives
    num_negative_edges = int(1.1 * val_num_negs)
    # negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_edges)
    orig_g = dgl.load_graphs(os.path.join(dataset_root, "esci.dgl"))[0][0]
    # get valid, test edge id
    asins_to_sample = torch.arange(33804, orig_g.number_of_nodes())
    num_asins = g.number_of_nodes() - 33804
    sampled_negatives = []
    
    if (os.path.isfile(os.path.join(dataset_root, "esci_valid_neg_"+str(val_num_negs)+".pt"))):
        neg_valid_edges = torch.load(os.path.join(dataset_root, "esci_valid_neg_"+str(val_num_negs)+".pt"))
    else:
        for i in tqdm(range(load_split['valid_pos'].shape[0])):
            u = load_split['valid_pos'][i][0]
            v = load_split['valid_pos'][i][1]
            num_negatives = torch.randperm(num_asins)[:num_negative_edges]
            sampled_negative = asins_to_sample[num_negatives]
            repeat_u = torch.full((num_negative_edges, 1), u)[:, 0]
            check_valid = orig_g.has_edges_between(repeat_u, sampled_negative)
            choose_valid = check_valid == False
            choose_negative = sampled_negative[choose_valid]
            if torch.sum(choose_valid) > val_num_negs:
                sampled_negatives.append(choose_negative[:val_num_negs])
            else:
                print("insufficient negatives sampled")
                import pdb
                pdb.set_trace()
        neg_valid_edges = torch.stack((sampled_negatives))
        torch.save(neg_valid_edges, os.path.join(dataset_root, "esci_valid_neg_"+str(val_num_negs)+".pt"))


    if (os.path.isfile(os.path.join(dataset_root, "esci_test_neg_"+str(val_num_negs)+".pt"))):
        neg_test_edges = torch.load(os.path.join(dataset_root, "esci_test_neg_"+str(val_num_negs)+".pt"))
    else:
        sampled_negatives = []
        for i in tqdm(range(load_split['test_pos'].shape[0])):
            u = load_split['test_pos'][i][0]
            v = load_split['test_pos'][i][1]
            num_negatives = torch.randperm(num_asins)[:num_negative_edges]
            sampled_negative = asins_to_sample[num_negatives]
            repeat_u = torch.full((num_negative_edges, 1), u)[:, 0]
            check_test = orig_g.has_edges_between(repeat_u, sampled_negative)
            choose_test = check_test == False
            choose_negative = sampled_negative[choose_test]
            if torch.sum(choose_test) > val_num_negs:
                sampled_negatives.append(choose_negative[:val_num_negs])
            else:
                print("insufficient negatives sampled")
                import pdb
                pdb.set_trace()
        neg_test_edges = torch.stack((sampled_negatives))
        torch.save(neg_test_edges, os.path.join(dataset_root, "esci_test_neg_"+str(val_num_negs)+".pt"))

    # valid_eids = orig_g.edge_ids(load_split['valid_pos'][:, 0], load_split['valid_pos'][:, 1])
    # test_eids = orig_g.edge_ids(load_split['test_pos'][:, 0], load_split['test_pos'][:, 1])
    # negative_valid_edges = negative_sampler(g, valid_eids)
    # negative_test_edges = negative_sampler(g, test_eids)
    # import pdb 
    # pdb.set_trace()
    # double check that negative edges does not exist in the graph
    # check_valid = orig_g.has_edges_between(negative_valid_edges[0], negative_valid_edges[1])
    # check_test = orig_g.has_edges_between(negative_test_edges[0], negative_test_edges[1])
    # choose_valid = check_valid == False
    # choose_test = check_test == False
    # print(torch.sum(check_valid))
    # print(torch.sum(check_test))
    # negative_valid_edges_u = negative_valid_edges[0][choose_valid].unsqueeze(1)
    # negative_valid_edges_v = negative_valid_edges[1][choose_valid].unsqueeze(1)
    # negative_test_edges_u = negative_test_edges[0][choose_test].unsqueeze(1)
    # negative_test_edges_v = negative_test_edges[1][choose_test].unsqueeze(1)
    # neg_valid_edges = torch.cat((negative_valid_edges_u, negative_valid_edges_v), dim=1)
    # neg_test_edges = torch.cat((negative_test_edges_u, negative_test_edges_v), dim=1)
    # check_valid = g.has_edges_between(neg_valid_edges[:, 0], neg_valid_edges[:, 1])
    # check_test = g.has_edges_between(neg_test_edges[:, 0], neg_test_edges[:, 1])
    # print(torch.sum(check_valid))
    # print(torch.sum(check_test))
    edge_split = {}
    edge_split['train'] = {}
    edge_split['train']['source_node'] = load_split['train_pos'][:, 0].long()
    edge_split['train']['target_node'] = load_split['train_pos'][:, 1].long()
    edge_split['valid'] = {}
    edge_split['valid']['source_node'] = load_split['valid_pos'][:, 0].long()
    edge_split['valid']['target_node'] = load_split['valid_pos'][:, 1].long()
    edge_split['valid']['target_node_neg'] = neg_valid_edges.long()
    edge_split['test'] = {}
    edge_split['test']['source_node'] = load_split['test_pos'][:, 0].long()
    edge_split['test']['target_node'] = load_split['test_pos'][:, 1].long()
    edge_split['test']['target_node_neg'] = neg_test_edges.long()
    g.ndata['feat'] = g.ndata['feats']
    return g, reverse_eids, seed_edges, edge_split


def remove_collab_dissimilar_edges(dataset):
    """
    Remove the edges in edge_split()
    :param dataset:
    :return: processed collab dataset
    """
    edge_split = dataset.get_edge_split()
    dataset, label_dict = dgl.load_graphs('ogbl-collab_low_degree.dgl')
    g = dataset[0]
    new_edge_split = {
        'train': {'edge': [], 'weight': [], 'year': []},
        'valid': {'edge': edge_split['valid']['edge'], 'weight': edge_split['valid']['weight'],
                  'year': edge_split['valid']['year'], 'edge_neg': edge_split['valid']['edge_neg']},
        'test': {'edge': edge_split['test']['edge'], 'weight': edge_split['test']['weight'],
                 'year': edge_split['test']['year'], 'edge_neg': edge_split['test']['edge_neg']}
    }
    for split in ['train']:
        edge_tuple = edge_split[split]['edge']
        weight_list = edge_split[split]['weight']
        year_list = edge_split[split]['year']
        for index, edge in enumerate(edge_tuple):
            if g.has_edges_between(edge[0], edge[1]):
                new_edge_split[split]['edge'].append([edge[0], edge[1]])
                new_edge_split[split]['weight'].append(weight_list[index])
                new_edge_split[split]['year'].append(year_list[index])
        new_edge_split[split]['edge'] = torch.tensor(new_edge_split[split]['edge'], dtype=torch.int64)
        new_edge_split[split]['weight'] = torch.tensor(new_edge_split[split]['weight'], dtype=torch.int64)
        new_edge_split[split]['year'] = torch.tensor(new_edge_split[split]['year'], dtype=torch.int64)

    import pickle
    with open('ogbl-collab_low_degree.pickle', 'wb') as handle:
        pickle.dump(new_edge_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return g, new_edge_split


def construct_low_degree_collab(dataset):
    """
    Construct the low-degree collab, for each node, remove the edge connected with most dissimilar nodes
    """
    g = dataset[0]
    node_feature = g.ndata['feat']
    cosi = torch.nn.CosineSimilarity(dim=0)
    num_of_nodes = g.num_nodes()
    index = 0
    for head_node in range(num_of_nodes):
        if index % 100 == 0:
            print(index)
        index += 1
        head_degree = g.in_degrees(head_node)
        if head_degree > 1:
            _, dst_node_list = g.out_edges(head_node)
            simi_vector = []
            for dst_node in dst_node_list:
                simi_vector.append(cosi(node_feature[head_node], node_feature[dst_node]))
            simi_vector = torch.tensor(simi_vector)
            dissimi_dst_node = dst_node_list[torch.argmin(simi_vector)]
            tail_degree = g.in_degrees(dissimi_dst_node)
            if tail_degree > 1 and g.has_edges_between(head_node, dissimi_dst_node):
                remove_edge_id = g.edge_ids(head_node, dissimi_dst_node)
                g.remove_edges(remove_edge_id)

    dgl.save_graphs('ogbl-collab_low_degree.dgl', [g])
    print("Finish removing edges")


class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]
        self.metric_name = None

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2%}')
            print(f'   Final Test: {result[argmax, 1]:.2%}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            valid_stats = (r.mean().item(), r.std().item())
            print(f'Highest Valid: {valid_stats[0]:.2%} ± {valid_stats[1]:.2%}')
            r = best_result[:, 1]
            test_stats = (r.mean().item(), r.std().item())
            print(f'   Final Test: {test_stats[0]:.2%} ± {test_stats[1]:.2%}')
            return valid_stats, test_stats

    @property
    def results_df(self):
        results_arr = np.array(self.results, dtype=object)
        run_df_list = []
        for i in range(results_arr.shape[0]):
            run_arr = np.array(self.results[i])
            if len(run_arr) == 0:
                continue
            run_df = pd.DataFrame(run_arr, columns=['valid', 'test'])
            run_df.index.name = 'epoch'
            run_df_list.append(run_df)
        df = pd.concat(run_df_list, keys=range(results_arr.shape[0]), names=["run"])
        if self.metric_name is not None:
            df["metric"] = self.metric_name
        return df
    

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
