"""
testing / inference functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.evaluation import evaluate_auc, evaluate_hits, evaluate_mrr, compute_mrr_esci
from src.utils import get_num_samples


def get_test_func(model_str):
    if model_str == 'ELPH':
        return get_elph_preds
    elif model_str == 'BUDDY':
        return get_buddy_preds
    else:
        return get_preds


@torch.no_grad()
def test(model, evaluator, train_loader, val_loader, test_loader, args, device, emb=None, eval_metric='hits'):
    print('starting testing')
    t0 = time.time()
    model.eval()
    print("get train predictions")
    test_func = get_test_func(args.model)
    pos_train_pred, neg_train_pred, train_pred, train_true = test_func(model, train_loader, device, args, split='train')
    print("get val predictions")
    pos_val_pred, neg_val_pred, val_pred, val_true, links = test_func(model, val_loader, device, args, split='val', return_links=True)
    pos_val_links = links[val_true == 1]
    print("get test predictions")
    pos_test_pred, neg_test_pred, test_pred, test_true, links = test_func(model, test_loader, device, args, split='test', return_links=True)
    pos_test_links = links[test_true == 1]

    if eval_metric == 'hits':
        results = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                                neg_test_pred, Ks=[args.K])
        rr = torch.concat(results["RR"][1:])
        rr_df = pd.DataFrame({
            "src": torch.concat((pos_val_links[:, 0], pos_test_links[:, 0])),
            "dst": torch.concat((pos_val_links[:, 1], pos_test_links[:, 1])),
            f"hits@{args.K}": rr,
            "keys": ["valid"] * len(pos_val_links) + ["test"] * len(pos_test_links)
        })
        results["RR"] = rr_df
    elif eval_metric == 'mrr':
        if args.dataset_name in {'grid', 'amazon-computer','esci', "attributed-facebook", "attributed-ppi"}:
            results = compute_mrr_esci(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
            rr = torch.concat(results["RR"][1:])
            rr_df = pd.DataFrame({
                "src": torch.concat((pos_val_links[:, 0], pos_test_links[:, 0])),
                "dst": torch.concat((pos_val_links[:, 1], pos_test_links[:, 1])),
                "MRR": rr,
                "keys": ["valid"] * len(pos_val_links) + ["test"] * len(pos_test_links)
            })
            results["RR"] = rr_df
        else:
            results = evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)
            rr = torch.concat(results["RR"][1:])
            rr_df = pd.DataFrame({
                "src": torch.concat((pos_val_links[:, 0], pos_test_links[:, 0])),
                "dst": torch.concat((pos_val_links[:, 1], pos_test_links[:, 1])),
                "MRR": rr,
                "keys": ["valid"] * len(pos_val_links) + ["test"] * len(pos_test_links)
            })
            results["RR"] = rr_df
    elif eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    print(f'testing ran in {time.time() - t0}')

    return results


@torch.no_grad()
def get_preds(model, loader, device, args, emb=None, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    y_pred, y_true = [], []
    pbar = tqdm(loader, ncols=70)
    batch_processing_times = []
    t0 = time.time()
    for batch_count, data in enumerate(pbar):
        start_time = time.time()
        if args.model == 'BUDDY':
            data_dev = [elem.squeeze().to(device) for elem in data]
            logits = model(*data_dev[:-1])
            y_true.append(data[-1].view(-1).cpu().to(torch.float))
        else:
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.src_degree,
                           data.dst_degree)
            y_true.append(data.y.view(-1).cpu().to(torch.float))
        y_pred.append(logits.view(-1).cpu())
        batch_processing_times.append(time.time() - start_time)
        if (batch_count + 1) * args.batch_size > n_samples:
            del data
            torch.cuda.empty_cache()
            break
        del data
        torch.cuda.empty_cache()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = pred[true == 1]
    neg_pred = pred[true == 0]
    samples_used = len(loader.dataset) if n_samples > len(loader.dataset) else n_samples
    print(f'{len(pos_pred)} positives and {len(neg_pred)} negatives for sample of {samples_used} edges')
    return pos_pred, neg_pred, pred, true


@torch.no_grad()
def get_buddy_preds(model, loader, device, args, split=None, return_links=False):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    link_list = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    for batch_count, indices in enumerate(tqdm(loader)):
        curr_links = links[indices]
        link_list.append(curr_links)
        batch_emb = None if emb is None else emb[curr_links.to(torch.long)].to(device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links.to(torch.long)].to(device)
        degrees = data.degrees[curr_links.to(torch.long)].to(device)
        if args.use_RA:
            RA = data.RA[indices].to(device)
        else:
            RA = None
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    pred = torch.cat(preds)
    link = torch.cat(link_list)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    if not return_links:
        return pos_pred, neg_pred, pred, labels
    else:
        return pos_pred, neg_pred, pred, labels, link


def get_split_samples(split, args, dataset_len):
    """
    get the
    :param split: train, val, test
    :param args: Namespace object
    :param dataset_len: total size of dataset
    :return:
    """
    samples = dataset_len
    if split == 'train':
        if args.dynamic_train:
            samples = get_num_samples(args.train_samples, dataset_len)
    elif split in {'val', 'valid'}:
        if args.dynamic_val:
            samples = get_num_samples(args.val_samples, dataset_len)
    elif split == 'test':
        if args.dynamic_test:
            samples = get_num_samples(args.test_samples, dataset_len)
    else:
        raise NotImplementedError(f'split: {split} is not a valid split')
    return samples


@torch.no_grad()
def get_elph_preds(model, loader, device, args, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    # get node features
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
    for batch_count, indices in enumerate(tqdm(loader)):
        curr_links = links[indices].to(device)
        batch_emb = None if emb is None else emb[curr_links.to(torch.long)].to(device)
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        batch_node_features = None if node_features is None else node_features[curr_links.to(torch.long)]
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred, pred, labels
