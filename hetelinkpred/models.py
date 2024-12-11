import torch
import torch.nn.functional as F
import dgl.nn as dglnn
import tqdm
import math
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear
import dgl
from dgl.nn.pytorch.glob import SumPooling



class NGNN_GCNConv(torch.nn.Module):
    def __init__(
            self, in_channels, hidden_channels, out_channels, num_nonl_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = dglnn.GraphConv(in_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        # n-layer GraphSAGE-mean
        for i in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y

class DOTPredictor(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)

class DistMultSimplePredictor(nn.Module):
    def __init__(self, hid_size) -> None:
        super().__init__()
        self.hid_size = hid_size
        self.linear = nn.Linear(
            hid_size, 1, bias=False
        )
    
    def forward(self, x):
        return self.linear(x)
    

class SAGE_DOT(SAGE):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__(in_size, hid_size, num_layers)
        self.predictor = DOTPredictor()

class SAGE_DistMultS(SAGE):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__(in_size, hid_size, num_layers)
        self.predictor = DistMultSimplePredictor(hid_size)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']


# class scorepredictor(nn.Module):
#     def forward(self, x_1, x_2):
#         return (x_1 * x_2).sum(dim=1)


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size))
        # n-layer GraphConv
        for i in range(num_layers - 1):
            self.layers.append(
                dglnn.GraphConv(hid_size, hid_size))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                # h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y

# class GPR_prop(MessagePassing):
#     '''
#     propagation class for GPR_GNN
#     '''

#     def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
#         super(GPR_prop, self).__init__(aggr='add', **kwargs)
#         self.K = K
#         self.Init = Init
#         self.alpha = alpha
#         self.Gamma = Gamma

#         assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
#         if Init == 'SGC':
#             # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
#             TEMP = 0.0*np.ones(K+1)
#             TEMP[alpha] = 1.0
#         elif Init == 'PPR':
#             # PPR-like
#             TEMP = alpha*(1-alpha)**np.arange(K+1)
#             TEMP[-1] = (1-alpha)**K
#         elif Init == 'NPPR':
#             # Negative PPR
#             TEMP = (alpha)**np.arange(K+1)
#             TEMP = TEMP/np.sum(np.abs(TEMP))
#         elif Init == 'Random':
#             # Random
#             bound = np.sqrt(3/(K+1))
#             TEMP = np.random.uniform(-bound, bound, K+1)
#             TEMP = TEMP/np.sum(np.abs(TEMP))
#         elif Init == 'WS':
#             # Specify Gamma
#             TEMP = Gamma

#         self.temp = Parameter(torch.tensor(TEMP))

#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.temp)
#         if self.Init == 'SGC':
#             self.temp.data[self.alpha] = 1.0
#         elif self.Init == 'PPR':
#             # for k in range(self.K+1):
#             #     self.temp.data[k] = self.alpha*(1-self.alpha)**k
#             self.temp.data = self.alpha*(1-self.alpha)**np.arange(self.K+1)
#             self.temp.data[-1] = (1-self.alpha)**self.K
#         elif self.Init == 'NPPR':
#             # for k in range(self.K+1):
#             #     self.temp.data[k] = self.alpha**k
#             self.temp.data = self.alpha**np.arange(self.K+1)
#             self.temp.data = self.temp.data / \
#                 torch.sum(torch.abs(self.temp.data))
#         elif self.Init == 'Random':
#             bound = np.sqrt(3/(self.K+1))
#             torch.nn.init.uniform_(self.temp, -bound, bound)
#             self.temp.data = self.temp.data / \
#                 torch.sum(torch.abs(self.temp.data))
#         elif self.Init == 'WS':
#             self.temp.data = self.Gamma

#     def forward(self, x, edge_index, edge_weight=None):
#         edge_index, norm = gcn_norm(
#             edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

#         hidden = x*(self.temp[0])
#         for k in range(self.K):
#             x = self.propagate(edge_index, x=x, norm=norm)
#             gamma = self.temp[k+1]
#             hidden = hidden + gamma*x
#         return hidden

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def __repr__(self):
#         return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
#                                           self.temp)

class GPRGNN(nn.Module):
     def __init__(self, in_size, hid_size, alpha=0.1, num_layers=3, dropout=0.5, dprate=0.5):
         super().__init__()
         self.layers = nn.ModuleList()
         self.lin1 = nn.Linear(in_size, hid_size)
         self.lin2 = nn.Linear(hid_size, hid_size)

         self.num_layers = num_layers
         self.alpha = alpha
         self._init_gamma()

         # n-layer GraphConv
         for i in range(num_layers):
             self.layers.append(
                 dglnn.GraphConv(hid_size, hid_size, 
                                 weight=False, bias=False))
         self.hid_size = hid_size
         self.predictor = nn.Sequential(
             nn.Linear(hid_size, hid_size),
             nn.ReLU(),
             nn.Linear(hid_size, hid_size),
             nn.ReLU(),
             nn.Linear(hid_size, 1))
         self.dropout = nn.Dropout(dropout)
         self.dp_dropout = nn.Dropout(dprate)

     def reset_parameters(self):
         super().reset_parameters()
         self._init_gamma()

     def _init_gamma(self):
         gamma = self.alpha * (1 - self.alpha) ** torch.arange(self.num_layers + 1)
         gamma[-1] = (1 - self.alpha) ** self.num_layers
         self.gamma = nn.Parameter(gamma)

     def forward(self, pair_graph, neg_pair_graph, blocks, x):
         h = x
         h = self.dropout(h)
         h = F.relu(self.lin1(h))
         h = self.dropout(h)
         h = self.lin2(h)
         h = self.dp_dropout(h)

         rep = self.gamma[0] * h
         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
             h = layer(block, h)
             rep = rep[block.dstnodes()]
             rep += self.gamma[l + 1] * h

         h = rep
         pos_src, pos_dst = pair_graph.edges()
         neg_src, neg_dst = neg_pair_graph.edges()
         h_pos = self.predictor(h[pos_src] * h[pos_dst])
         h_neg = self.predictor(h[neg_src] * h[neg_dst])
         return h_pos, h_neg


     def inference(self, g, device, batch_size):
         """Layer-wise inference algorithm to compute GNN node embeddings."""
         feat = g.ndata['feat'].float()
         sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
         dataloader = DataLoader(
             g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
             batch_size=batch_size, shuffle=False, drop_last=False,
             num_workers=0)
         buffer_device = torch.device('cpu')
         pin_memory = (buffer_device != device)

         node_rep = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                                pin_memory=pin_memory)
         for l, layer in enumerate(self.layers):
             y = torch.zeros(g.num_nodes(), self.hid_size, device=device)
             feat = feat.to(device)
             for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                 x = feat[input_nodes]
                 if l == 0:
                     h = F.relu(self.lin1(x))
                     h = self.lin2(h)
                     node_rep[input_nodes] = (self.gamma[0] * h).to(buffer_device)
                 else:
                     h = x
                 h = layer(blocks[0], h)
                 node_rep[output_nodes.cpu()] += (self.gamma[l + 1] * h).to(buffer_device)
                 y[output_nodes] = h.to(device)
             feat = y
             # assert (y.abs().sum(axis=1, keepdim=True) > 0).all()

         return node_rep


class GPRGNN_DOT(GPRGNN):
    def __init__(self, in_size, hid_size, num_layers=3, ppnp='PPNP', K=10, alpha=0.1, Init='PPR', Gamma=None, dprate=0.5, dropout=0.5):
        super().__init__(in_size, hid_size, alpha, num_layers, dropout, dprate)
        self.predictor = DOTPredictor()


class GPRGNN_DistMultS(GPRGNN):
    def __init__(self, in_size, hid_size, num_layers=3, ppnp='PPNP', K=10, alpha=0.1, Init='PPR', Gamma=None, dprate=0.5, dropout=0.5):
        super().__init__(in_size, hid_size, alpha, num_layers, dropout, dprate)
        self.predictor = DistMultSimplePredictor(hid_size)


class MLPDecoder(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        # for l, (layer, block) in enumerate(zip(self.layers, blocks)):
        #     h = layer(block, h)
        #     if l != len(self.layers) - 1:
        #         h = F.relu(h)
        #         # h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        # sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        # dataloader = DataLoader(
        #     g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
        #     batch_size=batch_size, shuffle=False, drop_last=False,
        #     num_workers=0)
        # buffer_device = torch.device('cpu')
        # pin_memory = (buffer_device != device)
        # for l, layer in enumerate(self.layers):
        #     y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
        #                     pin_memory=pin_memory)
        #     feat = feat.to(device)
        #     for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
        #         x = feat[input_nodes]
        #         h = layer(blocks[0], x)
        #         if l != len(self.layers) - 1:
        #             h = F.relu(h)
        #         y[output_nodes] = h.to(buffer_device)
        #     feat = y
        return feat

class DistMultSDecoder(MLPDecoder):
    def __init__(self, in_size, hid_size, num_layers=3):
        super(MLPDecoder, self).__init__()
        self.predictor = DistMultSimplePredictor(in_size)


class GCN_DOT(GCN):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__(in_size, hid_size, num_layers)
        self.predictor = DOTPredictor()

class GCN_DistMultS(GCN):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__(in_size, hid_size, num_layers)
        self.predictor = DistMultSimplePredictor(hid_size)

class GATv2(nn.Module):
    def __init__(self, in_size, hid_size, num_layers, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        self.heads = heads
        self.hid_size = hid_size
        self.layer_norms = torch.nn.ModuleList()
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(in_size, hid_size, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation,
                      bias=False, share_weights=True)
        )
        # hidden layers
        for l in range(num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layer_norms.append(nn.LayerNorm(hid_size * heads[l]))
            self.gatv2_layers.append(
                GATv2Conv(hid_size * heads[l], hid_size, heads[l + 1], feat_drop, attn_drop, negative_slope, residual,
                          self.activation, bias=False, share_weights=True)
            )
        # output projection
        self.predictor = nn.Sequential(
            nn.Linear(hid_size * heads[-1], hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gatv2_layers, blocks)):
            h = layer(block, h).flatten(1)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.gatv2_layers):
            y = torch.empty(g.num_nodes(), self.hid_size * self.heads[l], device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x).flatten(1)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        # h = F.relu(self.batch_norm(self.linears[0](h)))
        h = F.relu(self.linears[0](h))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.hid_size = hid_size
        self.batch_norms = nn.ModuleList()

        mlp = MLP(in_size, hid_size, hid_size)
        self.ginlayers.append(GINConv(mlp, learn_eps=False))
        self.batch_norms.append(nn.BatchNorm1d(hid_size))
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            mlp = MLP(hid_size, hid_size, hid_size)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hid_size))

        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        # list of hidden representation at each layer (including the input layer)
        h = x
        for l, (layer, block) in enumerate(zip(self.ginlayers, blocks)):
            h = layer(block, h)
            # h = self.batch_norms[l](h)
            if l != len(self.ginlayers) - 1:
                h = F.relu(h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.ginlayers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                h = self.batch_norms[l](h)
                if l != len(self.ginlayers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
