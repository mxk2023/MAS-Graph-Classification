import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from graphmae.utils import create_activation, NormLayer, create_norm


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.layers.append(GCNConv(in_dim,num_hidden,normalize=norm))
        else:
            # input projection (no residual)
            self.layers.append(GCNConv(in_dim,num_hidden,normalize=norm)
                )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GCNConv(num_hidden,num_hidden,normalize=norm)
                )
            # output projection
            self.layers.append(GCNConv(num_hidden,num_hidden,normalize=norm))

        self.head = nn.Identity()

    def forward(self, inputs, edge_index, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](h, edge_index)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

