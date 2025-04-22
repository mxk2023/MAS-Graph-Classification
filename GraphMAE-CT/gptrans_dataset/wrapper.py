# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import pyximport
import gptrans_dataset.augmentors as A 
from torch_geometric.transforms import AddRandomWalkPE

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos


@torch.jit.script
def convert_to_single_emb(x, offset: int = 8):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


# def process_aug(x,edge_index,edge_attr,aug_ratio):
#     aug = A.RandomChoice([A.FeatureDropout(pf = aug_ratio),
#                            A.FeatureMasking(pf=aug_ratio)],2)
#     return aug(x,edge_index,edge_attr)

def process_aug(x,edge_index,edge_attr,aug_ratio):
    aug = A.RandomChoice([A.FeatureDropout(pf = aug_ratio),
                           A.FeatureMasking(pf=aug_ratio),
                           A.EdgeRemoving(pe=aug_ratio),
                           A.EdgeAdding(pe=aug_ratio)],1)
    return aug(x,edge_index,edge_attr)



def preprocess_item(item,device,random_walk_length):
    edge_attr, edge_index,x = item.edge_attr, item.edge_index,item.x
    N = x.size(0)
    # x = convert_to_single_emb(x)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True


    # edge feature here
    if edge_attr is not None:
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]
                       ] = convert_to_single_emb(edge_attr.long()) + 1
    else:
        attn_edge_type = None

    # shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    # max_dist = np.amax(shortest_path_result)
    # if edge_attr is not None:
    #     edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    #     item.edge_input = torch.from_numpy(edge_input).long()
    # else:
    #     edge_input = None
    #     item.edge_input = edge_input
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    # attn_bias = torch.zeros([N, N], dtype=torch.float)
    
    item = item.cuda(device)
    add_pe=AddRandomWalkPE(random_walk_length)
    pe=add_pe(item).random_walk_pe
    pe=pe.cpu()
    item = item.cpu()

    # x2,edge_index2,edge_attr2 = process_aug(x,edge_index,edge_attr,aug_ratio)
    # item.x = x2
    # item.edge_index = edge_index2
    # item.edge_attr=edge_attr2

    # add_pe2=AddRandomWalkPE(random_walk_length)
    # pe2=add_pe2(item).random_walk_pe
    # item=item.cpu()
    # pe2=pe2.cpu()

    # adj2 = torch.zeros([x2.size(0), x2.size(0)], dtype=torch.bool)
    # adj2[edge_index2[0, :], edge_index2[1, :]] = True

    # combine
    item.x = x
    # item.x2 = x2
    item.pe = pe
    item.edge_index = edge_index
    item.edge_attr = edge_attr
    # item.adj = adj
    # item.attn_bias = attn_bias
    # item.attn_edge_type = attn_edge_type
    # item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)

    return item

