from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gat import GAT
from .gin import GIN
from .gcn import GCN
from .transformer import enc_transformerLayer,dec_transformerLayer
from .loss_func import sce_loss
from graphmae.utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, remove_self_loops,to_dense_batch
import gptrans_dataset.augmentors as A
from torch_geometric.transforms import AddRandomWalkPE


def setup_encoder_module(m_type, enc_dec, in_dim, num_hidden,ffn_dim, trans_num_heads,num_layers_enc_gnn,num_layers_enc_transformer, gnn_dropout,trans_dropout,attention_dropout, nhead, nhead_out, activation, residual, norm, negative_slope=0.2, concat_out=True):
    if m_type == "gat":
        encoder_gnn = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            num_layers=num_layers_enc_gnn,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=gnn_dropout,
            attn_drop=attention_dropout,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        encoder_gnn = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            num_layers=num_layers_enc_gnn,
            dropout=gnn_dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == 'gcn':
        encoder_gnn = GCN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            num_layers=num_layers_enc_gnn,
            dropout=gnn_dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError
    
    if m_type == 'gat':
        num_hidden = num_hidden * nhead
    encoders_trans=nn.ModuleList()
    for i in range(num_layers_enc_transformer):
        encoders_trans.append(enc_transformerLayer(num_hidden, ffn_dim, trans_dropout, attention_dropout, trans_num_heads))
    
    return encoder_gnn,encoders_trans

def setup_decoder_module(num_hidden,ffn_dim, trans_num_heads,num_layers_dec_transformer,trans_dropout,attention_dropout) -> nn.ModuleList:
    decoders=nn.ModuleList()
    for i in range(num_layers_dec_transformer):
        decoders.append(dec_transformerLayer(num_hidden, ffn_dim, trans_dropout, attention_dropout, trans_num_heads))
    return decoders


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            ffn_dim:int,
            num_layers_enc_gnn: int,
            num_layers_enc_transformer:int,
            num_layers_dec:int,
            nhead: int,
            nhead_out: int,
            trans_num_heads:int,
            activation: str,
            gnn_dropout: float,
            trans_dropout:float,
            attn_drop: float,
            negative_slope: float,
            edge_type:str,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            random_walk_length:int = 10,
            aug_type:str = '1', 
            aug_ratio:float=0.2,
            concat_hidden: bool = False,
            n_edge_features:int = None,
            degree_num_embeddings = 512,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.n_edge_features = n_edge_features
        self.mask_token = nn.Parameter(torch.zeros(1, num_hidden))
        self.random_walk_length = random_walk_length
        self.aug_type = aug_type
        self.aug_ratio=aug_ratio

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden 


        if self.n_edge_features is not None:
            self.edge_encoder = nn.Embedding(
                512 * self.n_edge_features + 1, trans_num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                128 * trans_num_heads * trans_num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(512, trans_num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            degree_num_embeddings, num_hidden, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            degree_num_embeddings, num_hidden, padding_idx=0)
        
        self.decoder_final=nn.Linear(ffn_dim,in_dim)
        # build encoder

        self.encoder_aug_gnn,self.encoder_aug_trans = setup_encoder_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            ffn_dim=ffn_dim,
            trans_num_heads=trans_num_heads,
            num_layers_enc_gnn=num_layers_enc_gnn,
            num_layers_enc_transformer=num_layers_enc_transformer,
            gnn_dropout=gnn_dropout,
            trans_dropout=trans_dropout,
            attention_dropout=attn_drop,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            activation=activation,
            residual=residual,
            norm=norm,
            negative_slope=negative_slope,
            concat_out=True
        )

        # build decoder
        self.decoder_layers = setup_decoder_module(
            num_hidden=dec_in_dim,
            ffn_dim=ffn_dim,
            trans_num_heads=trans_num_heads,
            num_layers_dec_transformer=num_layers_dec,
            trans_dropout=trans_dropout,
            attention_dropout=attn_drop,
        )

        

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_gnn_to_trans = nn.Linear(num_hidden * (num_layers_enc_gnn), num_hidden, bias=False)
        else:
            self.encoder_gnn_to_trans = nn.Linear(num_hidden, num_hidden, bias=False)
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.enc_pe_encoder = nn.Linear(num_hidden+random_walk_length,num_hidden)
        self.dec_pe_encoder = nn.Linear(num_hidden+random_walk_length,num_hidden)
        self.gnn2trans = nn.Linear(num_hidden,num_hidden)
        self.gnn2trans_act = nn.ReLU()

        self.node_norm = nn.BatchNorm1d(num_hidden)
        self.dec_node_norm = nn.BatchNorm1d(num_hidden)
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.ema_beta = 0.9999
        self.mask_token = nn.Parameter(torch.zeros(1, num_hidden))

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    

    def compute_pos_embeddings(self, batched_data):
        attn_bias, spatial_pos, x,x2  = batched_data.attn_bias.cuda(), batched_data.spatial_pos.cuda(), batched_data.x.cuda(), batched_data.x2.cuda()

        in_degree, out_degree = batched_data.in_degree.cuda(), batched_data.out_degree.cuda()
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias

        if batched_data.edge_input is not None:
            edge_input, attn_edge_type = batched_data.edge_input.cuda(), batched_data.attn_edge_type.cuda()
            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)
            graph_attn_bias = graph_attn_bias + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias

    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes,_ = perm[: num_mask_nodes].sort()
        keep_nodes,_ = perm[num_mask_nodes: ].sort()

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)
    

    # def process_aug(self,x,edge_index,edge_attr,aug_ratio):
    #     aug = A.RandomChoice([
    #                         A.FeatureMasking(pf=aug_ratio)],1)
    #     return aug(x,edge_index,edge_attr)
    def process_aug(self,x,edge_index,edge_attr,aug_ratio):
        if self.aug_type == '1':
            aug = A.RandomChoice([A.FeatureMasking(pf=aug_ratio)],1)
        elif self.aug_type == '2':
            aug = A.RandomChoice([A.FeatureDropout(pf=aug_ratio)],1)
        elif self.aug_type == '3':
            aug = A.RandomChoice([A.FeatureDropout(pf = aug_ratio),
                                A.FeatureMasking(pf=aug_ratio)],2)
        elif self.aug_type == '4':
            if aug_ratio == 0.2:
                aug = A.RandomChoice([A.RWSampling(512,5)],1)
            if aug_ratio == 0.4:
                aug = A.RandomChoice([A.RWSampling(512,10)],1)
            if aug_ratio == 0.6:
                aug = A.RandomChoice([A.RWSampling(512,20)],1)  
            if aug_ratio == 0.8:
                aug = A.RandomChoice([A.RWSampling(512,40)],1)  
        return aug(x,edge_index,edge_attr)



    def forward(self, batched_data):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(batched_data)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    
    def momentum_update_noisy_encoder(self):
        """
        Momentum update of the noisy_encoder
        """
        for param_noise_encoder, param_mask_encoder in zip(self.encoder_aug_gnn.parameters(),self.encoder_mask_gnn.parameters()):
            param_noise_encoder.data = param_noise_encoder.data * self.ema_beta + param_mask_encoder.data * (1.0 - self.ema_beta)
        
        for param_noise_encoder, param_mask_encoder in zip(self.encoder_aug_trans.parameters(),self.encoder_mask_trans.parameters()):
            param_noise_encoder.data = param_noise_encoder.data * self.ema_beta + param_mask_encoder.data * (1.0 - self.ema_beta)

    

    def encoder_aug(self,use_x,use_edge_index,batch,in_degree,out_degree,pe):
        
        enc_rep, all_hidden = self.encoder_aug_gnn(use_x, use_edge_index, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
            enc_rep = self.encoder_gnn_to_trans(enc_rep)


        rep = self.node_norm(enc_rep)+self.in_degree_encoder(in_degree)+self.out_degree_encoder(out_degree)
        # keep_rep = torch.cat((keep_rep,keep_pe),dim=1)
        # keep_rep = self.enc_pe_encoder(keep_rep)
        
        rep = self.gnn2trans_act(self.gnn2trans(rep))

        h, h_mask = to_dense_batch(rep, batch)
        for enc_layer in self.encoder_aug_trans:
            h = enc_layer(h)
        h = h[h_mask]

        return h
    
    def decoder(self,rep,aug_rep,batch_rep,batch_aug_rep):
        rep,rep_mask = to_dense_batch(rep,batch_rep)
        aug_rep,aug_rep_mask = to_dense_batch(aug_rep,batch_aug_rep)
        batch_size = min(max(batch_rep),max(batch_aug_rep))+1
        rep = rep[:batch_size,:,:]
        aug_rep = aug_rep[:batch_size,:,:]
        rep_mask = rep_mask[:batch_size,:]

        for dec_layer in self.decoder_layers:
            rep = dec_layer(rep,aug_rep)
            aug_rep = rep
        rep = self.decoder_final(rep)
        recon = rep[rep_mask]

        return recon


    def mask_attr_prediction(self,batched_data ):
        x,edge_index,edge_attr,batch,pe,in_degree,out_degree = batched_data.x,batched_data.edge_index,batched_data.edge_attr,batched_data.batch,batched_data.pe,batched_data.in_degree,batched_data.out_degree


        use_x,(mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        # if self.aug_ratio >0:
        #     x2,edge_index2,edge_attr2 = self.process_aug(x,edge_index,edge_attr,self.aug_ratio)
        # else:          
        x2,edge_index2,edge_attr2 = x,edge_index,edge_attr
        edge_index2 = add_self_loops(edge_index2)[0]

        #encoder
        h_aug = self.encoder_aug(x2,edge_index2,batch,in_degree,out_degree,pe)
        h_aug = h_aug[keep_nodes]

        # padding mask tokens
        padding_h = torch.zeros([len(mask_nodes),h_aug.size(1)]).cuda(x.device)
        padding_h[:] = self.mask_token

        t_rep = torch.cat((self.dec_node_norm(padding_h),pe[mask_nodes]),1)
        t_rep = self.dec_pe_encoder(t_rep)

        t_aug = torch.cat((self.dec_node_norm(h_aug),pe[keep_nodes]),1)
        t_aug = self.dec_pe_encoder(t_aug)
        
        # t_rep = padding_h + self.in_degree_encoder(in_degree)+self.out_degree_encoder(out_degree)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(t_rep)
        aug_rep = self.encoder_to_decoder(t_aug)
        batch_rep = batch[mask_nodes]
        batch_aug_rep = batch[keep_nodes]
        #decoder
        recon = self.decoder(rep,aug_rep,batch_rep,batch_aug_rep)
        x_init = x[mask_nodes[:recon.size(0)]]
        x_rec = recon


        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self,batch_g):
        x,edge_index,batch,in_degree,out_degree,pe = batch_g.x,batch_g.edge_index,batch_g.batch,batch_g.in_degree,batch_g.out_degree,batch_g.pe
        rep = self.encoder_aug(x,edge_index,batch,in_degree,out_degree,pe)
        # rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
