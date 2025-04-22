from .model_contra import PreModel

def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    trans_num_heads = args.num_trans_heads
    num_hidden = args.num_hidden
    ffn_dim = args.num_hidden
    random_walk_length = args.random_walk_length
    num_layers_enc_gnn = args.num_layers_enc_gnn
    num_layers_enc_transformer = args.num_layers_enc_transformer
    num_layers_dec = args.num_layers_dec
    edge_type = args.edge_type
    n_edge_features = args.n_edge_features

    residual = args.residual
    attn_drop = args.attn_drop
    gnn_dropout = args.gnn_dropout
    trans_dropout = args.trans_dropout
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    random_walk_length = args.random_walk_length

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features
    aug_type = args.aug_type
    aug_ratio = args.aug_ratio
    degree_num_embeddings = args.degree_num_embeddings

    model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        ffn_dim=int(ffn_dim),
        num_layers_enc_gnn=num_layers_enc_gnn,
        num_layers_enc_transformer = num_layers_enc_transformer,
        num_layers_dec = num_layers_dec,
        nhead=num_heads,
        nhead_out=num_out_heads,
        trans_num_heads = trans_num_heads,
        activation=activation,
        gnn_dropout=gnn_dropout,
        trans_dropout = trans_dropout,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        edge_type=edge_type,
        residual=residual,
        encoder_type=encoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        random_walk_length =random_walk_length,
        aug_type = aug_type,
        aug_ratio = aug_ratio,
        concat_hidden=concat_hidden,
        n_edge_features=n_edge_features,
        degree_num_embeddings=degree_num_embeddings
    )
    return model
