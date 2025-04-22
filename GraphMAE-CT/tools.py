import os
import csv


def graph_results_to_file(args, acc, std):

    if not os.path.exists(f"./{args.saveFold}"):
        print("Create Results File !!!")

        os.makedirs(f"./{args.saveFold}")

    filename = f"./{args.saveFold}/graph-result-{args.dataset}.csv"

    headerList = ["dataset", "enc_type",  "num_layers_enc_gnn","num_layers_enc_trans","num_layers_dec",
                  "pe_dim","mask_rate","aug_ratio","alpha_l","max_epoch","gnn_dropout","attn_drop","aug_type","num_hidden",
                  "acc", "std"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(7)
        if header != "dataset":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4f},{:.4f}\n".format(
            args.dataset, args.encoder, args.num_layers_enc_gnn,args.num_layers_enc_transformer,args.num_layers_dec,args.random_walk_length,
            args.mask_rate,args.aug_ratio,args.alpha_l,args.max_epoch,args.gnn_dropout,args.attn_drop,args.aug_type,args.num_hidden, acc, std
        )
        f.write(line)
