import warnings

warnings.filterwarnings("ignore")

import os
import torch
import random
import argparse
import matplotlib
import numpy as np
import scanpy as sc
import seaborn as sns
from model import GARQ  
from matplotlib import rcParams
from alive_progress import alive_bar
from engine import train_one_epoch, warm_one_epoch, inference, init_gart_anchors
from data_utils import load_data, compute_metacell
from eval_utils import (
    plot_metacell_umap,
    plot_metacell_size,
    plot_celltype_purity,
    plot_compactness_separation,
)


def main(args):
    device = torch.device(args.device)

    adata_list, dataloader_train, dataloader_eval, input_dims = load_data(args)
    omics_num = len(adata_list)

    print("n_GARQs:", args.n_GARQs)

    net = GARQ(
        input_dims=input_dims,
        data_types=args.data_type,
        entry_num=args.n_GARQs,
        k_knn=args.k_knn  
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    print("======= Training Start =======")

    warm_epochs = min(50, int(args.epoch * 0.2))
    print("warm_epochs:", warm_epochs)

    init_gart_anchors(
        model=net,
        data_types=args.data_type,
        dataloader=dataloader_train,  
        device=device,
        init_method=args.anchers_init
    )
    if omics_num == 1:
        net.copy_decoder_q()

    with alive_bar(args.epoch, enrich_print=False) as bar:
        loss_rec_his = loss_vq_his = 1e7
        stable_epochs = 0
        
        for epoch in range(args.epoch):
            bar()
            if epoch < warm_epochs:
                warm_one_epoch(
                    model=net,
                    data_types=args.data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
            else:
                loss_rec, loss_vq = train_one_epoch(
                    model=net,
                    data_types=args.data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
                converge = (abs(loss_vq_his - loss_vq) <= 1e-5) and (
                    abs(loss_rec_his - loss_rec) <= 1e-5
                )
                if converge:
                    stable_epochs += 1
                    if stable_epochs >= args.converge_threshold:
                        print("Early Stopping.")
                        break
                else:
                    stable_epochs = 0
                    loss_rec_his = loss_rec
                    loss_vq_his = loss_vq


    print("======= Training Done =======")

    print("")

    print("======= Inference Start =======")
    embeds, ids, delta_confs, rec_q_percent, loss_ancher = inference(
        model=net, data_types=args.data_type, data_loader=dataloader_eval, device=device
    )

    print("* Quantized Recon Percent:", rec_q_percent)
    # print("* Metacell Delta Assignment Confidence:", np.mean(delta_confs))
    print("* Anchers Loss:", loss_ancher)
    print("")

    if not os.path.exists("./save/"):
        os.makedirs("./save/")

    for i in range(omics_num):
        metacell_path = (
            "./save/"
            + args.save_name
            + "_"
            + args.data_type[i]
            + "_"
            + str(args.n_GARQs)
            + "metacell_k"
            + str(args.k_knn)
            + ".h5ad"
        )
        adata = adata_list[i]
        metacell_adata = compute_metacell(adata, ids, args)
        metacell_adata.write_h5ad(metacell_path)
        print(args.data_type[i] + " metacell data saved at:", metacell_path)

    assignment_path = (
        "./save/" + args.save_name + "_" + str(args.n_GARQs) + "metacell_k" + str(args.k_knn) + "_ids.h5ad"
    )
    adata = sc.AnnData(embeds, dtype=np.float32)
    adata.obs["metacell"] = ids
    if args.type_key in adata_list[0].obs_keys():
        adata.obs[args.type_key] = adata_list[0].obs[args.type_key].values

    sc.set_figure_params(figsize=(7, 7), dpi=300)
    sc.pp.neighbors(adata, use_rep="X", metric="cosine")
    sc.tl.umap(adata)
    if args.type_key in adata.obs_keys():
        sc.pl.umap(
            adata,
            color=[args.type_key],
            save="_" + args.save_name + "_embedding_k" + str(args.k_knn) + ".png",
            palette=sns.color_palette(
                "husl", np.unique(adata.obs[args.type_key].values).size
            ),
            show=False,
        )
    rcParams.update(matplotlib.rcParamsDefault)
    adata.write_h5ad(assignment_path)
    print("Metacell assignment saved at:", assignment_path)
    print("")


    fig_save_name = args.save_name + "_" + str(args.n_GARQs) + "metacell_k" + str(args.k_knn)
    plot_metacell_umap(adata, fig_save_name)
    plot_metacell_size(adata, fig_save_name)
    if args.type_key in adata.obs_keys():
        plot_celltype_purity(adata, adata.obs[args.type_key], fig_save_name)
    for i in range(omics_num):
        plot_compactness_separation(
            dataloader_train.dataset.raw_list[i].numpy(),
            adata,
            fig_save_name + "_" + args.data_type[i],
            omics_num > 1,
        )

    print("======= Inference Done =======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", nargs="+", type=str)
    parser.add_argument(
        "--data_type", nargs="+", type=str, choices=["RNA", "ADT", "ATAC"]
    )
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--n_GARQs", type=int)

    parser.add_argument("--type_key", type=str, default="celltype")

    parser.add_argument(
        "--anchers_init", 
        type=str,
        default="Kmeans",
    )
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--converge_threshold", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--k_knn",
        type=int,
        default=5,
        help="Number of neighbors for cell KNN graph (default: 5, recommended 5-10)"
    )

    args = parser.parse_args()

    assert len(args.data_file) == len(
        args.data_type
    ), "There is a mismatch between the number of data paths and the number of data types"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    main(args)
