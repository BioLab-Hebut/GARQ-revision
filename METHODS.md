# GARQ implementation notes

These notes describe the computation in the original GARQ implementation (`GARQ.py`, `model.py`, `engine.py` and `data_utils.py`). The revision updates the framework illustration and clarifies the method and reproducibility settings; the model computation is unchanged. The original repository remains available at https://github.com/BioLab-Hebut/GARQ.

## Input and separate preprocessing

Each input is a modality-specific `.h5ad` count matrix. The modalities must contain the same paired cells in the same row order; the loader does not match or reorder cell identifiers. `--data_file` and `--data_type` must have corresponding order. Cell-type annotations are optional for construction and are used for evaluation when present.

`preprocess` in `data_utils.py` processes each modality separately. CSR/CSC matrices are converted to dense arrays. Counts are normalized to a total of 10,000 per cell and transformed with `log1p`; the size factor is the original library size divided by 10,000. RNA uses Scanpy highly variable gene selection (`n_top_genes=3000` when the input has fewer than 5,000 genes; otherwise the Scanpy defaults in the documented environment). ATAC uses the same variability routine with `n_top_genes=30000` to select accessibility peaks. ADT retains all measured protein features. Selected inputs are scaled with `max_value=10`. Original counts of the selected features are retained as reconstruction targets. A separate copy of the normalized, log-transformed matrix is retained before feature selection and scaling for final aggregation.

## Cell-wise encoding and batch-local quantization

Each modality has a linear input projection and a Transformer encoder that produces a 32-dimensional embedding by default. `TransformerEncoder.forward` inserts a sequence dimension of length one: the tensor has shape `batch_size x 1 x embedding_dim`. Attention therefore acts within each cell's single-token representation. Modality embeddings are concatenated after encoding to form a joint embedding of dimension `d = 32 x number_of_modalities`.

For a current batch of `B` embeddings, `build_cell_knn_graph` computes a dense cosine similarity matrix. With the default `k_knn=5`, the implementation takes `k_knn+1` nearest entries, discards the first entry, constructs a directed binary adjacency `A`, and symmetrizes it as `G = (A + A.T)/2`. It does not row-normalize this graph. For `B <= k_knn`, it instead uses an all-ones matrix divided by `B`. The graph is rebuilt within each training or inference batch.

`encode_relation` computes direct cosine similarity `S` between detached normalized cell embeddings and normalized anchors, then uses `S + alpha * G @ S` for assignment. The implemented residual coefficient is 0.2. Specifically, the current code recreates `alpha` with this value inside each call and clamps it to be nonnegative; it is not a persistently learned optimizer parameter. Hard assignment takes the largest score over the `K` anchors. The quantized embedding is the selected anchor vector. The returned assignment confidence is the difference between the two largest scores.

## Anchor usage and repositioning

Anchor initialization uses embeddings from the first two shuffled training batches, before the warm-up loop. FAISS spherical K-means (`gpu=True`) assigns this sample to `K` clusters, and the means of the original sample embeddings initialize the existing anchors.

During each quantized training call, the hard-assignment proportion `p_k` updates usage as `U_k <- 0.9 * U_k + 0.1 * p_k`, starting from zero. Here 0.9 is the EMA decay factor. Repositioning happens in the forward pass in addition to gradient-based optimization of anchors.

The long-distance branch is evaluated on every quantized training batch. For each cell, the sampling weight is the maximum over anchors of `softmax(1 - similarity)`; these nonnegative cell weights are supplied to `torch.multinomial`. The branch samples `K` cell embeddings with replacement and updates each existing anchor with weight `beta_long,k = exp(-100 * K * U_k - 0.001)`.

The local branch runs when `sum(U) + 1e-4 >= 1`. For each anchor it samples one cell from the current batch with weights proportional to `exp(-abs(similarity - median_batch_similarity))`. Its update weight is `beta_local,k = exp(-10 * mean(U) / U_k - 0.001)`. Each branch updates an anchor as `(1 - beta) * anchor + beta * sampled_embedding`. The usage weights are continuous; there are no low/high usage thresholds. Both branches move existing anchors without splitting or creating anchors. The configured number `K` is fixed, but the number of occupied output metacells can be smaller than `K`.

## Decoder branches and objective

Each modality has a continuous decoder receiving its modality embedding and a separate quantized decoder receiving the full joint anchor embedding. Decoder Transformer blocks also receive one token per cell. RNA and ADT decoders predict a positive mean, positive dispersion and a zero-inflation probability. Their reconstruction loss is the zero-inflated negative-binomial (ZINB) negative log-likelihood. ATAC uses a Poisson negative log-likelihood; its predicted dispersion is not used by that objective. Decoder means are multiplied by the cell's size factor when compared with the selected original counts.

Warm-up optimizes the sum of continuous reconstruction losses. Quantized training optimizes the sum, over modalities, of continuous and quantized reconstruction losses plus `mean((quantized_embedding - continuous_embedding.detach())**2)`. Each reconstruction term is averaged over its cell-feature entries. The squared-error term uses detached continuous embeddings. `GARQ.py` uses AdamW with learning rate `1e-3` and weight decay `1e-2`. Defaults are a maximum of 300 epochs and `min(50, int(epoch_limit * 0.2))` warm-up epochs. Early stopping requires both monitored quantized reconstruction and anchor losses to remain within `1e-5` of their retained reference values for 10 consecutive stable epochs. The reconstruction monitor sums the per-batch mean losses and averages over modalities without dividing by the number of batches; the anchor monitor is averaged over batches. The reconstruction monitor scale therefore also depends on the number of batches.

## Batching and reproducibility

The default training batch size is 512, with shuffling, four loader workers, pinned host memory and `drop_last=True`. When `K > 1000` and the requested batch size is at most 512, `load_data` changes the effective training batch size to 4096. Inference uses four times the effective training batch size, canonical input order and `drop_last=False`. The default seed is 1.

Because graph construction and neighborhood smoothing use the current batch, cell-to-anchor membership can depend on inference batch size and input order, even with frozen embeddings and anchors. Changing training batch size also changes the initialization sample and number of optimizer steps. Reproduction should retain the input row order, effective training and inference batch sizes, seed, `K`, neighbor count and software environment. Sparse preprocessing and batch-independent graph construction would be separate implementation changes requiring validation.

## Outputs and downstream analysis

`compute_metacell` averages member-cell profiles separately for each modality using the normalized, log-transformed matrices retained before feature selection and scaling. Empty anchors are excluded. These outputs are mean profiles rather than raw summed count matrices or decoder reconstructions. The assignment file stores continuous cell embeddings and each cell's anchor index; it does not store one latent row per metacell. When annotations are available, output metacell labels are based on member-cell majority labels.

MOFA+ integration, visualization and molecular association analysis occur downstream of GARQ metacell construction. MOFA+ is not called by the core construction scripts. Dataset-specific downstream parameters should be reproduced from the corresponding analysis scripts and recorded configurations; the framework schematic does not establish a common MOFA+ configuration across datasets.

## Computational and resource scope

The encoder does not create a full-dataset cell-cell attention matrix. For current batch size `B`, joint dimension `d` and `K` anchors, the quantizer's dense cell-cell similarity costs `O(B^2 d)`, cell-anchor similarity costs `O(B K d)`, and graph-score multiplication costs `O(B^2 K)`. Graph and assignment working storage includes `O(B^2 + B K + K d)` values, in addition to neural activations, model parameters and optimizer state. There are approximately `N/B` batches per epoch for `N` cells. Host memory also includes dense modality arrays and preprocessing copies. The current molecular aggregation loop costs `O(N K + N * sum(D_m))`, where `D_m` denotes each retained full feature dimension, and produces up to `O(K * sum(D_m))` values. A fixed batch size alone does not imply linear total cost if `K` also increases with `N`.

Historical fitting records and complete-workflow profiles have different accounting definitions. A single end-of-fit `torch.cuda.memory_allocated()` reading measures current-device tensor allocations; a single `process.memory_info().rss` reading measures the current process's resident memory. Neither is a peak, and CPU RSS and GPU allocations should not be merged into one total-memory ranking. Complete-workflow timing can include reading, preprocessing, initialization, training, inference, aggregation and writing, whereas fitting-only timing excludes separate stages. Resource claims must identify the measured stages, memory definition, sampling method and hardware rather than infer them from this schematic.

## Framework illustration

[`frame.svg`](frame.svg) and [`frame.pdf`](frame.pdf) reproduce the revised manuscript's Figure 1. Panel a retains separate preprocessing paths; b shows cell-wise encoding before concatenation; c shows batch-local graph-aware assignment; d shows EMA-based repositioning with fixed anchor identities; e shows continuous and quantized reconstruction branches; and f shows aggregated profiles entering downstream applications outside GARQ. Matrix values and before/after positions are schematic examples, not additional measurements.


