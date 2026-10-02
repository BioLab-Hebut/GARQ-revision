# GARQ revision material

This repository combines the original GARQ implementation with the framework and implementation notes accompanying the revised manuscript, and recovered scripts for revision experiments. The original repository remains [BioLab-Hebut/GARQ](https://github.com/BioLab-Hebut/GARQ).

The five model files at the repository root are preserved byte for byte from original commit `5da45adcd62f1be8ee318d8742c80c59cb242ca2`. Revision controls use separate archived backends; they do not replace the root implementation. See [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for file hashes and [METHODS.md](METHODS.md) for implemented behavior.

## Framework

![GARQ framework](frame.svg)

The editable vector illustration and [matching PDF](frame.pdf) reproduce Figure 1 of the revised manuscript. Each modality is preprocessed separately and encoded cell-wise before concatenation. Quantization uses a batch-local graph, and usage-weighted updates reposition a fixed number of anchors. Aggregated profiles are used for downstream analyses, including MOFA+.

## Run the original model

Use the original [tutorial](GARQ_Tutorial.ipynb) or the checked launcher:

```bash
python tools/run_garq.py --data_file datasets/rna.h5ad datasets/adt.h5ad --data_type RNA ADT --save_name example --n_GARQs 500 --seed 1 --device cuda
```

Input files contain original modality-specific counts, with the same paired cells in the same row order. Cell-type annotations are optional for construction and enable annotation-based evaluation. The launcher validates input alignment and batch/anchor constraints, creates output directories, and calls the unchanged `GARQ.py`.

The original output contains normalized, log-transformed metacell mean profiles and a cell-level embedding/assignment file. [TOOLS.md](TOOLS.md) explains how to restore cell identifiers, feature names and occupied-anchor row metadata into separate copies without changing numerical values.

## Revision experiments

[experiments/README.md](experiments/README.md) lists the available scripts, their individual environments, data requirements and execution entry points. Profiling, training-batch controls and frozen-model inference controls use distinct protocols. The supplied scripts retain those distinctions.

The source package contains code and configurations. Data, model checkpoints, machine logs and large analysis outputs are not included. Public example datasets are available from [Figshare](https://doi.org/10.6084/m9.figshare.32751672). Dataset-specific membership tables and other inputs are described by each workflow. This package does not claim that every historical server experiment has been recovered or rerun locally.

## Environment and verification

`environment.yaml` is the historical Linux environment export with the machine-specific prefix removed. `requirement.ymal` is kept as a compatibility filename with the same contents. The archive records Python 3.11.6, PyTorch 2.1.1, NumPy 1.26.0, Scanpy 1.9.6 and scikit-learn 1.1.3; some experiments also require their documented R/MOFA+ environment.

The export contains CUDA-specific wheel builds and is not a portable installer. Use a compatible CUDA/PyTorch/FAISS environment for the original GPU workflow. Package verification covers Python syntax, configuration references, source hashes, vector assets and dependency-light utility checks. It does not replace execution of the complete training or downstream analyses in the recorded server environment.
