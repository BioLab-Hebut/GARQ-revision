"""Input paths and recorded analysis settings."""
CONFIG = {'datasets': {'D5': {'config': {'n': 12103,
                                'mods': ['RNA', 'ADT'],
                                'paths': ['${GARQ_INPUT_ROOT}/inputs/3377fe9f9a60_D8_rna.h5ad',
                                          '${GARQ_INPUT_ROOT}/inputs/4deb5a2b96ec_D8_adt.h5ad'],
                                'fixed': '${GARQ_INPUT_ROOT}/inputs/bd04e3436a4f_D8_fixed_embedding.h5ad'},
                     'fixed': {'ids_match': True,
                               'labels_equal_in_order': True,
                               'uns': {'fixed_representation': "{'adt': 'CLR + PCA20', 'atac': 'TF-IDF + "
                                                               "LSI50', 'block_processing': 'standardize "
                                                               'each block + per-cell L2 normalization + '
                                                               "equal-weight concatenation', 'blocks': "
                                                               "array(['X_rna_pca', 'X_adt_pca'], "
                                                               "dtype=object), 'dataset': 'D8', 'pair': "
                                                               "'RNA_ADT', 'random_state': 0, 'rna': "
                                                               "'normalize_total + log1p + HVG + PCA50'}"}}},
              'D11': {'config': {'n': 9631,
                                 'mods': ['RNA', 'ATAC'],
                                 'paths': ['${GARQ_INPUT_ROOT}/inputs/519de1da8e3e_pbmc10k_rna.h5ad',
                                           '${GARQ_INPUT_ROOT}/inputs/463ed4bb3662_pbmc10k_atac.h5ad'],
                                 'fixed': '${GARQ_INPUT_ROOT}/inputs/c86e79ee7879_pbmc10k_fixed_embedding.h5ad'},
                      'fixed': {'ids_match': True,
                                'labels_equal_in_order': True,
                                'uns': {'fixed_representation': "{'adt': 'CLR + PCA20', 'atac': 'TF-IDF + "
                                                                "LSI50', 'block_processing': 'standardize "
                                                                'each block + per-cell L2 normalization + '
                                                                "equal-weight concatenation', 'blocks': "
                                                                "array(['X_rna_pca', 'X_atac_lsi'], "
                                                                "dtype=object), 'dataset': 'pbmc10k', "
                                                                "'pair': 'RNA_ATAC', 'random_state': 0, "
                                                                "'rna': 'normalize_total + log1p + HVG + "
                                                                "PCA50'}"}}}},
 'source_sha256': {'model.py': 'c629f59de63f4d6e3b3554a3a2b3c425b504ec4b92134167b60407b3dbbac1e8',
                   'engine.py': '25a2ba5fda0c4ed06602d65ceef3a061f8bdf9c021740aa59fbde05198498765',
                   'data_utils.py': 'f86c775960003b070d5fcd71f789bdc7d53357bae16db21b1b54aedd5760e97a',
                   'GARQ.py': '49c97de6440cd8ed76eca677a9225214752856461cc27930a0d0a465ae3179ab',
                   'eval_utils.py': 'e38819dc53085912c128401c852e86382f2d52abf1cb0e028ff9a72c43bbf9d8'}}
