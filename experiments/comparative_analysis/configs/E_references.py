"""Runtime scientific settings. Edit local paths only; preserve conditions and hashes."""
CONFIG = [{'dataset': 'D5',
  'references': [{'name': 'X_rna_pca',
                  'path': '${GARQ_INPUT_ROOT}/inputs/cdaa5b10bbba_X_rna_pca.npy',
                  'sha256': '39c6762274532c74fbcf8319a2a104e5810154282b6488893ed759245848ceb5',
                  'cell_index_path': '${GARQ_INPUT_ROOT}/inputs/70c868d7812c_cell_index.csv',
                  'cell_index_sha256': '2bb0d0cdaa26f09ba114358ca9164fd6253d4e2de51222c4118e6d2ee33e7f4b',
                  'independent_of_runs': True,
                  'feature_coordinate_identity': 'Archived standard representation; X_rna_pca; '
                                                 'column-standardized and per-cell L2-normalized before '
                                                 'original save. Same unchanged coordinates for all R66 '
                                                 'runs; no new PCA, LSI, reference fit or partition.',
                  'provenance_path': '${GARQ_INPUT_ROOT}/inputs/80995df686db_E1B_reference_verification_20261020.json',
                  'provenance_sha256': 'bf6ef7523b090d8bc8f7652cb31c5ae5794461224209e3a3010e6e5bcaf2a28c'},
                 {'name': 'X_adt_pca',
                  'path': '${GARQ_INPUT_ROOT}/inputs/b05c05f5ed6a_X_adt_pca.npy',
                  'sha256': '8745b42d93bc16b6e9d72cf0da9a81b3a7b71597ce6ae2d4e359c739decdb381',
                  'cell_index_path': '${GARQ_INPUT_ROOT}/inputs/70c868d7812c_cell_index.csv',
                  'cell_index_sha256': '2bb0d0cdaa26f09ba114358ca9164fd6253d4e2de51222c4118e6d2ee33e7f4b',
                  'independent_of_runs': True,
                  'feature_coordinate_identity': 'Archived standard representation; X_adt_pca; '
                                                 'column-standardized and per-cell L2-normalized before '
                                                 'original save. Same unchanged coordinates for all R66 '
                                                 'runs; no new PCA, LSI, reference fit or partition.',
                  'provenance_path': '${GARQ_INPUT_ROOT}/inputs/80995df686db_E1B_reference_verification_20261020.json',
                  'provenance_sha256': 'bf6ef7523b090d8bc8f7652cb31c5ae5794461224209e3a3010e6e5bcaf2a28c'}]},
 {'dataset': 'D11',
  'references': [{'name': 'X_rna_pca',
                  'path': '${GARQ_INPUT_ROOT}/inputs/02dfde8bb114_X_rna_pca.npy',
                  'sha256': 'f8e33bb1903408317506615d7ed4cba410760f46a0931209e8a9bc7d0628a35b',
                  'cell_index_path': '${GARQ_INPUT_ROOT}/inputs/570282d09191_cell_index.csv',
                  'cell_index_sha256': '1fc661b8967dfe361f8c7bd5de12e20101f68f0e63dc15a985104958f68d19fd',
                  'independent_of_runs': True,
                  'feature_coordinate_identity': 'Archived standard representation; X_rna_pca; '
                                                 'column-standardized and per-cell L2-normalized before '
                                                 'original save. Same unchanged coordinates for all R66 '
                                                 'runs; no new PCA, LSI, reference fit or partition.',
                  'provenance_path': '${GARQ_INPUT_ROOT}/inputs/80995df686db_E1B_reference_verification_20261020.json',
                  'provenance_sha256': 'bf6ef7523b090d8bc8f7652cb31c5ae5794461224209e3a3010e6e5bcaf2a28c'},
                 {'name': 'X_atac_lsi',
                  'path': '${GARQ_INPUT_ROOT}/inputs/0a3b0dc1842e_X_atac_lsi.npy',
                  'sha256': '1b3b9a1960cc643988c4704f999efa4e37dfec91ad2f071edfb30548812063b6',
                  'cell_index_path': '${GARQ_INPUT_ROOT}/inputs/570282d09191_cell_index.csv',
                  'cell_index_sha256': '1fc661b8967dfe361f8c7bd5de12e20101f68f0e63dc15a985104958f68d19fd',
                  'independent_of_runs': True,
                  'feature_coordinate_identity': 'Archived standard representation; X_atac_lsi; '
                                                 'column-standardized and per-cell L2-normalized before '
                                                 'original save. Same unchanged coordinates for all R66 '
                                                 'runs; no new PCA, LSI, reference fit or partition.',
                  'provenance_path': '${GARQ_INPUT_ROOT}/inputs/80995df686db_E1B_reference_verification_20261020.json',
                  'provenance_sha256': 'bf6ef7523b090d8bc8f7652cb31c5ae5794461224209e3a3010e6e5bcaf2a28c'}]}]
