import warnings
warnings.simplefilter("ignore")

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import MuData
import os


def check_feature_selection(adata, modality_name, n_metacells, 
                            min_expected=None, max_expected=None):
    """
    Check if the number of selected features is reasonable
    """
    n_selected = adata.var['highly_variable'].sum()
    print(f"  ✓ Selected {n_selected} {modality_name} highly variable features")
    
    # Set default expected ranges based on modality
    if min_expected is None:
        if modality_name == "RNA":
            min_expected = 1500
            max_expected = 4000
        elif modality_name == "ATAC":
            min_expected = 2000
            max_expected = 6000
        elif modality_name == "ADT":
            min_expected = 10  # ADT通常特征少
            max_expected = 200
    
    # Check if in reasonable range
    if n_selected < min_expected:
        print(f"  ⚠️  WARNING: Too few {modality_name} features!")
        print(f"      Recommended range: {min_expected}-{max_expected}")
        print(f"      Suggestion: Try decreasing min_mean and min_disp")
        return False
    elif n_selected > max_expected:
        print(f"  ⚠️  WARNING: Too many {modality_name} features!")
        print(f"      Recommended range: {min_expected}-{max_expected}")
        print(f"      Suggestion: Try increasing min_mean and min_disp")
        return False
    else:
        print(f"  ✓ Feature count is reasonable ({min_expected}-{max_expected})")
        return True


def main(args):
    print("=" * 60)
    print("Starting Trimodal Metacell MOFA+ Integration")
    print("RNA + ATAC + ADT")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # ========== Load Metacell Data ==========
    print("\n[1/4] Loading metacell data...")
    
    # Load RNA metacells
    print("  - Loading RNA metacells...")
    rna_metacell = sc.read_h5ad(args.metacell_rna_path)
    print(f"    → RNA metacells: {rna_metacell.n_obs}")
    print(f"    → RNA features: {rna_metacell.n_vars}")
    n_metacells = rna_metacell.n_obs
    
    # Load ATAC metacells
    print("  - Loading ATAC metacells...")
    atac_metacell = sc.read_h5ad(args.metacell_atac_path)
    print(f"    → ATAC metacells: {atac_metacell.n_obs}")
    print(f"    → ATAC features: {atac_metacell.n_vars}")
    
    # Load ADT metacells
    print("  - Loading ADT metacells...")
    adt_metacell = sc.read_h5ad(args.metacell_adt_path)
    print(f"    → ADT metacells: {adt_metacell.n_obs}")
    print(f"    → ADT features: {adt_metacell.n_vars}")
    
    # Verify same number of metacells
    assert rna_metacell.n_obs == atac_metacell.n_obs == adt_metacell.n_obs, \
        "All three modalities must have the same number of metacells!"
    
    # ========== Preprocess RNA Metacells ==========
    print("\n[2/4] Preprocessing RNA metacells...")
    
    # Save raw counts
    if 'counts' not in rna_metacell.layers:
        rna_metacell.layers['counts'] = rna_metacell.X.copy()
    
    # Normalize
    print(f"  → Normalizing to target_sum={args.target_sum}")
    sc.pp.normalize_total(rna_metacell, target_sum=args.target_sum)
    
    # Log-transform
    print(f"  → Log-transforming")
    sc.pp.log1p(rna_metacell)
    rna_metacell.layers['lognorm'] = rna_metacell.X.copy()
    
    # Feature selection
    print(f"  → Selecting highly variable genes")
    print(f"     Parameters: min_mean={args.rna_min_mean}, max_mean={args.rna_max_mean}, min_disp={args.rna_min_disp}")
    
    if args.rna_n_top_genes is not None:
        print(f"     Forcing selection of top {args.rna_n_top_genes} genes")
        sc.pp.highly_variable_genes(
            rna_metacell, 
            n_top_genes=args.rna_n_top_genes,
            flavor='seurat_v3'
        )
    else:
        sc.pp.highly_variable_genes(
            rna_metacell, 
            min_mean=args.rna_min_mean, 
            max_mean=args.rna_max_mean, 
            min_disp=args.rna_min_disp,
            flavor='seurat'
        )
    
    rna_ok = check_feature_selection(rna_metacell, "RNA", n_metacells)
    
    # Scale
    print(f"  → Scaling (z-score, clipping at ±{args.max_scale_value})")
    sc.pp.scale(rna_metacell, max_value=args.max_scale_value)
    
    # ========== Preprocess ATAC Metacells ==========
    print("\n[3/4] Preprocessing ATAC metacells...")
    
    # Save raw counts
    if 'counts' not in atac_metacell.layers:
        atac_metacell.layers['counts'] = atac_metacell.X.copy()
    
    # Normalize
    print(f"  → Normalizing to target_sum={args.target_sum}")
    sc.pp.normalize_total(atac_metacell, target_sum=args.target_sum)
    
    # Log-transform
    print(f"  → Log-transforming")
    sc.pp.log1p(atac_metacell)
    atac_metacell.layers['lognorm'] = atac_metacell.X.copy()
    
    # Feature selection
    print(f"  → Selecting highly variable peaks")
    print(f"     Parameters: min_mean={args.atac_min_mean}, max_mean={args.atac_max_mean}, min_disp={args.atac_min_disp}")
    
    if args.atac_n_top_genes is not None:
        print(f"     Forcing selection of top {args.atac_n_top_genes} peaks")
        sc.pp.highly_variable_genes(
            atac_metacell, 
            n_top_genes=args.atac_n_top_genes,
            flavor='seurat_v3'
        )
    else:
        sc.pp.highly_variable_genes(
            atac_metacell, 
            min_mean=args.atac_min_mean, 
            max_mean=args.atac_max_mean, 
            min_disp=args.atac_min_disp,
            flavor='seurat'
        )
    
    atac_ok = check_feature_selection(atac_metacell, "ATAC", n_metacells)
    
    # Scale
    print(f"  → Scaling (z-score, clipping at ±{args.max_scale_value})")
    sc.pp.scale(atac_metacell, max_value=args.max_scale_value)
    
    # ========== Preprocess ADT Metacells ==========
    print("\n[3.5/4] Preprocessing ADT metacells...")
    
    # Save raw counts
    if 'counts' not in adt_metacell.layers:
        adt_metacell.layers['counts'] = adt_metacell.X.copy()
    
    # ADT-specific preprocessing: CLR transformation (recommended)
    if args.adt_use_clr:
        print("  → Applying CLR (Centered Log-Ratio) normalization")
        from scipy.stats import gmean
        
        # Get data as array
        adt_data = adt_metacell.X.toarray() if hasattr(adt_metacell.X, 'toarray') else adt_metacell.X
        
        # Add pseudocount to avoid log(0)
        adt_data = adt_data + 1
        
        # Calculate geometric mean per cell (row)
        geo_means = gmean(adt_data, axis=1, keepdims=True)
        
        # Apply CLR transformation
        adt_metacell.X = np.log(adt_data / geo_means)
        adt_metacell.layers['clr'] = adt_metacell.X.copy()
    else:
        # Standard normalization
        print("  → Applying standard normalization")
        sc.pp.normalize_total(adt_metacell, target_sum=args.target_sum)
        sc.pp.log1p(adt_metacell)
        adt_metacell.layers['lognorm'] = adt_metacell.X.copy()
    
    # Feature selection for ADT (optional, usually keep all proteins)
    if args.adt_feature_selection:
        print(f"  → Selecting highly variable proteins")
        print(f"     Parameters: min_mean={args.adt_min_mean}, max_mean={args.adt_max_mean}, min_disp={args.adt_min_disp}")
        
        if args.adt_n_top_genes is not None:
            print(f"     Forcing selection of top {args.adt_n_top_genes} proteins")
            sc.pp.highly_variable_genes(
                adt_metacell, 
                n_top_genes=args.adt_n_top_genes,
                flavor='seurat_v3'
            )
        else:
            sc.pp.highly_variable_genes(
                adt_metacell, 
                min_mean=args.adt_min_mean, 
                max_mean=args.adt_max_mean, 
                min_disp=args.adt_min_disp,
                flavor='seurat'
            )
        
        adt_ok = check_feature_selection(adt_metacell, "ADT", n_metacells)
    else:
        print("  → Using all ADT features (no feature selection)")
        adt_metacell.var['highly_variable'] = True
        adt_ok = True
    
    # Scale ADT data
    print(f"  → Scaling (z-score, clipping at ±{args.max_scale_value})")
    sc.pp.scale(adt_metacell, max_value=args.max_scale_value)
    
    # ========== Feature Selection Summary ==========
    if not (rna_ok and atac_ok and adt_ok):
        print("\n" + "⚠️ " * 20)
        print("WARNING: Feature selection may not be optimal!")
        print("Consider adjusting preprocessing parameters.")
        print("⚠️ " * 20)
    
    # ========== Run MOFA+ ==========
    print("\n[4/4] Running MOFA+ trimodal integration...")
    
    # Create MuData object with three modalities
    mdata = MuData({
        'rna': rna_metacell, 
        'atac': atac_metacell,
        'adt': adt_metacell
    })
    print(f"  ✓ MuData created with {mdata.n_obs} metacells")
    print(f"  ✓ Modalities: {list(mdata.mod.keys())}")
    
    # Print MOFA+ configuration
    print(f"\n  - MOFA+ Configuration:")
    print(f"      n_factors: {args.n_factors}")
    print(f"      gpu_mode: {args.gpu_mode}")
    print(f"      convergence_mode: {args.convergence_mode}")
    print(f"      groups_label: None (no batch correction)")
    
    # Run MOFA+
    print("\n  - Performing MOFA+ decomposition...")
    try:
        mu.tl.mofa(
            mdata,
            use_var=None,  # Use highly variable features
            groups_label=None,  # No batch information
            gpu_mode=args.gpu_mode,
            save_data=False,
            n_factors=args.n_factors,
            verbose=True
        )
        print(f"  ✓ MOFA+ completed successfully!")
        print(f"  ✓ Latent dimensions: {mdata.obsm['X_mofa'].shape}")
    except Exception as e:
        print(f"  ✗ MOFA+ failed with error: {e}")
        print("  Trying with default parameters...")
        mu.tl.mofa(
            mdata,
            n_factors=args.n_factors,
            verbose=True
        )
        print(f"  ✓ MOFA+ completed with default settings!")
        print(f"  ✓ Latent dimensions: {mdata.obsm['X_mofa'].shape}")
    
    # ========== Save Results ==========
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    # Save MOFA+ latent representation (embedding)
    latent_filename = f"{os.path.basename(args.output_path).replace('.h5ad', '')}_mofa_latent_seacell.csv"
    latent_output_path = os.path.join(args.output_dir, latent_filename)
    np.savetxt(latent_output_path, mdata.obsm['X_mofa'], delimiter=',')
    print(f"  ✓ MOFA+ latent (factor scores) saved to: {latent_output_path}")
    
    # Save integrated MuData object
    mdata.write(args.output_path)
    print(f"  ✓ Integrated MuData saved to: {args.output_path}")
    
    # Save individual modality data (for downstream analysis)
    rna_output = args.output_path.replace('.h5ad', '_rna.h5ad')
    atac_output = args.output_path.replace('.h5ad', '_atac.h5ad')
    adt_output = args.output_path.replace('.h5ad', '_adt.h5ad')
    
    rna_metacell.write(rna_output)
    atac_metacell.write(atac_output)
    adt_metacell.write(adt_output)
    print(f"  ✓ Individual modalities saved:")
    print(f"      - RNA: {rna_output}")
    print(f"      - ATAC: {atac_output}")
    print(f"      - ADT: {adt_output}")
    
    # ========== Summary Statistics ==========
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total metacells: {mdata.n_obs}")
    print(f"\nRNA modality:")
    print(f"  - Total features: {mdata['rna'].n_vars}")
    print(f"  - Highly variable: {mdata['rna'].var['highly_variable'].sum()}")
    print(f"\nATAC modality:")
    print(f"  - Total features: {mdata['atac'].n_vars}")
    print(f"  - Highly variable: {mdata['atac'].var['highly_variable'].sum()}")
    print(f"\nADT modality:")
    print(f"  - Total features: {mdata['adt'].n_vars}")
    print(f"  - Highly variable: {mdata['adt'].var['highly_variable'].sum()}")
    print(f"\nMOFA+ factors: {mdata.obsm['X_mofa'].shape[1]}")
    

    
    print("\n" + "=" * 60)
    print("Trimodal Integration Complete!")
    print("=" * 60)
    print("\nNext steps for analysis:")
    print("  1. Load the MuData object: mdata = mu.read('{args.output_path}')")
    print("  2. Access MOFA embeddings: mdata.obsm['X_mofa']")
    print("  3. Perform clustering on MOFA embeddings")
    print("  4. Extract factor weights to identify modality-specific features")
    print("  5. Perform downstream analysis (trajectory, differential factors, etc.)")


# rna = sc.read('/workspace/garq/vscode/MetaQ-main/ours/A_xiaorong/A_ag_gart/bash/save/GSE158013637_RNA_637metacell_k5.h5ad')
# atac = sc.read('/workspace/garq/vscode/MetaQ-main/ours/A_xiaorong/A_ag_gart/bash/save/GSE158013637_ATAC_637metacell_k5.h5ad')
# adt = sc.read('/workspace/garq/vscode/MetaQ-main/ours/A_xiaorong/A_ag_gart/bash/save/GSE158013637_ADT_637metacell_k5.h5ad')
# meta = sc.read('/workspace/garq/vscode/MetaQ-main/ours/A_xiaorong/A_ag_gart/bash/save/GSE158013637_637metacell_k5_ids.h5ad')

# rna_ad = sc.read('/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad')
# atac_ad = sc.read('/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_atac.h5ad')
# adt_ad = sc.read('/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrate RNA + ATAC + ADT metacells using MOFA+ (No Batch Correction)"
    )
    
    # ==================== Data Paths ====================
    parser.add_argument(
        "--metacell_rna_path", 
        type=str, 
        default="/workspace/garq/results/3_seacell/three/GSE158013/GSE158013_rna_metacells.h5ad",
        help="Path to RNA metacell h5ad file"
    )
    parser.add_argument(
        "--metacell_atac_path", 
        type=str, 
        default="/workspace/garq/results/3_seacell/three/GSE158013/GSE158013_atac_metacells.h5ad",
        help="Path to ATAC metacell h5ad file"
    )
    parser.add_argument(
        "--metacell_adt_path", 
        type=str, 
        default="/workspace/garq/results/3_seacell/three/GSE158013/GSE158013_adt_metacells.h5ad",
        help="Path to ADT metacell h5ad file"
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        default="./save/trimodal_mofa_integrated.h5ad",
        help="Path to save integrated MuData results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results/trimodal_mofa",
        help="Directory to save outputs and figures"
    )
    
    # ==================== General Preprocessing ====================
    parser.add_argument(
        "--target_sum",
        type=float,
        default=1e4,
        help="Target sum for normalization (default: 10000)"
    )
    parser.add_argument(
        "--max_scale_value",
        type=float,
        default=10,
        help="Maximum value for scaling (clip outliers)"
    )
    
    # ==================== MOFA+ Parameters ====================
    parser.add_argument(
        "--n_factors", 
        type=int, 
        default=10,
        help="Number of factors for MOFA+ (recommended: 15-20 for trimodal)"
    )
    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="medium",
        choices=["fast", "medium", "slow"],
        help="Convergence mode (slow = more iterations, better quality)"
    )
    parser.add_argument(
        "--gpu_mode", 
        action="store_true",
        help="Use GPU for MOFA+ (if available)"
    )
    
    # ==================== RNA Preprocessing ====================
    parser.add_argument(
        "--rna_min_mean", 
        type=float, 
        default=0.1,
        help="Min mean expression for RNA HVG selection"
    )
    parser.add_argument(
        "--rna_max_mean", 
        type=float, 
        default=3.0,
        help="Max mean expression for RNA HVG selection"
    )
    parser.add_argument(
        "--rna_min_disp", 
        type=float, 
        default=0.8,
        help="Min dispersion for RNA HVG selection"
    )
    parser.add_argument(
        "--rna_n_top_genes",
        type=int,
        default=2500,
        help="Force selection of top N genes (recommended: 2000-3000 for metacells)"
    )
    
    # ==================== ATAC Preprocessing ====================
    parser.add_argument(
        "--atac_min_mean", 
        type=float, 
        default=0.1,
        help="Min mean accessibility for ATAC HVP selection"
    )
    parser.add_argument(
        "--atac_max_mean", 
        type=float, 
        default=1.2,
        help="Max mean accessibility for ATAC HVP selection"
    )
    parser.add_argument(
        "--atac_min_disp", 
        type=float, 
        default=0.8,
        help="Min dispersion for ATAC HVP selection"
    )
    parser.add_argument(
        "--atac_n_top_genes",
        type=int,
        default=3500,
        help="Force selection of top N peaks (recommended: 3000-5000 for metacells)"
    )
    
    # ==================== ADT Preprocessing ====================
    parser.add_argument(
        "--adt_use_clr", 
        action="store_true", 
        default=True,
        help="Use CLR (Centered Log-Ratio) normalization for ADT (recommended)"
    )
    parser.add_argument(
        "--adt_feature_selection", 
        action="store_true",
        default=False,
        help="Perform feature selection on ADT (usually keep all proteins)"
    )
    parser.add_argument(
        "--adt_min_mean", 
        type=float, 
        default=0.01,
        help="Min mean for ADT feature selection (if enabled)"
    )
    parser.add_argument(
        "--adt_max_mean", 
        type=float, 
        default=3.0,
        help="Max mean for ADT feature selection (if enabled)"
    )
    parser.add_argument(
        "--adt_min_disp", 
        type=float, 
        default=0.5,
        help="Min dispersion for ADT feature selection (if enabled)"
    )
    parser.add_argument(
        "--adt_n_top_genes",
        type=int,
        default=None,
        help="Force selection of top N proteins (if feature selection enabled)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print("Input data:")
    print(f"  - RNA metacells: {args.metacell_rna_path}")
    print(f"  - ATAC metacells: {args.metacell_atac_path}")
    print(f"  - ADT metacells: {args.metacell_adt_path}")
    print(f"\nOutput:")
    print(f"  - Integrated data: {args.output_path}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"\nRNA preprocessing:")
    print(f"  - min_mean={args.rna_min_mean}, max_mean={args.rna_max_mean}, min_disp={args.rna_min_disp}")
    if args.rna_n_top_genes:
        print(f"  - Forcing top {args.rna_n_top_genes} genes")
    print(f"\nATAC preprocessing:")
    print(f"  - min_mean={args.atac_min_mean}, max_mean={args.atac_max_mean}, min_disp={args.atac_min_disp}")
    if args.atac_n_top_genes:
        print(f"  - Forcing top {args.atac_n_top_genes} peaks")
    print(f"\nADT preprocessing:")
    print(f"  - use_clr={args.adt_use_clr}")
    print(f"  - feature_selection={args.adt_feature_selection}")
    if args.adt_feature_selection and args.adt_n_top_genes:
        print(f"  - Forcing top {args.adt_n_top_genes} proteins")
    print(f"\nMOFA+ settings:")
    print(f"  - n_factors: {args.n_factors}")
    print(f"  - convergence_mode: {args.convergence_mode}")
    print(f"  - gpu_mode: {args.gpu_mode}")
    print("=" * 60 + "\n")
    
    main(args)