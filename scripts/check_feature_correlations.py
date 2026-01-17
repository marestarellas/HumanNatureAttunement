"""
Script to check correlations between EEG features.

This script loads feature data and plots correlation matrices to verify
relationships between features (PSD bands and entropy measures).

Usage:
    python check_feature_correlations.py --subjects 2 3 4 5 6 --condition VIZ
    python check_feature_correlations.py --subjects 2 3 4 5 6 --condition MULTI
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_subject_data(subject_id, condition, data_dir):
    """Load feature data for one subject and condition."""
    subject_folder = f'sub-{subject_id:02d}'
    feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
    
    if not feature_file.exists():
        print(f"  Warning: {feature_file} not found")
        return None
    
    df = pd.read_csv(feature_file)
    df['subject_id'] = subject_id
    return df


def load_all_data(subjects, condition, data_dir):
    """Load and combine data from multiple subjects for one condition."""
    print(f"\nLoading data for subjects {subjects}, condition: {condition}")
    
    dfs = []
    for subject_id in subjects:
        df = load_subject_data(subject_id, condition, data_dir)
        if df is not None:
            print(f"  Subject {subject_id:02d}: {len(df)} observations")
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No data loaded!")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(df_combined)} observations")
    
    return df_combined


def plot_correlation_matrix(df, features, title, output_file):
    """Plot correlation matrix for selected features."""
    
    # Select only the features we want
    df_features = df[features].copy()
    
    # Remove rows with any NaN
    df_features = df_features.dropna()
    
    print(f"\nComputing correlations for {len(df_features)} observations")
    print(f"Features: {features}")
    
    # Compute correlation matrix
    corr_matrix = df_features.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Pearson Correlation'},
                ax=ax, annot_kws={'size': 8})
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return corr_matrix


def plot_pairwise_scatter(df, features, condition, output_dir):
    """Plot pairwise scatter plots for entropy features."""
    
    df_clean = df[features].dropna()
    
    # Create pairplot
    print(f"\nCreating pairwise scatter plots...")
    
    g = sns.pairplot(df_clean, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
    g.fig.suptitle(f'Pairwise Feature Relationships - {condition}', 
                   y=1.02, fontsize=14, fontweight='bold')
    
    output_file = output_dir / f'pairplot_{condition}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check feature correlations')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs to include')
    parser.add_argument('--condition', type=str, required=True,
                       help='Condition to analyze (e.g., VIZ, AUD, MULTI)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    if args.output_dir is None:
        output_dir = project_dir / 'results' / f'feature_correlations_{args.condition}'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    print(f"Condition: {args.condition}")
    print(f"Subjects: {args.subjects}")
    print(f"Output: {output_dir}")
    
    # Load data
    df = load_all_data(args.subjects, args.condition, data_dir)
    
    # Define feature groups
    psd_rel = ['delta_rel', 'theta_rel', 'alpha_rel', 'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
    psd_abs = ['delta_abs', 'theta_abs', 'alpha_abs', 'low_beta_abs', 'high_beta_abs', 'gamma1_abs']
    entropy = ['lzc', 'perm_entropy', 'spectral_entropy', 'svd_entropy', 'sample_entropy']
    
    # Filter to features that exist in data
    psd_rel = [f for f in psd_rel if f in df.columns]
    psd_abs = [f for f in psd_abs if f in df.columns]
    entropy = [f for f in entropy if f in df.columns]
    
    # 1. Entropy features correlation
    if entropy:
        print("\n" + "="*60)
        print("ENTROPY FEATURES")
        print("="*60)
        corr_entropy = plot_correlation_matrix(
            df, entropy,
            f'Entropy Feature Correlations - {args.condition}',
            output_dir / f'correlation_entropy_{args.condition}.png'
        )
        
        # Print summary
        print("\nEntropy feature correlations:")
        print(corr_entropy.to_string())
        
        # Pairplot for entropy
        plot_pairwise_scatter(df, entropy, args.condition, output_dir)
    
    # 2. PSD relative power correlation
    if psd_rel:
        print("\n" + "="*60)
        print("PSD RELATIVE POWER")
        print("="*60)
        corr_psd_rel = plot_correlation_matrix(
            df, psd_rel,
            f'PSD Relative Power Correlations - {args.condition}',
            output_dir / f'correlation_psd_rel_{args.condition}.png'
        )
    
    # 3. All features together (may be large)
    all_features = psd_rel + entropy
    if len(all_features) > 2:
        print("\n" + "="*60)
        print("ALL FEATURES COMBINED")
        print("="*60)
        corr_all = plot_correlation_matrix(
            df, all_features,
            f'All Feature Correlations - {args.condition}',
            output_dir / f'correlation_all_{args.condition}.png'
        )
    
    # 4. Check for problematic features
    if entropy:
        print("\n" + "="*60)
        print("DIAGNOSTIC CHECKS")
        print("="*60)
        
        for feature in entropy:
            feature_data = df[feature].dropna()
            print(f"\n{feature}:")
            print(f"  N observations: {len(feature_data)}")
            print(f"  Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
            print(f"  Mean: {feature_data.mean():.4f}")
            print(f"  Std: {feature_data.std():.4f}")
            print(f"  N zeros: {(feature_data == 0).sum()}")
            print(f"  N NaN: {df[feature].isna().sum()}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
