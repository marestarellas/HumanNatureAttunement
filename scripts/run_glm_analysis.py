"""
Script to run mixed-effects GLM analysis on EEG features.

This script:
1. Loads feature CSVs for multiple subjects
2. Runs linear mixed models (LMM) comparing two conditions
3. One model per electrode with random intercept for participant
4. Plots topomaps showing fixed effects for each feature
5. Supports both PSD bands and entropy features

Usage:
    python run_glm_analysis.py --cond1 VIZ --cond2 MULTI --features delta theta alpha
    python run_glm_analysis.py --cond1 AUD --cond2 VIZ --features alpha lzc perm_entropy
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import mne


# Available features
PSD_FEATURES = ['delta_abs', 'theta_abs', 'alpha_abs', 'low_beta_abs', 'high_beta_abs', 'gamma1_abs',
                'delta_rel', 'theta_rel', 'alpha_rel', 'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
ENTROPY_FEATURES = ['lzc', 'perm_entropy', 'spectral_entropy', 'svd_entropy', 'sample_entropy']
ALL_FEATURES = PSD_FEATURES + ENTROPY_FEATURES


def load_subject_data(subject_id, condition, data_dir):
    """
    Load feature data for one subject and condition.
    
    Parameters
    ----------
    subject_id : int
        Subject number (e.g., 2, 3, 4)
    condition : str
        Condition name (e.g., 'VIZ', 'AUD', 'MULTI')
    data_dir : Path
        Path to data directory
    
    Returns
    -------
    df : pd.DataFrame or None
        Feature dataframe with subject and condition info
    """
    subject_folder = f'sub-{subject_id:02d}'
    feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
    
    if not feature_file.exists():
        print(f"  Warning: {feature_file} not found")
        return None
    
    df = pd.read_csv(feature_file)
    
    # Add subject ID as integer for mixed models
    df['subject_id'] = subject_id
    
    return df


def load_all_data(subjects, conditions, data_dir):
    """
    Load and combine data from multiple subjects and conditions.
    
    Parameters
    ----------
    subjects : list
        List of subject IDs
    conditions : list
        List of condition names (should be 2 for comparison)
    data_dir : Path
        Path to data directory
    
    Returns
    -------
    df_combined : pd.DataFrame
        Combined dataframe with all data
    """
    print(f"\nLoading data for subjects {subjects}")
    print(f"Conditions: {conditions}")
    
    dfs = []
    
    for subject_id in subjects:
        print(f"\n  Subject {subject_id:02d}:")
        for condition in conditions:
            df = load_subject_data(subject_id, condition, data_dir)
            if df is not None:
                print(f"    {condition}: {len(df)} observations")
                dfs.append(df)
    
    if not dfs:
        raise ValueError("No data loaded!")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal combined data: {len(df_combined)} observations")
    print(f"Subjects: {sorted(df_combined['subject_id'].unique())}")
    print(f"Conditions: {sorted(df_combined['condition'].unique())}")
    print(f"Channels: {len(df_combined['channel'].unique())}")
    
    return df_combined


def run_mixed_model(df, channel, feature, formula_str):
    """
    Run mixed-effects model for one channel and one feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for this channel
    channel : str
        Channel name
    feature : str
        Feature name (column in df)
    formula_str : str
        Model formula
    
    Returns
    -------
    result : dict or None
        Dictionary with model results
    """
    try:
        # Fit mixed model
        model = smf.mixedlm(formula_str, df, groups=df["subject_id"])
        result = model.fit(method='lbfgs', maxiter=200, reml=False)
        
        # Extract fixed effects - use actual parameter names from model
        fixed_effects = result.fe_params
        pvalues = result.pvalues
        
        # Get the condition effect parameter (should be 'condition_encoded')
        condition_param = 'condition_encoded'
        
        return {
            'channel': channel,
            'feature': feature,
            'intercept': fixed_effects.get('Intercept', np.nan),
            'condition_effect': fixed_effects.get(condition_param, np.nan),
            'intercept_pval': pvalues.get('Intercept', np.nan),
            'condition_pval': pvalues.get(condition_param, np.nan),
            'converged': result.converged,
            'aic': result.aic if hasattr(result, 'aic') else np.nan
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def run_models_all_channels(df_combined, feature, cond1, cond2):
    """
    Run mixed-effects models for all channels for one feature.
    
    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined data from all subjects
    feature : str
        Feature to model
    cond1, cond2 : str
        Condition names to compare
    
    Returns
    -------
    results_df : pd.DataFrame
        Results for all channels
    """
    print(f"\n{'='*60}")
    print(f"Running models for feature: {feature}")
    print(f"Comparing: {cond1} vs {cond2}")
    print(f"Model: {cond2} - {cond1} (positive effect = {cond2} > {cond1})")
    print(f"{'='*60}")
    
    # Filter to only the two conditions
    df_two_conds = df_combined[df_combined['condition'].isin([cond1, cond2])].copy()
    
    # Encode condition as 0/1 for model (numeric, not categorical)
    df_two_conds['condition_encoded'] = (df_two_conds['condition'] == cond2).astype(float)
    
    # Formula: feature ~ condition_encoded (as numeric predictor)
    # Random intercept is handled by groups parameter in mixedlm
    formula = f"{feature} ~ condition_encoded"
    
    channels = sorted(df_two_conds['channel'].unique())
    results = []
    
    for i, channel in enumerate(channels, 1):
        # Get data for this channel
        df_channel = df_two_conds[df_two_conds['channel'] == channel].copy()
        
        # Remove NaN values
        df_channel = df_channel.dropna(subset=[feature])
        
        if len(df_channel) < 10:
            print(f"  [{i}/{len(channels)}] {channel}... insufficient data")
            continue
        
        # Check for variance in condition
        if df_channel['condition_encoded'].std() == 0:
            print(f"  [{i}/{len(channels)}] {channel}... no condition variance")
            continue
        
        # Run model
        result = run_mixed_model(df_channel, channel, feature, formula)
        
        if result is not None:
            pval = result['condition_pval']
            if np.isnan(pval):
                print(f"  [{i}/{len(channels)}] {channel}... ⚠ p-value is NaN")
            elif result['converged']:
                print(f"  [{i}/{len(channels)}] {channel}... ✓ (p={pval:.4f})")
            else:
                print(f"  [{i}/{len(channels)}] {channel}... ⚠ did not converge (p={pval:.4f})")
            results.append(result)
        else:
            print(f"  [{i}/{len(channels)}] {channel}... ✗ failed")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Apply FDR correction across all channels
        _, pvals_corrected, _, _ = multipletests(
            results_df['condition_pval'].values, 
            alpha=0.05, 
            method='fdr_bh'
        )
        results_df['condition_pval_fdr'] = pvals_corrected
        
        n_sig_uncorrected = (results_df['condition_pval'] < 0.05).sum()
        n_sig_fdr = (results_df['condition_pval_fdr'] < 0.05).sum()
        print(f"\nCompleted: {len(results_df)} channels")
        print(f"  Significant (uncorrected p < 0.05): {n_sig_uncorrected}")
        print(f"  Significant (FDR corrected p < 0.05): {n_sig_fdr}")
    
    return results_df


def plot_topomap_results(results_df, feature, cond1, cond2, output_dir, alpha=0.05, use_fdr=True):
    """
    Plot topomap showing condition effects from mixed models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_models_all_channels
    feature : str
        Feature name
    cond1, cond2 : str
        Condition names
    output_dir : Path
        Directory to save plots
    alpha : float
        Significance threshold
    use_fdr : bool
        If True, use FDR-corrected p-values (default: True)
    """
    print(f"\nPlotting topomap for {feature}...")
    
    if len(results_df) == 0:
        print("  No results to plot")
        return
    
    # Standard 32 channel names for MNE
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    n_channels = len(results_df)
    mne_ch_names = standard_32_channels[:n_channels]
    
    # Create MNE Info object
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    # Extract condition effects and p-values
    effects = results_df['condition_effect'].values
    
    # Use FDR-corrected or uncorrected p-values
    if use_fdr and 'condition_pval_fdr' in results_df.columns:
        pvals = results_df['condition_pval_fdr'].values
        pval_label = 'FDR-corrected'
    else:
        pvals = results_df['condition_pval'].values
        pval_label = 'uncorrected'
    
    # Create significance mask
    sig_mask = pvals < alpha
    
    # Mask non-significant effects
    masked_effects = effects.copy()
    masked_effects[~sig_mask] = 0
    
    # Determine colormap
    if len(masked_effects[sig_mask]) > 0:
        sig_effects = masked_effects[sig_mask]
        has_positive = np.any(sig_effects > 0)
        has_negative = np.any(sig_effects < 0)
        
        if has_positive and has_negative:
            cmap = 'RdBu_r'
            max_abs = np.max(np.abs(masked_effects))
            vlim = (-max_abs, max_abs) if max_abs > 0 else (-1e-6, 1e-6)
        elif has_positive:
            cmap = 'Reds'
            vlim = (0, np.max(masked_effects) if np.max(masked_effects) > 0 else 1e-6)
        else:
            cmap = 'Blues_r'
            vlim = (np.min(masked_effects) if np.min(masked_effects) < 0 else -1e-6, 0)
    else:
        cmap = 'RdBu_r'
        vlim = (-1e-6, 1e-6)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im, _ = mne.viz.plot_topomap(
        masked_effects,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=vlim,
        contours=0,
        sensors=False
    )
    
    # Add significance markers
    if np.any(sig_mask):
        from mne.viz.topomap import _get_pos_outlines
        pos_xy, outlines = _get_pos_outlines(info, None, sphere=None)
        sig_positions = pos_xy[sig_mask]
        ax.scatter(sig_positions[:, 0], sig_positions[:, 1],
                  s=80, c='white', marker='o', edgecolors='black',
                  linewidths=1, zorder=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Condition Effect', rotation=270, labelpad=20)
    
    # Title
    n_sig = sig_mask.sum()
    # Effect represents cond2 - cond1, so positive = cond2 > cond1
    ax.set_title(f'{feature}\n{cond1} vs {cond2}\n'
                f'Effect: {cond2} - {cond1} (positive = {cond2} higher)\n'
                f'{n_sig}/{n_channels} significant channels ({pval_label} p < {alpha})',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'topomap_{feature}_{cond1}_vs_{cond2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run mixed-effects GLM analysis on EEG features')
    parser.add_argument('--cond1', type=str, required=True, help='First condition (e.g., VIZ)')
    parser.add_argument('--cond2', type=str, required=True, help='Second condition (e.g., MULTI)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs to include (default: 2 3 4 5 6)')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Features to analyze (default: all relative PSD bands)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance threshold (default: 0.05)')
    parser.add_argument('--no-fdr', action='store_true',
                       help='Disable FDR correction (use uncorrected p-values)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/glm_analysis_COND1_vs_COND2)')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    # Default output directory
    if args.output_dir is None:
        output_dir = project_dir / 'results' / f'glm_analysis_{args.cond1}_vs_{args.cond2}'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default features: all relative PSD bands
    if args.features is None:
        features = ['delta_rel', 'theta_rel', 'alpha_rel', 'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
    else:
        features = args.features
    
    # Validate features
    invalid_features = [f for f in features if f not in ALL_FEATURES]
    if invalid_features:
        print(f"ERROR: Invalid features: {invalid_features}")
        print(f"Available features:")
        print(f"  PSD: {PSD_FEATURES}")
        print(f"  Entropy: {ENTROPY_FEATURES}")
        sys.exit(1)
    
    print("="*60)
    print("MIXED-EFFECTS GLM ANALYSIS")
    print("="*60)
    print(f"Conditions: {args.cond1} vs {args.cond2}")
    print(f"Subjects: {args.subjects}")
    print(f"Features: {features}")
    print(f"Alpha: {args.alpha}")
    print(f"FDR correction: {'No' if args.no_fdr else 'Yes (Benjamini-Hochberg)'}")
    print(f"Random effects: Random intercept per subject")
    print(f"Output: {output_dir}")
    
    # Load data
    df_combined = load_all_data(args.subjects, [args.cond1, args.cond2], data_dir)
    
    # Run models for each feature
    all_results = {}
    
    for feature in features:
        if feature not in df_combined.columns:
            print(f"\nWARNING: Feature '{feature}' not found in data, skipping")
            continue
        
        results_df = run_models_all_channels(df_combined, feature, args.cond1, args.cond2)
        
        if len(results_df) > 0:
            all_results[feature] = results_df
            
            # Save results to CSV
            csv_file = output_dir / f'results_{feature}_{args.cond1}_vs_{args.cond2}.csv'
            results_df.to_csv(csv_file, index=False)
            print(f"  Saved results: {csv_file}")
            
            # Plot topomap
            plot_topomap_results(results_df, feature, args.cond1, args.cond2, 
                               output_dir, args.alpha, use_fdr=not args.no_fdr)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Features analyzed: {len(all_results)}")
    
    pval_col = 'condition_pval' if args.no_fdr else 'condition_pval_fdr'
    pval_type = 'uncorrected' if args.no_fdr else 'FDR-corrected'
    
    for feature, results_df in all_results.items():
        n_sig = (results_df[pval_col] < args.alpha).sum()
        n_total = len(results_df)
        print(f"  {feature}: {n_sig}/{n_total} significant channels ({pval_type})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
