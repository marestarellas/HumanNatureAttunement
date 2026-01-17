"""
Script to run statistical analysis on audio-EEG coherence results.

This script:
1. Loads coherence results CSV
2. Runs linear mixed models (LMM) comparing conditions
3. One model per channel per frequency band with random intercept for participant
4. Plots topomaps showing coherence differences between conditions
5. Applies FDR correction across channels

Usage:
    python run_coherence_stats.py --cond1 VIZ --cond2 MULTI
    python run_coherence_stats.py --cond1 AUD --cond2 VIZ --bands delta theta alpha
    python run_coherence_stats.py --cond1 VIZ --cond2 MULTI --metric peak_coherence
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

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats as sp_stats
import mne


def load_coherence_data(coherence_file, subjects=None, conditions=None):
    """
    Load coherence results CSV.
    
    Parameters
    ----------
    coherence_file : Path
        Path to coherence results CSV
    subjects : list, optional
        Subject IDs to include (default: all)
    conditions : list, optional
        Conditions to include (default: all)
    
    Returns
    -------
    df : pd.DataFrame
        Coherence data
    """
    print(f"\nLoading coherence data from: {coherence_file}")
    
    if not coherence_file.exists():
        raise FileNotFoundError(f"Coherence file not found: {coherence_file}")
    
    df = pd.read_csv(coherence_file)
    
    # Filter subjects
    if subjects is not None:
        df = df[df['subject_id'].isin(subjects)]
    
    # Filter conditions
    if conditions is not None:
        df = df[df['condition'].isin(conditions)]
    
    print(f"Loaded {len(df)} observations")
    print(f"Subjects: {sorted(df['subject_id'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Bands: {sorted(df['band'].unique())}")
    print(f"Channels: {len(df['channel'].unique())}")
    
    return df


def run_paired_ttest(df, channel, band, metric, cond1, cond2):
    """
    Run paired t-test for one channel (when only 2 subjects available).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for this channel and band (must have 2 observations per subject)
    channel : str
        Channel name
    band : str
        Frequency band name
    metric : str
        Metric to test
    cond1, cond2 : str
        Condition names
    
    Returns
    -------
    result : dict or None
        Dictionary with test results
    """
    try:
        # Pivot data: one row per subject, columns for each condition
        pivot = df.pivot(index='subject_id', columns='condition', values=metric)
        
        if cond1 not in pivot.columns or cond2 not in pivot.columns:
            return None
        
        # Remove subjects with missing data
        pivot_clean = pivot[[cond1, cond2]].dropna()
        
        if len(pivot_clean) < 2:
            return None
        
        # Paired t-test
        t_stat, p_val = sp_stats.ttest_rel(pivot_clean[cond2], pivot_clean[cond1])
        
        # Effect size (mean difference)
        effect = pivot_clean[cond2].mean() - pivot_clean[cond1].mean()
        intercept = pivot_clean[cond1].mean()
        
        return {
            'channel': channel,
            'band': band,
            'intercept': intercept,
            'condition_effect': effect,
            'intercept_pval': np.nan,  # Not computed for t-test
            'condition_pval': p_val,
            'converged': True,
            'aic': np.nan,
            't_statistic': t_stat
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def run_mixed_model(df, channel, band, metric, formula_str):
    """
    Run mixed-effects model for one channel and band.
    Falls back to OLS if mixed model fails.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for this channel and band
    channel : str
        Channel name
    band : str
        Frequency band name
    metric : str
        Metric to model (e.g., 'mean_coherence')
    formula_str : str
        Model formula
    
    Returns
    -------
    result : dict or None
        Dictionary with model results
    """
    try:
        # Try mixed model first
        model = smf.mixedlm(formula_str, df, groups=df["subject_id"])
        result = model.fit(method='lbfgs', maxiter=200, reml=False)
        
        # Extract fixed effects
        fixed_effects = result.fe_params
        pvalues = result.pvalues
        
        # Get the condition effect parameter
        condition_param = 'condition_encoded'
        
        return {
            'channel': channel,
            'band': band,
            'intercept': fixed_effects.get('Intercept', np.nan),
            'condition_effect': fixed_effects.get(condition_param, np.nan),
            'intercept_pval': pvalues.get('Intercept', np.nan),
            'condition_pval': pvalues.get(condition_param, np.nan),
            'converged': result.converged,
            'aic': result.aic if hasattr(result, 'aic') else np.nan,
            'model_type': 'mixed'
        }
    except (np.linalg.LinAlgError, Exception) as e:
        # Fall back to OLS if mixed model fails (singular matrix, etc.)
        if 'Singular matrix' in str(e) or 'LinAlgError' in str(type(e).__name__):
            try:
                # Use OLS instead
                ols_model = smf.ols(formula_str, data=df)
                ols_result = ols_model.fit()
                
                fixed_effects = ols_result.params
                pvalues = ols_result.pvalues
                condition_param = 'condition_encoded'
                
                return {
                    'channel': channel,
                    'band': band,
                    'intercept': fixed_effects.get('Intercept', np.nan),
                    'condition_effect': fixed_effects.get(condition_param, np.nan),
                    'intercept_pval': pvalues.get('Intercept', np.nan),
                    'condition_pval': pvalues.get(condition_param, np.nan),
                    'converged': True,
                    'aic': ols_result.aic if hasattr(ols_result, 'aic') else np.nan,
                    'model_type': 'ols'
                }
            except Exception as ols_error:
                print(f"Error (OLS fallback): {str(ols_error)}")
                return None
        else:
            print(f"Error: {str(e)}")
            return None


def run_models_all_channels(df_combined, band, metric, cond1, cond2):
    """
    Run statistical models for all channels for one frequency band.
    Uses paired t-test for 2 subjects, mixed models for 3+.
    
    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined coherence data
    band : str
        Frequency band to analyze
    metric : str
        Coherence metric to model
    cond1, cond2 : str
        Condition names to compare
    
    Returns
    -------
    results_df : pd.DataFrame
        Results for all channels
    """
    print(f"\n{'='*60}")
    print(f"Running models for band: {band}")
    print(f"Metric: {metric}")
    print(f"Comparing: {cond1} vs {cond2}")
    print(f"Model: {cond2} - {cond1} (positive effect = {cond2} > {cond1})")
    print(f"{'='*60}")
    
    # Filter to this band and two conditions
    df_band = df_combined[
        (df_combined['band'] == band) &
        (df_combined['condition'].isin([cond1, cond2]))
    ].copy()
    
    if len(df_band) == 0:
        print(f"  No data for band {band}")
        return pd.DataFrame()
    
    # Check number of subjects
    n_subjects = df_band['subject_id'].nunique()
    print(f"  Number of subjects: {n_subjects}")
    
    # Choose appropriate model based on number of subjects
    if n_subjects < 3:
        print(f"  Using paired t-test (< 3 subjects)")
        use_ttest = True
    else:
        print(f"  Using mixed-effects model")
        use_ttest = False
        # Encode condition as 0/1 for mixed model
        df_band['condition_encoded'] = (df_band['condition'] == cond2).astype(float)
        formula = f"{metric} ~ condition_encoded"
    
    channels = sorted(df_band['channel'].unique())
    results = []
    
    for i, channel in enumerate(channels, 1):
        # Get data for this channel
        df_channel = df_band[df_band['channel'] == channel].copy()
        
        # Remove NaN values
        df_channel = df_channel.dropna(subset=[metric])
        
        min_obs = 4 if use_ttest else 6
        if len(df_channel) < min_obs:
            print(f"  [{i}/{len(channels)}] {channel}... insufficient data")
            continue
        
        # Check for variance in metric
        if df_channel[metric].std() == 0:
            print(f"  [{i}/{len(channels)}] {channel}... no variance in metric")
            continue
        
        # Run appropriate model
        if use_ttest:
            result = run_paired_ttest(df_channel, channel, band, metric, cond1, cond2)
        else:
            result = run_mixed_model(df_channel, channel, band, metric, formula)
        
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


def plot_topomap_results(results_df, band, metric, cond1, cond2, output_dir, alpha=0.05, use_fdr=True):
    """
    Plot topomap showing coherence differences between conditions.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_models_all_channels
    band : str
        Frequency band name
    metric : str
        Coherence metric
    cond1, cond2 : str
        Condition names
    output_dir : Path
        Directory to save plots
    alpha : float
        Significance threshold
    use_fdr : bool
        If True, use FDR-corrected p-values
    """
    print(f"\nPlotting topomap for {band}...")
    
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
    cbar.set_label('Coherence Difference', rotation=270, labelpad=20)
    
    # Title
    n_sig = sig_mask.sum()
    metric_label = metric.replace('_', ' ').title()
    ax.set_title(f'{band.upper()} - {metric_label}\n{cond1} vs {cond2}\n'
                f'Effect: {cond2} - {cond1} (positive = {cond2} higher)\n'
                f'{n_sig}/{n_channels} significant channels ({pval_label} p < {alpha})',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'topomap_coherence_{band}_{cond1}_vs_{cond2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_all_bands_summary(all_results, metric, cond1, cond2, output_dir, alpha=0.05, use_fdr=True):
    """
    Plot summary figure with all frequency bands.
    
    Parameters
    ----------
    all_results : dict
        Dictionary mapping band name to results DataFrame
    metric : str
        Coherence metric
    cond1, cond2 : str
        Condition names
    output_dir : Path
        Directory to save plot
    alpha : float
        Significance threshold
    use_fdr : bool
        Use FDR correction
    """
    print(f"\nPlotting summary across all bands...")
    
    bands = list(all_results.keys())
    n_bands = len(bands)
    
    if n_bands == 0:
        print("  No results to plot")
        return
    
    # Standard 32 channel names
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    pval_col = 'condition_pval_fdr' if use_fdr else 'condition_pval'
    
    # Find global min/max for consistent colorbar
    all_effects = []
    for results_df in all_results.values():
        if len(results_df) > 0:
            effects = results_df['condition_effect'].values
            pvals = results_df[pval_col].values
            sig_mask = pvals < alpha
            masked = effects.copy()
            masked[~sig_mask] = 0
            all_effects.extend(masked[sig_mask])
    
    if len(all_effects) > 0:
        max_abs = np.max(np.abs(all_effects))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = -1e-6, 1e-6
    
    for idx, band in enumerate(bands):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        results_df = all_results[band]
        
        if len(results_df) == 0:
            ax.axis('off')
            continue
        
        # Get data
        n_channels = len(results_df)
        mne_ch_names = standard_32_channels[:n_channels]
        
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        effects = results_df['condition_effect'].values
        pvals = results_df[pval_col].values
        sig_mask = pvals < alpha
        
        masked_effects = effects.copy()
        masked_effects[~sig_mask] = 0
        
        # Plot
        im, _ = mne.viz.plot_topomap(
            masked_effects,
            info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(vmin, vmax),
            contours=0,
            sensors=False
        )
        
        # Add significance markers
        if np.any(sig_mask):
            from mne.viz.topomap import _get_pos_outlines
            pos_xy, outlines = _get_pos_outlines(info, None, sphere=None)
            sig_positions = pos_xy[sig_mask]
            ax.scatter(sig_positions[:, 0], sig_positions[:, 1],
                      s=60, c='white', marker='o', edgecolors='black',
                      linewidths=1, zorder=10)
        
        n_sig = sig_mask.sum()
        ax.set_title(f'{band.upper()}\n{n_sig}/{n_channels} sig. channels', fontsize=10)
    
    # Remove unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Coherence Difference', rotation=270, labelpad=20)
    
    # Overall title
    metric_label = metric.replace('_', ' ').title()
    pval_label = 'FDR-corrected' if use_fdr else 'uncorrected'
    fig.suptitle(f'Audio-EEG Coherence: {cond1} vs {cond2}\n'
                f'{metric_label} ({pval_label} p < {alpha})',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'summary_coherence_all_bands_{cond1}_vs_{cond2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run statistical analysis on audio-EEG coherence')
    parser.add_argument('--cond1', type=str, required=True, help='First condition (e.g., VIZ)')
    parser.add_argument('--cond2', type=str, required=True, help='Second condition (e.g., MULTI)')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to include (default: all)')
    parser.add_argument('--bands', type=str, nargs='+', default=None,
                       help='Frequency bands to analyze (default: all)')
    parser.add_argument('--metric', type=str, default='mean_coherence',
                       choices=['mean_coherence', 'peak_coherence'],
                       help='Coherence metric to analyze (default: mean_coherence)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance threshold (default: 0.05)')
    parser.add_argument('--no-fdr', action='store_true',
                       help='Disable FDR correction (use uncorrected p-values)')
    parser.add_argument('--coherence-file', type=str, default=None,
                       help='Path to coherence results CSV (default: results/audio_eeg_coherence/audio_eeg_coherence_results.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/coherence_stats_COND1_vs_COND2)')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Default coherence file
    if args.coherence_file is None:
        coherence_file = project_dir / 'results' / 'audio_eeg_coherence' / 'audio_eeg_coherence_results.csv'
    else:
        coherence_file = Path(args.coherence_file)
    
    # Default output directory
    if args.output_dir is None:
        output_dir = project_dir / 'results' / f'coherence_stats_{args.cond1}_vs_{args.cond2}'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("AUDIO-EEG COHERENCE STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Conditions: {args.cond1} vs {args.cond2}")
    print(f"Subjects: {args.subjects if args.subjects else 'all'}")
    print(f"Metric: {args.metric}")
    print(f"Alpha: {args.alpha}")
    print(f"FDR correction: {'No' if args.no_fdr else 'Yes (Benjamini-Hochberg)'}")
    print(f"Random effects: Random intercept per subject")
    print(f"Output: {output_dir}")
    
    # Load coherence data
    df_coherence = load_coherence_data(coherence_file, 
                                       subjects=args.subjects,
                                       conditions=[args.cond1, args.cond2])
    
    # Default bands: all available
    if args.bands is None:
        bands = sorted(df_coherence['band'].unique())
    else:
        bands = args.bands
    
    print(f"\nAnalyzing bands: {bands}")
    
    # Run models for each band
    all_results = {}
    
    for band in bands:
        results_df = run_models_all_channels(df_coherence, band, args.metric, 
                                            args.cond1, args.cond2)
        
        if len(results_df) > 0:
            all_results[band] = results_df
            
            # Save results to CSV
            csv_file = output_dir / f'results_{band}_{args.cond1}_vs_{args.cond2}.csv'
            results_df.to_csv(csv_file, index=False)
            print(f"  Saved results: {csv_file}")
            
            # Plot topomap
            plot_topomap_results(results_df, band, args.metric, 
                               args.cond1, args.cond2, 
                               output_dir, args.alpha, 
                               use_fdr=not args.no_fdr)
    
    # Plot summary across all bands
    if len(all_results) > 0:
        plot_all_bands_summary(all_results, args.metric, 
                             args.cond1, args.cond2, 
                             output_dir, args.alpha,
                             use_fdr=not args.no_fdr)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Bands analyzed: {len(all_results)}")
    
    pval_col = 'condition_pval' if args.no_fdr else 'condition_pval_fdr'
    pval_type = 'uncorrected' if args.no_fdr else 'FDR-corrected'
    
    for band, results_df in all_results.items():
        n_sig = (results_df[pval_col] < args.alpha).sum()
        n_total = len(results_df)
        print(f"  {band}: {n_sig}/{n_total} significant channels ({pval_type})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
