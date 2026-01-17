"""
Statistical analysis of audio-EEG correlation results.

This script analyzes the correlation results from compute_audio_eeg_correlation.py,
comparing conditions using appropriate statistical models and generating visualizations.

Supports both direct envelope correlation and time-lagged cross-correlation metrics.

Usage:
    python run_correlation_stats.py --metric correlation_direct --condition1 VIZ --condition2 AUD
    python run_correlation_stats.py --metric correlation_max_lagged --condition1 VIZ --condition2 AUD
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


def fisher_z_transform(r):
    """
    Apply Fisher's z-transformation to correlation coefficients.
    z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
    
    This transformation makes correlations approximately normally distributed,
    which is required for valid parametric statistical tests.
    """
    # Clip to avoid numerical issues at boundaries
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r_clipped)


def load_correlation_results(results_dir, condition1, condition2):
    """Load correlation results for two conditions."""
    results_file = results_dir / 'audio_eeg_correlation_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Filter to specified conditions
    df = df[df['condition'].isin([condition1, condition2])].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for conditions {condition1}, {condition2}")
    
    # Apply Fisher's z-transformation to correlation coefficients
    print("\nApplying Fisher's z-transformation to correlation coefficients...")
    if 'correlation_direct' in df.columns:
        df['correlation_direct_z'] = fisher_z_transform(df['correlation_direct'])
    if 'correlation_max_lagged' in df.columns:
        df['correlation_max_lagged_z'] = fisher_z_transform(df['correlation_max_lagged'])
    
    print(f"Loaded {len(df)} records")
    print(f"  Subjects: {sorted(df['subject_id'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Bands: {sorted(df['band'].unique())}")
    print(f"  Channels: {len(df['channel'].unique())}")
    
    return df


def run_paired_ttest(df, metric, channel, band):
    """
    Run paired t-test for two conditions (when n_subjects < 3).
    
    Parameters:
    -----------
    df : DataFrame
        Data for this channel and band
    metric : str
        Metric to test (correlation_direct or correlation_max_lagged)
    channel : str
        Channel name
    band : str
        Band name
    
    Returns:
    --------
    result : dict
        Test results
    """
    conditions = df['condition'].unique()
    if len(conditions) != 2:
        return None
    
    cond1_values = df[df['condition'] == conditions[0]][metric].values
    cond2_values = df[df['condition'] == conditions[1]][metric].values
    
    # Check variance
    if np.std(cond1_values) < 1e-10 or np.std(cond2_values) < 1e-10:
        return None
    
    if len(cond1_values) != len(cond2_values):
        return None
    
    t_stat, p_value = stats.ttest_rel(cond1_values, cond2_values)
    
    return {
        'channel': channel,
        'band': band,
        'test': 't-test',
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': np.mean(cond1_values) - np.mean(cond2_values),
        'n_subjects': len(cond1_values),
        'model_type': 'ttest'
    }


def run_mixed_model(df, metric, channel, band):
    """
    Run linear mixed-effects model with subject random intercept.
    Falls back to OLS if singular matrix error occurs.
    
    Parameters:
    -----------
    df : DataFrame
        Data for this channel and band
    metric : str
        Metric to test
    channel : str
        Channel name
    band : str
        Band name
    
    Returns:
    --------
    result : dict
        Model results
    """
    # Prepare data
    df = df.copy()
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Check variance
    if df[metric].std() < 1e-10:
        return None
    
    # Formula
    formula = f"{metric} ~ C(condition)"
    
    try:
        # Try mixed-effects model
        model = smf.mixedlm(formula, df, groups=df["subject_id"])
        result = model.fit(method='powell', maxiter=1000, reml=True)
        
        # Extract results
        coef = result.params.get('C(condition)[T.VIZ]', 
                                 result.params.get('C(condition)[T.AUD]',
                                 result.params.get('C(condition)[T.MULTI]', np.nan)))
        pvalue = result.pvalues.get('C(condition)[T.VIZ]',
                                    result.pvalues.get('C(condition)[T.AUD]',
                                    result.pvalues.get('C(condition)[T.MULTI]', np.nan)))
        
        return {
            'channel': channel,
            'band': band,
            'test': 'mixed-lm',
            'coefficient': coef,
            'p_value': pvalue,
            'n_obs': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'model_type': 'mixed'
        }
        
    except (np.linalg.LinAlgError, Exception) as e:
        # Fall back to OLS
        if 'Singular matrix' in str(e) or 'LinAlgError' in str(type(e).__name__):
            try:
                ols_model = smf.ols(formula, data=df)
                ols_result = ols_model.fit()
                
                coef = ols_result.params.get('C(condition)[T.VIZ]',
                                            ols_result.params.get('C(condition)[T.AUD]',
                                            ols_result.params.get('C(condition)[T.MULTI]', np.nan)))
                pvalue = ols_result.pvalues.get('C(condition)[T.VIZ]',
                                               ols_result.pvalues.get('C(condition)[T.AUD]',
                                               ols_result.pvalues.get('C(condition)[T.MULTI]', np.nan)))
                
                return {
                    'channel': channel,
                    'band': band,
                    'test': 'ols',
                    'coefficient': coef,
                    'p_value': pvalue,
                    'n_obs': len(df),
                    'n_subjects': df['subject_id'].nunique(),
                    'model_type': 'ols'
                }
            except Exception as e2:
                print(f"    Warning: OLS also failed for {channel}, {band}: {e2}")
                return None
        else:
            print(f"    Warning: Model failed for {channel}, {band}: {e}")
            return None


def run_models_all_channels(df, metric, condition1, condition2):
    """
    Run statistical models for all channels and bands.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    metric : str
        Metric to analyze (will automatically use Fisher z-transformed version)
    condition1, condition2 : str
        Conditions to compare
    
    Returns:
    --------
    results_df : DataFrame
        Statistical results
    """
    # Use Fisher z-transformed metric for statistical testing
    if metric == 'correlation_direct' and 'correlation_direct_z' in df.columns:
        analysis_metric = 'correlation_direct_z'
        print(f"\nUsing Fisher z-transformed metric: {analysis_metric}")
    elif metric == 'correlation_max_lagged' and 'correlation_max_lagged_z' in df.columns:
        analysis_metric = 'correlation_max_lagged_z'
        print(f"\nUsing Fisher z-transformed metric: {analysis_metric}")
    else:
        analysis_metric = metric
        print(f"\nWarning: Using raw metric {metric} without Fisher transformation")
    
    # Check number of subjects
    n_subjects = df['subject_id'].nunique()
    print(f"Number of subjects detected: {n_subjects}")
    
    # Choose model based on sample size
    if n_subjects < 3:
        print("Using paired t-test (n < 3)")
        model_func = run_paired_ttest
    else:
        print("Using mixed-effects model (with OLS fallback)")
        model_func = run_mixed_model
    
    results_list = []
    
    # Get all bands and channels
    bands = sorted(df['band'].unique())
    channels = sorted(df['channel'].unique())
    
    print(f"\nProcessing {len(bands)} bands x {len(channels)} channels...")
    
    for band in bands:
        print(f"\n  {band.upper()} band:")
        band_df = df[df['band'] == band]
        
        succeeded = 0
        failed = 0
        
        for channel in channels:
            channel_df = band_df[band_df['channel'] == channel]
            
            if len(channel_df) < 4:  # Need at least 4 observations
                failed += 1
                continue
            
            try:
                result = model_func(channel_df, analysis_metric, channel, band)
                if result is not None:
                    results_list.append(result)
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    Error: {channel}: {e}")
                failed += 1
        
        print(f"    Succeeded: {succeeded}/{len(channels)}, Failed: {failed}")
    
    if len(results_list) == 0:
        raise ValueError("No valid results obtained")
    
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction within each band
    print("\nApplying FDR correction within each band...")
    results_df['p_fdr'] = np.nan
    
    for band in results_df['band'].unique():
        band_mask = results_df['band'] == band
        p_values = results_df.loc[band_mask, 'p_value'].values
        
        # Benjamini-Hochberg FDR
        n_tests = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate critical values
        fdr_threshold = 0.05
        critical_values = (np.arange(n_tests) + 1) / n_tests * fdr_threshold
        
        # Find significant tests
        p_fdr = np.ones(n_tests)
        for i in range(n_tests):
            p_fdr[sorted_idx[i]] = sorted_p[i] * n_tests / (i + 1)
        
        # Ensure monotonicity
        p_fdr = np.minimum.accumulate(p_fdr[::-1])[::-1]
        p_fdr = np.minimum(p_fdr, 1.0)
        
        results_df.loc[band_mask, 'p_fdr'] = p_fdr
        
        # Count significant
        n_sig = np.sum(p_fdr < 0.05)
        print(f"  {band}: {n_sig}/{n_tests} channels significant (FDR < 0.05)")
    
    return results_df


def plot_topomap_results(results_df, metric, condition1, condition2, output_dir):
    """
    Plot topographic maps of correlation effects for each band in a single figure.
    Uses MNE standard 10-20 montage for proper scalp representation.
    
    Parameters:
    -----------
    results_df : DataFrame
        Statistical results
    metric : str
        Metric name
    condition1, condition2 : str
        Conditions compared
    output_dir : Path
        Output directory
    """
    import mne
    
    # Define band order and frequency ranges
    band_order = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'gamma1']
    freq_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma1': (30, 50)
    }
    
    # Get bands in correct order
    available_bands = [b for b in band_order if b in results_df['band'].unique()]
    n_bands = len(available_bands)
    
    # Standard 32 channel names for MNE 10-20 system
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    # Create figure with subplots for all bands
    ncols = 3
    nrows = (n_bands + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Collect all values to determine global colormap range
    all_values = []
    for band in available_bands:
        band_df = results_df[results_df['band'] == band].copy()
        for _, row in band_df.iterrows():
            if 'coefficient' in row:
                all_values.append(row['coefficient'])
            elif 'mean_diff' in row:
                all_values.append(row['mean_diff'])
    
    global_vmax = np.max(np.abs(all_values)) if len(all_values) > 0 else 1e-6
    
    for idx, band in enumerate(available_bands):
        ax = axes[idx]
        band_df = results_df[results_df['band'] == band].copy()
        
        # Get frequency range from predefined dict
        lowcut, highcut = freq_ranges.get(band, (0, 0))
        
        # Get number of channels
        n_channels = len(band_df)
        mne_ch_names = standard_32_channels[:n_channels]
        
        # Create MNE Info object
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        # Prepare data
        values = []
        sig_mask = []
        
        for _, row in band_df.iterrows():
            if 'coefficient' in row:
                values.append(row['coefficient'])
            elif 'mean_diff' in row:
                values.append(row['mean_diff'])
            else:
                values.append(0)
            sig_mask.append(row.get('p_fdr', 1) < 0.05)
        
        if len(values) == 0:
            ax.axis('off')
            continue
        
        values = np.array(values)
        sig_mask = np.array(sig_mask)
        
        # Plot topomap using MNE
        im, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(-global_vmax, global_vmax),
            contours=0,
            sensors=False
        )
        
        # Mark significant electrodes
        if np.any(sig_mask):
            from mne.viz.topomap import _get_pos_outlines
            pos_xy, outlines = _get_pos_outlines(info, None, sphere=None)
            sig_positions = pos_xy[sig_mask]
            ax.scatter(sig_positions[:, 0], sig_positions[:, 1],
                      s=60, c='white', marker='o', edgecolors='black',
                      linewidths=2, zorder=10)
        
        # Title with band name
        n_sig = np.sum(sig_mask)
        ax.set_title(f'{band.upper()} ({lowcut}-{highcut} Hz)\n{n_sig}/{len(sig_mask)} sig.', 
                     fontsize=12, fontweight='bold', pad=10)
    
    # Hide extra subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    # Add single colorbar for the whole figure
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Effect Size (Fisher z)', fontsize=11, rotation=270, labelpad=20)
    
    # Add directional labels at colorbar extremes
    # Red (positive) = condition2 > condition1
    # Blue (negative) = condition1 > condition2
    cbar.ax.text(1.5, 1.02, condition2, transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    cbar.ax.text(1.5, -0.02, condition1, transform=cbar.ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold', color='darkblue')
    
    # Overall title
    plt.suptitle(f'Audio-EEG Correlation Topomaps: {condition1} vs {condition2}\n{metric}', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    output_file = output_dir / f'topomaps_all_bands_{metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


def plot_all_bands_summary(results_df, metric, condition1, condition2, output_dir):
    """
    Create summary plot with all frequency bands.
    
    Parameters:
    -----------
    results_df : DataFrame
        Statistical results
    metric : str
        Metric name
    condition1, condition2 : str
        Conditions compared
    output_dir : Path
        Output directory
    """
    bands = sorted(results_df['band'].unique())
    n_bands = len(bands)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = results_df[results_df['band'] == band].copy()
        
        # Get values
        if 'coefficient' in band_df.columns:
            values = band_df['coefficient'].values
        elif 'mean_diff' in band_df.columns:
            values = band_df['mean_diff'].values
        else:
            continue
        
        p_fdr = band_df['p_fdr'].values
        channels = band_df['channel'].values
        
        # Sort by absolute value
        sort_idx = np.argsort(np.abs(values))[::-1]
        
        # Plot top 15 channels
        n_show = min(15, len(values))
        x_pos = np.arange(n_show)
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_fdr[sort_idx[:n_show]]]
        
        ax.barh(x_pos, values[sort_idx[:n_show]], color=colors, alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels([ch.replace('EEG-', '') for ch in channels[sort_idx[:n_show]]], fontsize=8)
        ax.set_xlabel(f'{metric}', fontsize=10)
        ax.set_title(f'{band.upper()}', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add significance info
        n_sig = np.sum(p_fdr < 0.05)
        ax.text(0.95, 0.95, f'{n_sig}/{len(p_fdr)} sig.',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplots
    for idx in range(n_bands, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{metric}\n{condition1} vs {condition2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'summary_{metric}_all_bands.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved summary: {output_file.name}")


def plot_lag_distribution(results_df, output_dir):
    """
    Plot distribution of optimal lags across channels and bands.
    Only applicable when analyzing correlation_max_lagged metric.
    """
    if 'optimal_lag_ms' not in results_df.columns:
        return
    
    df = results_df.copy()
    
    # Load original data to get lag values
    results_file = output_dir / 'audio_eeg_correlation_results.csv'
    if results_file.exists():
        full_df = pd.read_csv(results_file)
        
        # Merge to get lag values
        df = df.merge(
            full_df[['subject_id', 'condition', 'channel', 'band', 'optimal_lag_ms']].drop_duplicates(),
            on=['channel', 'band'],
            how='left',
            suffixes=('', '_orig')
        )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    bands = sorted(df['band'].unique())
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = df[df['band'] == band]
        
        if 'optimal_lag_ms_orig' in band_df.columns:
            lags = band_df['optimal_lag_ms_orig'].dropna().values
        elif 'optimal_lag_ms' in band_df.columns:
            lags = band_df['optimal_lag_ms'].dropna().values
        else:
            continue
        
        if len(lags) == 0:
            continue
        
        # Histogram
        ax.hist(lags, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero lag')
        ax.axvline(np.median(lags), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(lags):.1f}ms')
        ax.set_xlabel('Optimal Lag (ms)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{band.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(bands), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Distribution of Optimal Time Lags', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'optimal_lags_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved lag distribution: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of audio-EEG correlation')
    parser.add_argument('--metric', type=str, required=True,
                       choices=['correlation_direct', 'correlation_max_lagged'],
                       help='Metric to analyze')
    parser.add_argument('--condition1', type=str, required=True,
                       help='First condition')
    parser.add_argument('--condition2', type=str, required=True,
                       help='Second condition')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: results/audio_eeg_correlation)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    if args.results_dir is None:
        results_dir = project_dir / 'results' / 'audio_eeg_correlation'
    else:
        results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Create output directory
    output_dir = results_dir / 'statistics'
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("AUDIO-EEG CORRELATION STATISTICS")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Comparison: {args.condition1} vs {args.condition2}")
    print(f"Results: {results_dir}")
    
    # Load data
    df = load_correlation_results(results_dir, args.condition1, args.condition2)
    
    # Add frequency band info if missing
    if 'lowcut' not in df.columns:
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'low_beta': (13, 20),
            'high_beta': (20, 30),
            'gamma1': (30, 50)
        }
        df['lowcut'] = df['band'].map(lambda b: freq_bands.get(b, (0, 0))[0])
        df['highcut'] = df['band'].map(lambda b: freq_bands.get(b, (0, 0))[1])
    
    # Run statistical models
    results_df = run_models_all_channels(df, args.metric, args.condition1, args.condition2)
    
    # Save results
    stats_file = output_dir / f'stats_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    results_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics: {stats_file}")
    
    # Add note about Fisher transformation
    if args.metric in ['correlation_direct', 'correlation_max_lagged']:
        print("\nNote: Statistics computed on Fisher z-transformed correlations")
        print("      (coefficients are in z-space, not raw r-values)")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_topomap_results(results_df, args.metric, args.condition1, args.condition2, output_dir)
    plot_all_bands_summary(results_df, args.metric, args.condition1, args.condition2, output_dir)
    
    # Plot lag distribution if using lagged metric
    if args.metric == 'correlation_max_lagged':
        plot_lag_distribution(results_df, results_dir)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
