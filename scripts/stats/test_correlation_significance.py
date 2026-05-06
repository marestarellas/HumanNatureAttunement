"""
Test whether correlations are significantly different from zero within each condition.

This tests if audio-EEG coupling exists at all (not just differences between conditions).
Uses one-sample t-tests against zero for Fisher z-transformed correlations.

Usage:
    python test_correlation_significance.py --metric correlation_direct
    python test_correlation_significance.py --metric correlation_max_lagged
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def fisher_z_transform(r):
    """Apply Fisher's z-transformation to correlation coefficients."""
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r_clipped)


def test_correlation_vs_zero(df, metric, condition, channel, band):
    """
    Test if correlation is significantly different from zero using one-sample t-test.
    
    Parameters:
    -----------
    df : DataFrame
        Data for this channel-band-condition
    metric : str
        Metric to test (should be Fisher z-transformed)
    condition : str
        Condition name
    channel : str
        Channel name
    band : str
        Band name
    
    Returns:
    --------
    result : dict
        Test results
    """
    values = df[metric].values
    
    # Check variance
    if len(values) < 2 or np.std(values) < 1e-10:
        return None
    
    # One-sample t-test against zero
    t_stat, p_value = stats.ttest_1samp(values, 0)
    
    return {
        'condition': condition,
        'channel': channel,
        'band': band,
        'mean_z': np.mean(values),
        'std_z': np.std(values),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_subjects': len(values)
    }


def test_all_conditions(df, metric):
    """
    Test correlations vs zero for all conditions, channels, and bands.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with Fisher z-transformed correlations
    metric : str
        Metric to test
    
    Returns:
    --------
    results_df : DataFrame
        Statistical results
    """
    # Use Fisher z-transformed metric
    if metric == 'correlation_direct' and 'correlation_direct_z' in df.columns:
        analysis_metric = 'correlation_direct_z'
        print(f"\nUsing Fisher z-transformed metric: {analysis_metric}")
    elif metric == 'correlation_max_lagged' and 'correlation_max_lagged_z' in df.columns:
        analysis_metric = 'correlation_max_lagged_z'
        print(f"\nUsing Fisher z-transformed metric: {analysis_metric}")
    else:
        # Create z-transformed version
        if metric in df.columns:
            df[f'{metric}_z'] = fisher_z_transform(df[metric])
            analysis_metric = f'{metric}_z'
            print(f"\nCreated Fisher z-transformed metric: {analysis_metric}")
        else:
            print(f"Error: {metric} not found in data")
            return None
    
    results_list = []
    
    conditions = sorted(df['condition'].unique())
    bands = sorted(df['band'].unique())
    channels = sorted(df['channel'].unique())
    
    print(f"\nTesting {len(conditions)} conditions × {len(bands)} bands × {len(channels)} channels...")
    
    for condition in conditions:
        print(f"\n  Condition: {condition}")
        cond_df = df[df['condition'] == condition]
        
        for band in bands:
            band_df = cond_df[cond_df['band'] == band]
            
            succeeded = 0
            for channel in channels:
                channel_df = band_df[band_df['channel'] == channel]
                
                if len(channel_df) < 2:
                    continue
                
                try:
                    result = test_correlation_vs_zero(channel_df, analysis_metric, 
                                                     condition, channel, band)
                    if result is not None:
                        results_list.append(result)
                        succeeded += 1
                except Exception as e:
                    continue
            
            if succeeded > 0:
                print(f"    {band}: {succeeded}/{len(channels)} channels")
    
    if len(results_list) == 0:
        raise ValueError("No valid results obtained")
    
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction within each condition-band combination
    print("\nApplying FDR correction within each condition-band...")
    results_df['p_fdr'] = np.nan
    
    for condition in results_df['condition'].unique():
        for band in results_df['band'].unique():
            mask = (results_df['condition'] == condition) & (results_df['band'] == band)
            p_values = results_df.loc[mask, 'p_value'].values
            
            if len(p_values) == 0:
                continue
            
            # Benjamini-Hochberg FDR
            n_tests = len(p_values)
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            p_fdr = np.ones(n_tests)
            for i in range(n_tests):
                p_fdr[sorted_idx[i]] = sorted_p[i] * n_tests / (i + 1)
            
            # Ensure monotonicity
            p_fdr = np.minimum.accumulate(p_fdr[::-1])[::-1]
            p_fdr = np.minimum(p_fdr, 1.0)
            
            results_df.loc[mask, 'p_fdr'] = p_fdr
            
            n_sig = np.sum(p_fdr < 0.05)
            print(f"  {condition} - {band}: {n_sig}/{n_tests} channels significant")
    
    return results_df


def plot_topomap_per_condition(results_df, metric, output_dir):
    """
    Create topomaps showing significant correlations within each condition.
    One plot per condition with subplots for each band.
    Uses MNE standard 10-20 montage for proper scalp representation.
    """
    import mne
    
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
    
    conditions = sorted(results_df['condition'].unique())
    bands = sorted(results_df['band'].unique())
    
    for condition in conditions:
        cond_df = results_df[results_df['condition'] == condition]
        
        n_bands = len(bands)
        ncols = 3
        nrows = (n_bands + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, band in enumerate(bands):
            ax = axes[idx]
            band_df = cond_df[cond_df['band'] == band]
            
            if len(band_df) == 0:
                ax.axis('off')
                continue
            
            # Get frequency range
            freq_bands = {
                'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
                'low_beta': (13, 20), 'high_beta': (20, 30), 'gamma1': (30, 50)
            }
            lowcut, highcut = freq_bands.get(band, (0, 0))
            
            # Prepare data for MNE
            n_channels = len(band_df)
            mne_ch_names = standard_32_channels[:n_channels]
            
            # Create MNE Info object
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
            info.set_montage(montage)
            
            values = []
            sig_mask = []
            
            for _, row in band_df.iterrows():
                values.append(row['mean_z'])
                sig_mask.append(row['p_fdr'] < 0.05)
            
            if len(values) == 0:
                ax.axis('off')
                continue
            
            values = np.array(values)
            sig_mask = np.array(sig_mask)
            
            # Determine colormap range
            vmax = max(np.abs(np.nanmax(values)), np.abs(np.nanmin(values)))
            if vmax == 0:
                vmax = 1e-6
            
            # Plot topomap using MNE
            im, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(-vmax, vmax),
                contours=0,
                sensors=False
            )
            
            # Mark significant electrodes
            if np.any(sig_mask):
                from mne.viz.topomap import _get_pos_outlines
                pos_xy, outlines = _get_pos_outlines(info, None, sphere=None)
                sig_positions = pos_xy[sig_mask]
                ax.scatter(sig_positions[:, 0], sig_positions[:, 1],
                          s=80, c='yellow', marker='o', edgecolors='black',
                          linewidths=2, zorder=10)
            
            n_sig = np.sum(sig_mask)
            ax.set_title(f'{band.upper()} ({lowcut}-{highcut} Hz)\n{n_sig}/{len(sig_mask)} sig. (FDR<0.05)',
                        fontsize=11, fontweight='bold')
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Mean z')
        
        # Remove empty subplots
        for idx in range(len(bands), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Significant Audio-EEG Correlations: {condition} Condition\n({metric})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'significance_within_{condition}_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_file.name}")


def plot_comparison_across_conditions(results_df, metric, output_dir):
    """
    Create comparison showing which channels are significant in each condition.
    One plot per band showing all conditions side-by-side.
    """
    bands = sorted(results_df['band'].unique())
    conditions = sorted(results_df['condition'].unique())
    channels = sorted(results_df['channel'].unique())
    
    for band in bands:
        band_df = results_df[results_df['band'] == band]
        
        # Get frequency range
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'low_beta': (13, 20), 'high_beta': (20, 30), 'gamma1': (30, 50)
        }
        lowcut, highcut = freq_bands.get(band, (0, 0))
        
        # Create matrix: channels × conditions
        sig_matrix = np.zeros((len(channels), len(conditions)))
        value_matrix = np.zeros((len(channels), len(conditions)))
        
        for i, ch in enumerate(channels):
            for j, cond in enumerate(conditions):
                cond_ch_df = band_df[(band_df['channel'] == ch) & (band_df['condition'] == cond)]
                if len(cond_ch_df) > 0:
                    value_matrix[i, j] = cond_ch_df['mean_z'].values[0]
                    sig_matrix[i, j] = cond_ch_df['p_fdr'].values[0] < 0.05
                else:
                    value_matrix[i, j] = np.nan
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 16))
        
        # Heatmap
        vmax = np.nanmax(np.abs(value_matrix))
        im = ax.imshow(value_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        
        # Mark significant with stars
        for i in range(len(channels)):
            for j in range(len(conditions)):
                if sig_matrix[i, j]:
                    ax.text(j, i, '★', ha='center', va='center', 
                           color='yellow', fontsize=14, fontweight='bold')
        
        # Labels
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels([ch.replace('EEG-', '') for ch in channels], fontsize=8)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Mean Correlation (Fisher z)')
        
        ax.set_title(f'{band.upper()} ({lowcut}-{highcut} Hz)\nSignificant Audio-EEG Coupling by Condition\n(★ = FDR < 0.05)',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = output_dir / f'comparison_across_conditions_{band}_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Test correlation significance within conditions')
    parser.add_argument('--metric', type=str, required=True,
                       choices=['correlation_direct', 'correlation_max_lagged'],
                       help='Metric to analyze')
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
    
    # Load data
    results_file = results_dir / 'audio_eeg_correlation_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    print("="*60)
    print("WITHIN-CONDITION CORRELATION SIGNIFICANCE TESTING")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Testing if correlations ≠ 0 within each condition")
    print(f"Subjects: {sorted(df['subject_id'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Channels: {len(df['channel'].unique())}")
    
    # Test correlations
    results_df = test_all_conditions(df, args.metric)
    
    # Create output directory
    output_dir = results_dir / 'within_condition_significance'
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    stats_file = output_dir / f'within_condition_stats_{args.metric}.csv'
    results_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics: {stats_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_topomap_per_condition(results_df, args.metric, output_dir)
    plot_comparison_across_conditions(results_df, args.metric, output_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Significant Channels by Condition-Band")
    print("="*60)
    
    for condition in sorted(results_df['condition'].unique()):
        print(f"\n{condition}:")
        cond_df = results_df[results_df['condition'] == condition]
        for band in sorted(cond_df['band'].unique()):
            band_df = cond_df[cond_df['band'] == band]
            n_sig = np.sum(band_df['p_fdr'] < 0.05)
            n_total = len(band_df)
            print(f"  {band}: {n_sig}/{n_total} channels")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
