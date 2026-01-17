"""
Statistical analysis of audio-EEG mutual information results.

Compares MI between conditions using mixed-effects models and generates visualizations.
Note: MI does not require Fisher z-transformation (always non-negative).

Usage:
    python run_mi_stats.py --metric mi_direct --condition1 VIZ --condition2 AUD
    python run_mi_stats.py --metric mi_max_lagged --condition1 VIZ --condition2 AUD
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import mne
import warnings
warnings.filterwarnings('ignore')


def load_mi_results(results_dir, condition1, condition2):
    """Load MI results for two conditions."""
    results_file = results_dir / 'audio_eeg_mutual_information_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Filter to specified conditions
    df = df[df['condition'].isin([condition1, condition2])].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for conditions {condition1}, {condition2}")
    
    print(f"Loaded {len(df)} records")
    print(f"  Subjects: {sorted(df['subject_id'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Bands: {sorted(df['band'].unique())}")
    print(f"  Channels: {len(df['channel'].unique())}")
    
    return df


def run_paired_ttest(df, metric, channel, band):
    """Run paired t-test for two conditions (when n_subjects < 3)."""
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
    """Run linear mixed-effects model with OLS fallback."""
    df = df.copy()
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Check variance
    if df[metric].std() < 1e-10:
        return None
    
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
    """Run statistical models for all channels and bands."""
    n_subjects = df['subject_id'].nunique()
    print(f"\nNumber of subjects detected: {n_subjects}")
    
    if n_subjects < 3:
        print("Using paired t-test (n < 3)")
        model_func = run_paired_ttest
    else:
        print("Using mixed-effects model (with OLS fallback)")
        model_func = run_mixed_model
    
    results_list = []
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
            
            if len(channel_df) < 4:
                failed += 1
                continue
            
            try:
                result = model_func(channel_df, metric, channel, band)
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
    
    # Apply FDR correction
    print("\nApplying FDR correction within each band...")
    results_df['p_fdr'] = np.nan
    
    for band in results_df['band'].unique():
        band_mask = results_df['band'] == band
        p_values = results_df.loc[band_mask, 'p_value'].values
        
        n_tests = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        fdr_threshold = 0.05
        p_fdr = np.ones(n_tests)
        for i in range(n_tests):
            p_fdr[sorted_idx[i]] = sorted_p[i] * n_tests / (i + 1)
        
        p_fdr = np.minimum.accumulate(p_fdr[::-1])[::-1]
        p_fdr = np.minimum(p_fdr, 1.0)
        
        results_df.loc[band_mask, 'p_fdr'] = p_fdr
        
        n_sig = np.sum(p_fdr < 0.05)
        print(f"  {band}: {n_sig}/{n_tests} channels significant (FDR < 0.05)")
    
    return results_df


def plot_topomap_results(results_df, metric, condition1, condition2, output_dir):
    """Plot topographic maps showing MI effects using MNE 10-20 montage."""
    # Map channel names to standard 10-20 names
    channel_mapping = {
        'EEG-ch1': 'Fp1', 'EEG-ch2': 'Fp2',
        'EEG-ch3': 'F7', 'EEG-ch4': 'F3', 'EEG-ch5': 'Fz', 'EEG-ch6': 'F4', 'EEG-ch7': 'F8',
        'EEG-ch8': 'FC5', 'EEG-ch9': 'FC1', 'EEG-ch10': 'FC2', 'EEG-ch11': 'FC6',
        'EEG-ch12': 'T7', 'EEG-ch13': 'C3', 'EEG-ch14': 'Cz', 'EEG-ch15': 'C4', 'EEG-ch16': 'T8',
        'EEG-ch17': 'CP5', 'EEG-ch18': 'CP1', 'EEG-ch19': 'CP2', 'EEG-ch20': 'CP6',
        'EEG-ch21': 'P7', 'EEG-ch22': 'P3', 'EEG-ch23': 'Pz', 'EEG-ch24': 'P4', 'EEG-ch25': 'P8',
        'EEG-ch26': 'PO9', 'EEG-ch27': 'O1', 'EEG-ch28': 'Oz', 'EEG-ch29': 'O2', 'EEG-ch30': 'PO10',
        'EEG-ch31': 'AF7', 'EEG-ch32': 'AF8',
    }
    
    # Get standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    bands = sorted(results_df['band'].unique())
    
    for band in bands:
        band_df = results_df[results_df['band'] == band].copy()
        
        # Get frequency range
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'low_beta': (13, 20), 'high_beta': (20, 30), 'gamma1': (30, 50)
        }
        lowcut, highcut = freq_bands.get(band, (0, 0))
        
        # Prepare data for MNE
        ch_names = []
        values = []
        sig_mask = []
        
        for _, row in band_df.iterrows():
            ch = row['channel']
            if ch in channel_mapping:
                ch_names.append(channel_mapping[ch])
                if 'coefficient' in row:
                    values.append(row['coefficient'])
                elif 'mean_diff' in row:
                    values.append(row['mean_diff'])
                else:
                    values.append(0)
                sig_mask.append(row['p_fdr'] < 0.05)
        
        if len(values) == 0:
            continue
        
        values = np.array(values)
        sig_mask = np.array(sig_mask)
        
        # Create fake Info object for plotting
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Plot using MNE
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Use diverging colormap since MI difference can be positive/negative
        vmax = max(np.abs(np.nanmax(values)), np.abs(np.nanmin(values)))
        
        im, cn = mne.viz.plot_topomap(
            values, info, axes=ax, show=False,
            cmap='RdBu_r', vlim=(-vmax, vmax),
            contours=6, image_interp='cubic',
            sensors=True, ch_type='eeg'
        )
        
        # Mark significant channels with yellow circles
        if np.any(sig_mask):
            sig_channels = np.array(ch_names)[sig_mask]
            sig_indices = [i for i, ch in enumerate(ch_names) if ch in sig_channels]
            
            # Get channel positions from montage
            pos = mne.channels.layout._find_topomap_coords(info, picks=sig_indices)
            ax.scatter(pos[:, 0], pos[:, 1], s=120, 
                      facecolors='none', edgecolors='yellow', 
                      linewidths=3, zorder=11)
        
        n_sig = np.sum(sig_mask)
        ax.set_title(f'{band.upper()} ({lowcut}-{highcut} Hz)\n{condition1} vs {condition2}\n({n_sig}/{len(sig_mask)} channels FDR < 0.05)',
                    fontsize=12, fontweight='bold', pad=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('MI Difference', fontsize=10)
        
        # Save
        output_file = output_dir / f'topomap_{metric}_{band}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {output_file.name}")


def plot_all_bands_summary(results_df, metric, condition1, condition2, output_dir):
    """Create summary plot with all frequency bands."""
    bands = sorted(results_df['band'].unique())
    n_bands = len(bands)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = results_df[results_df['band'] == band].copy()
        
        if 'coefficient' in band_df.columns:
            values = band_df['coefficient'].values
        elif 'mean_diff' in band_df.columns:
            values = band_df['mean_diff'].values
        else:
            continue
        
        p_fdr = band_df['p_fdr'].values
        channels = band_df['channel'].values
        
        sort_idx = np.argsort(np.abs(values))[::-1]
        
        n_show = min(15, len(values))
        x_pos = np.arange(n_show)
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_fdr[sort_idx[:n_show]]]
        
        ax.barh(x_pos, values[sort_idx[:n_show]], color=colors, alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels([ch.replace('EEG-', '') for ch in channels[sort_idx[:n_show]]], fontsize=8)
        ax.set_xlabel(f'MI Difference', fontsize=10)
        ax.set_title(f'{band.upper()}', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        n_sig = np.sum(p_fdr < 0.05)
        ax.text(0.95, 0.95, f'{n_sig}/{len(p_fdr)} sig.',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(n_bands, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Mutual Information: {metric}\n{condition1} vs {condition2}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'summary_{metric}_all_bands.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved summary: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of audio-EEG mutual information')
    parser.add_argument('--metric', type=str, required=True,
                       choices=['mi_direct', 'mi_max_lagged'],
                       help='Metric to analyze')
    parser.add_argument('--condition1', type=str, required=True,
                       help='First condition')
    parser.add_argument('--condition2', type=str, required=True,
                       help='Second condition')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: results/audio_eeg_mutual_information)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    if args.results_dir is None:
        results_dir = project_dir / 'results' / 'audio_eeg_mutual_information'
    else:
        results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Create output directory
    output_dir = results_dir / 'statistics'
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("AUDIO-EEG MUTUAL INFORMATION STATISTICS")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Comparison: {args.condition1} vs {args.condition2}")
    print(f"Results: {results_dir}")
    
    # Load data
    df = load_mi_results(results_dir, args.condition1, args.condition2)
    
    # Add frequency band info if missing
    if 'lowcut' not in df.columns:
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'low_beta': (13, 20), 'high_beta': (20, 30), 'gamma1': (30, 50)
        }
        df['lowcut'] = df['band'].map(lambda b: freq_bands.get(b, (0, 0))[0])
        df['highcut'] = df['band'].map(lambda b: freq_bands.get(b, (0, 0))[1])
    
    # Run statistical models
    results_df = run_models_all_channels(df, args.metric, args.condition1, args.condition2)
    
    # Save results
    stats_file = output_dir / f'stats_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    results_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics: {stats_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_topomap_results(results_df, args.metric, args.condition1, args.condition2, output_dir)
    plot_all_bands_summary(results_df, args.metric, args.condition1, args.condition2, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
