"""
Compare HRV-audio coupling across conditions with temporal evolution plots.

Creates:
1. Temporal evolution plots with overlaid trajectories for each condition
2. Violin plots comparing coupling metrics across conditions
3. Statistical comparisons using Friedman test + pairwise Wilcoxon
4. Summary statistics

Usage:
    python compare_hrv_audio_coupling.py --subjects 2 3 4 5 6 --hrv-feature HRV_RMSSD
    python compare_hrv_audio_coupling.py --subjects 2 3 4 5 6 --hrv-feature HRV_RMSSD HRV_MeanNN
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Modern color palette for conditions
COLORS = {
    'VIZ': '#2E86AB',      # Deep blue
    'AUD': '#A23B72',      # Deep magenta
    'MULTI': '#F18F01',    # Orange
    'RS1': '#6A994E',      # Green
    'RS2': '#BC4749'       # Red
}

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results' / 'hrv_audio_coupling_comparison'


def load_coupling_data(subjects, data_dir, conditions, hrv_feature):
    """Load HRV-audio coupling JSON files for all subjects and conditions."""
    print(f"\nLoading {hrv_feature} coupling data for {len(subjects)} subjects")
    
    all_data = []
    
    for subj in subjects:
        subj_folder = f'sub-{subj:02d}'
        tables_dir = data_dir / 'processed' / subj_folder / 'tables'
        
        for cond in conditions:
            json_file = tables_dir / f'hrv_audio_coupling_{cond}_{hrv_feature}.json'
            
            if not json_file.exists():
                print(f"  WARNING: {json_file} not found")
                continue
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['subject_id'] = subj
                all_data.append(data)
                print(f"  Loaded: sub-{subj:02d}, {cond}")
    
    if not all_data:
        raise ValueError(f"No coupling data loaded for {hrv_feature}!")
    
    print(f"\nTotal: {len(all_data)} condition × subject combinations")
    return all_data


def create_temporal_evolution_plot(coupling_data, metric, output_file, hrv_feature):
    """
    Create overlaid temporal evolution plot for a coupling metric across conditions.
    
    Parameters
    ----------
    coupling_data : list of dict
        Coupling data from JSON files
    metric : str
        Metric to plot: 'xcorr', 'coherence', 'plv', or 'wpli'
    output_file : Path
        Output file path
    hrv_feature : str
        HRV feature name
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get conditions in order
    conditions = sorted(set(d['condition'] for d in coupling_data if d['condition'] != 'AUDIO_SYNC'))
    
    for cond in conditions:
        # Collect all time series for this condition across subjects
        all_times = []
        all_values = []
        
        for data in coupling_data:
            if data['condition'] != cond:
                continue
            
            if metric == 'xcorr':
                times = np.array(data['xcorr']['times_s'])
                values = np.array(data['xcorr']['peak_r'])
            elif metric == 'coherence':
                times = np.array(data['coherence']['times_s'])
                values = np.array(data['coherence']['band_avg_coh_win'])
            elif metric == 'plv':
                times = np.array(data['plv']['win_times_s'])
                values = np.array(data['plv']['win_plv'])
            elif metric == 'wpli':
                times = np.array(data['wpli']['win_times_s'])
                values = np.array(data['wpli']['win_wpli'])
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            all_times.append(times)
            all_values.append(values)
        
        if not all_times:
            continue
        
        # Find common time range
        min_length = min(len(t) for t in all_times)
        max_common_time = min(t[min(len(t)-1, min_length-1)] for t in all_times)
        
        # Create common time grid based on actual sampling
        dt = np.median([np.median(np.diff(t)) for t in all_times if len(t) > 1])
        time_grid = np.arange(0, max_common_time + dt/2, dt)
        
        # Interpolate all subjects to common grid
        interp_values = []
        for times, values in zip(all_times, all_values):
            if len(times) > 1:
                valid_mask = (time_grid >= times[0]) & (time_grid <= times[-1])
                interp = np.full_like(time_grid, np.nan, dtype=float)
                interp[valid_mask] = np.interp(time_grid[valid_mask], times, values)
                interp_values.append(interp)
        
        if not interp_values:
            continue
        
        interp_values = np.array(interp_values)
        
        # Compute mean and SEM
        mean_values = np.nanmean(interp_values, axis=0)
        sem_values = np.nanstd(interp_values, axis=0) / np.sqrt(np.sum(~np.isnan(interp_values), axis=0))
        
        # Only plot where we have data from at least 2 subjects
        valid_mask = np.sum(~np.isnan(interp_values), axis=0) >= 2
        
        if not np.any(valid_mask):
            continue
        
        # Plot mean with shaded SEM
        color = COLORS.get(cond, '#666666')
        ax.plot(time_grid[valid_mask], mean_values[valid_mask], 
               label=cond, color=color, linewidth=2.5, alpha=0.9)
        ax.fill_between(time_grid[valid_mask], 
                       mean_values[valid_mask] - sem_values[valid_mask],
                       mean_values[valid_mask] + sem_values[valid_mask],
                       color=color, alpha=0.2)
    
    # Format plot
    metric_labels = {
        'xcorr': 'Cross-correlation (r)',
        'coherence': 'Coherence',
        'plv': 'Phase Locking Value',
        'wpli': 'Weighted Phase Lag Index'
    }
    
    ax.set_xlabel('Time from condition onset (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    ax.set_title(f'{hrv_feature} ↔ Audio — {metric_labels.get(metric, metric)} over time',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(300, time_grid[valid_mask][-1]))
    
    # Add zero line if relevant
    if metric in ['xcorr']:
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_violin_comparison(coupling_data, metric, output_file, hrv_feature):
    """
    Create violin plots comparing coupling metric across conditions.
    
    Parameters
    ----------
    coupling_data : list of dict
        Coupling data from JSON files
    metric : str
        Metric to plot
    output_file : Path
        Output file path
    hrv_feature : str
        HRV feature name
    """
    # Extract summary values for each subject/condition
    rows = []
    
    for data in coupling_data:
        cond = data['condition']
        if cond == 'AUDIO_SYNC':
            continue
        
        subj = data['subject_id']
        
        if metric == 'xcorr':
            value = data['xcorr']['mean_peak_r']
        elif metric == 'coherence':
            value = data['coherence']['band_avg_coh']
        elif metric == 'plv':
            value = data['plv']['plv']
        elif metric == 'wpli':
            value = data['wpli']['wpli']
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        rows.append({
            'subject': subj,
            'condition': cond,
            'value': value
        })
    
    df = pd.DataFrame(rows)
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get condition order
    conditions = sorted(df['condition'].unique())
    palette = [COLORS.get(c, '#666666') for c in conditions]
    
    # Violin plot
    sns.violinplot(data=df, x='condition', y='value', order=conditions,
                  palette=palette, ax=ax, inner='box', cut=0)
    
    # Overlay individual points
    sns.stripplot(data=df, x='condition', y='value', order=conditions,
                 color='black', alpha=0.5, size=6, ax=ax)
    
    # Format
    metric_labels = {
        'xcorr': 'Mean Cross-correlation (r)',
        'coherence': 'Mean Coherence',
        'plv': 'Phase Locking Value',
        'wpli': 'Weighted Phase Lag Index'
    }
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    ax.set_title(f'{hrv_feature} ↔ Audio — {metric_labels.get(metric, metric)} comparison',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add zero line if relevant
    if metric in ['xcorr']:
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")
    
    return df


def run_statistical_tests(df, metric):
    """
    Run statistical tests comparing coupling across conditions.
    
    Uses Friedman test (non-parametric repeated measures) followed by
    pairwise Wilcoxon signed-rank tests with FDR correction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: subject, condition, value
    metric : str
        Metric name for reporting
    
    Returns
    -------
    results : dict
        Statistical test results
    """
    print(f"\n{'='*60}")
    print(f"STATISTICAL TESTS — {metric}")
    print('='*60)
    
    # Pivot to wide format (subjects × conditions)
    df_wide = df.pivot(index='subject', columns='condition', values='value')
    conditions = df_wide.columns.tolist()
    
    print(f"\nConditions: {conditions}")
    print(f"N subjects: {len(df_wide)}")
    
    # Summary statistics
    print(f"\nSummary statistics:")
    for cond in conditions:
        vals = df_wide[cond].dropna()
        print(f"  {cond}: mean={vals.mean():.4f}, SD={vals.std():.4f}, "
              f"median={vals.median():.4f}, n={len(vals)}")
    
    # Friedman test (non-parametric repeated measures)
    # Remove subjects with missing data
    df_complete = df_wide.dropna()
    
    if len(df_complete) < 3:
        print("\nWARNING: Not enough subjects with complete data for Friedman test")
        return None
    
    data_arrays = [df_complete[cond].values for cond in conditions]
    friedman_stat, friedman_p = stats.friedmanchisquare(*data_arrays)
    
    print(f"\n{'─'*60}")
    print(f"Friedman test:")
    print(f"  χ²({len(conditions)-1}) = {friedman_stat:.4f}, p = {friedman_p:.4e}")
    print('─'*60)
    
    # Pairwise Wilcoxon signed-rank tests
    print(f"\nPairwise comparisons (Wilcoxon signed-rank):")
    
    pairs = []
    p_values = []
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            # Use only subjects with both values
            df_pair = df_wide[[cond1, cond2]].dropna()
            
            if len(df_pair) < 3:
                print(f"  {cond1} vs {cond2}: Not enough paired samples")
                continue
            
            stat, p = stats.wilcoxon(df_pair[cond1], df_pair[cond2])
            
            pairs.append(f"{cond1} vs {cond2}")
            p_values.append(p)
    
    # FDR correction
    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        for pair, p_raw, p_corr, is_sig in zip(pairs, p_values, p_corrected, reject):
            sig_mark = '***' if is_sig else ''
            print(f"  {pair}: p={p_raw:.4e}, p_corr={p_corr:.4e} {sig_mark}")
    
    results = {
        'metric': metric,
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'pairwise_tests': list(zip(pairs, p_values, p_corrected, reject))
    }
    
    return results


def create_summary_report(all_results, output_file):
    """Create summary CSV of statistical results."""
    rows = []
    
    for result in all_results:
        if result is None:
            continue
        
        base_row = {
            'hrv_feature': result.get('hrv_feature', 'N/A'),
            'metric': result['metric'],
            'friedman_stat': result['friedman_stat'],
            'friedman_p': result['friedman_p'],
        }
        
        # Add pairwise test results
        for pair, p_raw, p_corr, is_sig in result['pairwise_tests']:
            row = base_row.copy()
            row['comparison'] = pair
            row['p_value'] = p_raw
            row['p_corrected'] = p_corr
            row['significant'] = is_sig
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Saved summary: {output_file}")
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Compare HRV-audio coupling across conditions')
    parser.add_argument('--subjects', type=int, nargs='+', required=True,
                       help='Subject IDs (e.g., 2 3 4 5 6)')
    parser.add_argument('--hrv-feature', type=str, nargs='+', default=['HRV_RMSSD'],
                       help='HRV features to analyze (default: HRV_RMSSD)')
    parser.add_argument('--conditions', nargs='+', 
                       default=['VIZ', 'AUD', 'MULTI', 'RS1', 'RS2'],
                       help='Conditions to compare')
    parser.add_argument('--metrics', nargs='+',
                       default=['xcorr', 'coherence', 'plv', 'wpli'],
                       help='Metrics to analyze')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HRV-AUDIO COUPLING COMPARISON")
    print("="*80)
    print(f"Subjects: {args.subjects}")
    print(f"HRV features: {args.hrv_feature}")
    print(f"Conditions: {args.conditions}")
    print(f"Metrics: {args.metrics}")
    
    all_results = []
    
    for hrv_feat in args.hrv_feature:
        print(f"\n{'='*80}")
        print(f"Processing: {hrv_feat}")
        print('='*80)
        
        # Create output directory for this feature
        feature_dir = RESULTS_DIR / hrv_feat
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        try:
            coupling_data = load_coupling_data(args.subjects, DATA_DIR, 
                                             args.conditions, hrv_feat)
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue
        
        # Create plots and run tests for each metric
        for metric in args.metrics:
            print(f"\n{'-'*60}")
            print(f"Analyzing: {metric}")
            print('-'*60)
            
            try:
                # Temporal evolution plot
                evo_file = feature_dir / f'{metric}_temporal_evolution.png'
                create_temporal_evolution_plot(coupling_data, metric, evo_file, hrv_feat)
                
                # Violin comparison plot
                violin_file = feature_dir / f'{metric}_violin_comparison.png'
                df = plot_violin_comparison(coupling_data, metric, violin_file, hrv_feat)
                
                # Statistical tests
                result = run_statistical_tests(df, metric)
                if result:
                    result['hrv_feature'] = hrv_feat
                    all_results.append(result)
                
            except Exception as e:
                print(f"  ERROR processing {metric}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create summary report
    if all_results:
        summary_file = RESULTS_DIR / 'statistical_summary.csv'
        create_summary_report(all_results, summary_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
