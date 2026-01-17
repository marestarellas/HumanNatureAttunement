"""
Compare respiration-audio coupling across conditions with temporal evolution plots.

Creates:
1. Temporal evolution plots with overlaid trajectories for each condition
2. Violin plots comparing coupling metrics across conditions
3. Statistical comparisons using linear mixed models
4. Summary statistics

Usage:
    python compare_coupling_conditions.py --subjects 2 3 4 5 6
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


def load_coupling_data(subjects, data_dir, conditions):
    """Load coupling JSON files for all subjects and conditions."""
    print(f"\nLoading coupling data for {len(subjects)} subjects")
    
    all_data = []
    
    for subj in subjects:
        subj_folder = f'sub-{subj:02d}'
        tables_dir = data_dir / 'processed' / subj_folder / 'tables'
        
        for cond in conditions:
            json_file = tables_dir / f'coupling_{cond}.json'
            
            if not json_file.exists():
                print(f"  WARNING: {json_file} not found")
                continue
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['subject_id'] = subj
                all_data.append(data)
                print(f"  Loaded: sub-{subj:02d}, {cond}")
    
    if not all_data:
        raise ValueError("No coupling data loaded!")
    
    print(f"\nTotal: {len(all_data)} condition × subject combinations")
    return all_data


def create_temporal_evolution_plot(coupling_data, metric, output_file):
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
            
            # Keep time in absolute seconds from start of condition
            # (times are already in seconds from the coupling analysis)
            
            all_times.append(times)
            all_values.append(values)
        
        if not all_times:
            continue
        
        # Find the shortest time series to avoid extrapolation
        min_length = min(len(t) for t in all_times)
        max_common_time = min(t[min(len(t)-1, min_length-1)] for t in all_times)
        
        # Create common time grid based on the actual sampling
        # Use the most common time points (typically 10s steps with 120s windows)
        dt = np.median([np.median(np.diff(t)) for t in all_times if len(t) > 1])
        time_grid = np.arange(0, max_common_time + dt/2, dt)
        
        # Interpolate all subjects to common grid
        interp_values = []
        for times, values in zip(all_times, all_values):
            if len(times) > 1:
                # Only interpolate within the valid range
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
        
        # Plot mean line with shaded error
        ax.plot(time_grid[valid_mask], mean_values[valid_mask], linewidth=2.5, label=cond, 
                color=COLORS.get(cond, 'gray'), alpha=0.9)
        ax.fill_between(time_grid[valid_mask], 
                        mean_values[valid_mask] - sem_values[valid_mask], 
                        mean_values[valid_mask] + sem_values[valid_mask],
                        color=COLORS.get(cond, 'gray'), alpha=0.2)
    
    # Labels and formatting
    metric_labels = {
        'xcorr': 'Cross-correlation (peak r)',
        'coherence': 'Coherence (0.05-0.5 Hz)',
        'plv': 'Phase Locking Value',
        'wpli': 'Weighted Phase Lag Index'
    }
    
    ax.set_xlabel('Time from condition onset (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    ax.set_title(f'Temporal Evolution: {metric_labels.get(metric, metric)}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable x-axis limits (show first 5 minutes)
    ax.set_xlim(0, min(300, ax.get_xlim()[1]))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


def create_summary_dataframe(coupling_data):
    """Convert coupling data to a summary dataframe for statistical analysis."""
    rows = []
    
    for data in coupling_data:
        if data['condition'] == 'AUDIO_SYNC':
            continue
        
        row = {
            'subject_id': data['subject_id'],
            'condition': data['condition'],
            'xcorr_peak_r': data['xcorr']['mean_peak_r'],
            'xcorr_lag_s': data['xcorr']['mean_peak_lag_s'],
            'coh_band_avg': data['coherence']['band_avg_coh'],
            'coh_peak': data['coherence']['peak_coh'],
            'plv': data['plv']['plv'],
            'plv_lag_s': data['plv']['preferred_lag_s'],
            'wpli': data['wpli']['wpli']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_violin_comparison(df, metric, output_file):
    """Create violin plot comparing a metric across conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get conditions in order
    conditions = ['VIZ', 'AUD', 'MULTI']
    conditions = [c for c in conditions if c in df['condition'].unique()]
    
    # Prepare data
    plot_data = []
    for cond in conditions:
        values = df[df['condition'] == cond][metric].values
        for val in values:
            plot_data.append({'condition': cond, 'value': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot
    parts = ax.violinplot(
        [plot_df[plot_df['condition'] == c]['value'].values for c in conditions],
        positions=range(len(conditions)),
        widths=0.6,
        showmeans=False,
        showextrema=False
    )
    
    # Color violins
    for i, (pc, cond) in enumerate(zip(parts['bodies'], conditions)):
        pc.set_facecolor(COLORS.get(cond, 'gray'))
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Add subject points
    for i, cond in enumerate(conditions):
        values = df[df['condition'] == cond][metric].values
        x = np.random.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.5, s=50, color='black', zorder=3)
    
    # Add mean line
    means = [df[df['condition'] == c][metric].mean() for c in conditions]
    ax.plot(range(len(conditions)), means, 'D-', color='gold', linewidth=3, 
            markersize=10, label='Mean', zorder=10, markeredgecolor='black', markeredgewidth=1)
    
    # Labels
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Comparison: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


def run_statistical_tests(df, metrics):
    """Run repeated-measures ANOVA and post-hoc tests."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    results = []
    
    for metric in metrics:
        print(f"\n{metric}:")
        
        # Get conditions (exclude RS1/RS2 for now)
        conditions = ['VIZ', 'AUD', 'MULTI']
        conditions = [c for c in conditions if c in df['condition'].unique()]
        
        df_metric = df[df['condition'].isin(conditions)]
        
        # Repeated measures ANOVA using Friedman test (non-parametric)
        # Prepare data in wide format
        subjects = df_metric['subject_id'].unique()
        data_matrix = []
        for subj in subjects:
            row = []
            for cond in conditions:
                val = df_metric[(df_metric['subject_id'] == subj) & 
                               (df_metric['condition'] == cond)][metric].values
                if len(val) > 0:
                    row.append(val[0])
                else:
                    row.append(np.nan)
            if not any(np.isnan(row)):
                data_matrix.append(row)
        
        if len(data_matrix) < 3:
            print(f"  Insufficient data for {metric}")
            continue
        
        data_matrix = np.array(data_matrix)
        
        # Friedman test
        try:
            stat, p = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(len(conditions))])
            print(f"  Friedman test: χ²={stat:.3f}, p={p:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            stat, p = np.nan, np.nan
        
        # Post-hoc pairwise Wilcoxon tests
        posthoc = []
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i < j:
                    data1 = data_matrix[:, i]
                    data2 = data_matrix[:, j]
                    try:
                        w_stat, w_p = stats.wilcoxon(data1, data2, alternative='two-sided')
                        posthoc.append({
                            'pair': f'{cond1}_vs_{cond2}',
                            'stat': w_stat,
                            'p': w_p
                        })
                        print(f"  {cond1} vs {cond2}: W={w_stat:.1f}, p={w_p:.4f}")
                    except Exception as e:
                        print(f"  {cond1} vs {cond2}: ERROR - {e}")
        
        results.append({
            'metric': metric,
            'friedman_stat': stat,
            'friedman_p': p,
            'posthoc': posthoc
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare coupling across conditions')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                      help='Subject IDs to include (default: 2 3 4 5 6)')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                      help='Root data directory')
    parser.add_argument('--output-dir', type=Path, 
                      default=Path('../results/coupling_comparison'),
                      help='Output directory for plots')
    parser.add_argument('--conditions', type=str, nargs='+', 
                      default=['VIZ', 'AUD', 'MULTI', 'RS1', 'RS2'],
                      help='Conditions to compare')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    coupling_data = load_coupling_data(args.subjects, args.data_dir, args.conditions)
    
    # Create summary dataframe
    df_summary = create_summary_dataframe(coupling_data)
    
    # Save summary CSV
    summary_file = args.output_dir / 'coupling_summary_all.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved summary: {summary_file}")
    
    # ===== TEMPORAL EVOLUTION PLOTS =====
    print("\n" + "="*80)
    print("CREATING TEMPORAL EVOLUTION PLOTS")
    print("="*80)
    
    metrics = ['xcorr', 'coherence', 'plv', 'wpli']
    
    for metric in metrics:
        output_file = args.output_dir / f'temporal_evolution_{metric}.png'
        create_temporal_evolution_plot(coupling_data, metric, output_file)
    
    # ===== VIOLIN COMPARISON PLOTS =====
    print("\n" + "="*80)
    print("CREATING VIOLIN COMPARISON PLOTS")
    print("="*80)
    
    summary_metrics = ['xcorr_peak_r', 'coh_band_avg', 'plv', 'wpli']
    
    for metric in summary_metrics:
        if metric in df_summary.columns:
            output_file = args.output_dir / f'violin_{metric}.png'
            plot_violin_comparison(df_summary, metric, output_file)
    
    # ===== STATISTICAL TESTS =====
    stat_results = run_statistical_tests(df_summary, summary_metrics)
    
    # Save statistical results
    stats_file = args.output_dir / 'statistical_results.json'
    with open(stats_file, 'w') as f:
        json.dump(stat_results, f, indent=2)
    print(f"\nSaved statistical results: {stats_file}")
    
    # ===== SUMMARY STATISTICS TABLE =====
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for metric in summary_metrics:
        if metric not in df_summary.columns:
            continue
        print(f"\n{metric}:")
        for cond in df_summary['condition'].unique():
            values = df_summary[df_summary['condition'] == cond][metric].values
            print(f"  {cond:8s}: mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, "
                  f"median={np.median(values):.4f}, "
                  f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    print("\n" + "="*80)
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
