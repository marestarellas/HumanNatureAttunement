"""
Create intuitive visualizations of correlation changes between conditions.

Shows individual subject changes and group patterns for audio-EEG correlations.

Usage:
    python plot_correlation_changes.py --metric correlation_direct --condition1 VIZ --condition2 AUD
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


def plot_subject_changes_per_band(df, metric, condition1, condition2, output_dir):
    """
    Plot showing individual subject changes between conditions for each band.
    Shows raw correlation values (not z-transformed) for interpretability.
    """
    # Use raw correlation metric for visualization
    if metric.endswith('_z'):
        raw_metric = metric.replace('_z', '')
    else:
        raw_metric = metric
    
    if raw_metric not in df.columns:
        print(f"Warning: {raw_metric} not found in data")
        return
    
    bands = sorted(df['band'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = df[df['band'] == band].copy()
        
        # Calculate mean per subject-condition
        subject_means = band_df.groupby(['subject_id', 'condition'])[raw_metric].mean().reset_index()
        
        # Pivot to get VIZ and AUD in columns
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=raw_metric)
        
        if condition1 not in pivot.columns or condition2 not in pivot.columns:
            continue
        
        # Calculate changes
        pivot['change'] = pivot[condition2] - pivot[condition1]
        
        # Plot 1: Individual trajectories
        subjects = sorted(pivot.index)
        colors = plt.cm.Set2(np.linspace(0, 1, len(subjects)))
        
        for i, subj in enumerate(subjects):
            if subj in pivot.index:
                viz_val = pivot.loc[subj, condition1]
                aud_val = pivot.loc[subj, condition2]
                
                # Plot line
                ax.plot([0, 1], [viz_val, aud_val], '-o', 
                       color=colors[i], alpha=0.7, linewidth=2, markersize=8,
                       label=f'Sub-{subj}')
        
        # Add mean trajectory
        mean_viz = pivot[condition1].mean()
        mean_aud = pivot[condition2].mean()
        ax.plot([0, 1], [mean_viz, mean_aud], 'k-o', 
               linewidth=3, markersize=10, label='Mean', zorder=100)
        
        # Styling
        ax.set_xticks([0, 1])
        ax.set_xticklabels([condition1, condition2], fontsize=11)
        ax.set_ylabel('Correlation (r)', fontsize=11)
        ax.set_title(f'{band.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
        
        # Add change statistics
        mean_change = pivot['change'].mean()
        std_change = pivot['change'].std()
        ax.text(0.5, 0.02, f'Δ = {mean_change:.4f} +/- {std_change:.4f}',
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplots
    for idx in range(len(bands), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Individual Subject Changes: {condition1} -> {condition2}\n{raw_metric}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'subject_changes_{raw_metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file.name}")


def plot_violin_comparison(df, metric, condition1, condition2, output_dir, stats_df=None, stats_type=None):
    """Paper-ready 2x3 grid: per-band distribution of EEG-audio correlation by condition."""
    # Toolbox style + canonical colors.
    import sys as _sys
    _REPO = Path(__file__).resolve().parents[2]
    if str(_REPO / "src") not in _sys.path:
        _sys.path.insert(0, str(_REPO / "src"))
    from HNA.modules.viz import use_paper_style, CONDITION_COLORS, save_figure, sig_stars
    use_paper_style()

    # Use raw correlation metric for visualization (z-transform happens in stats).
    raw_metric = metric.replace('_z', '') if metric.endswith('_z') else metric
    if raw_metric not in df.columns:
        print(f"Warning: {raw_metric} not found in data")
        return

    # Standard EEG-band order (left-to-right, top-to-bottom).
    band_order = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'gamma1']
    bands = [b for b in band_order if b in df['band'].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.4), sharey=False)
    axes = axes.flatten()

    color_a = CONDITION_COLORS.get(condition1, '#3B7DD8')
    color_b = CONDITION_COLORS.get(condition2, '#C9325F')

    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = df[df['band'] == band].copy()

        a_vals = band_df[band_df['condition'] == condition1][raw_metric].values
        b_vals = band_df[band_df['condition'] == condition2][raw_metric].values

        parts = ax.violinplot([a_vals, b_vals], positions=[0, 1], widths=0.7,
                              showmeans=False, showextrema=False)
        for pc, color in zip(parts['bodies'], [color_a, color_b]):
            pc.set_facecolor(color); pc.set_alpha(0.55)
            pc.set_edgecolor(color); pc.set_linewidth(0.9)

        # Subject means + paired connectors.
        subject_means = band_df.groupby(['subject_id', 'condition'])[raw_metric].mean().reset_index()
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=raw_metric)
        if condition1 in pivot.columns and condition2 in pivot.columns:
            for _, row in pivot.iterrows():
                if not np.isnan(row[condition1]) and not np.isnan(row[condition2]):
                    ax.plot([0, 1], [row[condition1], row[condition2]],
                            color='#9aa0a6', alpha=0.55, lw=0.9)
            avg_vals = [pivot[condition1].mean(), pivot[condition2].mean()]
            ax.plot([0, 1], avg_vals, '-D', color='#f5b400', lw=1.7, ms=8,
                    mec='black', mew=0.6, zorder=10)
        for cond, pos in [(condition1, 0), (condition2, 1)]:
            vals = subject_means[subject_means['condition'] == cond][raw_metric].values
            x_j = pos + np.random.uniform(-0.05, 0.05, len(vals))
            ax.scatter(x_j, vals, s=22, color='black', alpha=0.7, zorder=6)

        ax.set_xticks([0, 1]); ax.set_xticklabels([condition1, condition2])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(band.upper(), fontsize=11.5, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.30)
        # Per-subplot y-axis: tighten around the data of THIS band so subtle changes are visible.
        all_vals = np.concatenate([a_vals, b_vals])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            margin = max(1e-6, (hi - lo) * 0.12)
            ax.set_ylim(lo - margin, hi + margin)

        # Significance bracket between the two violins.
        # Per-band p-value: aggregated stats expose `p_value`; the per-channel
        # version is reduced to the smallest FDR-corrected channel p (so the bracket
        # answers "is there ANY channel that survives FDR within this band?").
        if stats_df is not None:
            band_stats = stats_df[stats_df['band'] == band]
            if len(band_stats) > 0:
                if stats_type == 'aggregated' and 'p_value' in band_stats.columns:
                    p_val = float(band_stats['p_value'].iloc[0])
                else:
                    p_val = float(band_stats['p_fdr'].min())
                stars = sig_stars(p_val)
                if np.isfinite(p_val):
                    # Bracket between x=0 and x=1 placed in the panel's headroom.
                    y_lo, y_hi = ax.get_ylim()
                    span = y_hi - y_lo
                    bracket_y = y_hi - 0.10 * span
                    ax.plot([0, 1], [bracket_y, bracket_y], color='#222', lw=0.9)
                    ax.text(0.5, bracket_y + 0.01 * span, stars,
                            ha='center', va='bottom',
                            fontsize=14 if stars not in ('ns', 'n/a') else 11,
                            fontweight='bold' if stars not in ('ns', 'n/a') else 'normal',
                            color='#222' if stars not in ('ns', 'n/a') else '#888')

    # Y-axis label on every left-column panel (since axes no longer share y).
    for ax in axes:
        ax.set_ylabel("Correlation (r)")
        ax.tick_params(axis='y', labelleft=True)

    # Title removed (in caption).
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = output_dir / f'violin_comparison_{raw_metric}_{condition1}_vs_{condition2}.png'
    save_figure(fig, Path(output_file).with_suffix(''))
    plt.close()
    print(f"Saved: {output_file.name}")


def plot_heatmap_all_channels(df, metric, condition1, condition2, output_dir):
    """
    Heatmap showing correlation change (condition2 - condition1) for all channels and bands.
    """
    # Use raw correlation metric
    if metric.endswith('_z'):
        raw_metric = metric.replace('_z', '')
    else:
        raw_metric = metric
    
    if raw_metric not in df.columns:
        print(f"Warning: {raw_metric} not found in data")
        return
    
    # Calculate mean correlation per channel-band-condition
    means = df.groupby(['channel', 'band', 'condition'])[raw_metric].mean().reset_index()
    
    # Pivot to get conditions in columns
    pivot = means.pivot_table(index='channel', columns=['band', 'condition'], values=raw_metric)
    
    # Calculate differences
    bands = sorted(df['band'].unique())
    diff_data = []
    
    for band in bands:
        if (band, condition1) in pivot.columns and (band, condition2) in pivot.columns:
            diff = pivot[(band, condition2)] - pivot[(band, condition1)]
            diff_data.append(diff)
    
    if len(diff_data) == 0:
        print("Warning: Could not create heatmap")
        return
    
    # Create DataFrame
    diff_df = pd.concat(diff_data, axis=1)
    diff_df.columns = [band.upper() for band in bands if (band, condition1) in pivot.columns]
    
    # Sort channels
    diff_df = diff_df.sort_index()
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    # Use diverging colormap
    vmax = np.abs(diff_df.values).max()
    heatmap = sns.heatmap(diff_df, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
               cbar_kws={'label': f'Δ Correlation ({condition2} - {condition1})'}, 
               linewidths=0.5, ax=ax, annot=False)
    
    # Add directional labels to colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.text(1.5, 1.02, condition2, transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    cbar.ax.text(1.5, -0.02, condition1, transform=cbar.ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold', color='darkblue')
    
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'Change in Correlation: {condition1} -> {condition2}\n{raw_metric}', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / f'heatmap_changes_{raw_metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Visualize correlation changes between conditions')
    parser.add_argument('--metric', type=str, required=True,
                       choices=['correlation_direct', 'correlation_max_lagged',
                               'correlation_direct_z', 'correlation_max_lagged_z'],
                       help='Metric to visualize')
    parser.add_argument('--condition1', type=str, required=True,
                       help='First condition')
    parser.add_argument('--condition2', type=str, required=True,
                       help='Second condition')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: <repo>/results/audio_eeg_correlation)')
    parser.add_argument('--figures-dir', type=Path, default=None,
                       help='Figures directory (default: <repo>/figures/audio_eeg_correlation)')

    args = parser.parse_args()

    # Setup paths
    project_dir = Path(__file__).resolve().parents[2]

    if args.results_dir is None:
        results_dir = project_dir / 'results' / 'audio_eeg_correlation'
    else:
        results_dir = Path(args.results_dir)

    if args.figures_dir is None:
        output_dir = project_dir / 'figures' / 'audio_eeg_correlation'
    else:
        output_dir = Path(args.figures_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load data
    results_file = results_dir / 'audio_eeg_correlation_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    df = df[df['condition'].isin([args.condition1, args.condition2])].copy()

    print("="*60)
    print("CORRELATION CHANGE VISUALIZATIONS")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Comparison: {args.condition1} vs {args.condition2}")
    print(f"Subjects: {sorted(df['subject_id'].unique())}")
    print(f"Channels: {len(df['channel'].unique())}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Load statistics - try aggregated first (better for violin plots), fallback to per-channel
    # The aggregated stats are typically computed on _z transformed data
    stats_file_agg = results_dir / 'statistics' / f'stats_aggregated_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    if not stats_file_agg.exists():
        # Try with _z suffix if not already present (aggregated stats use _z)
        if not args.metric.endswith('_z'):
            stats_file_agg = results_dir / 'statistics' / f'stats_aggregated_{args.metric}_z_{args.condition1}_vs_{args.condition2}.csv'
        else:
            # Try without _z suffix
            metric_base = args.metric.replace('_z', '')
            stats_file_agg = results_dir / 'statistics' / f'stats_aggregated_{metric_base}_{args.condition1}_vs_{args.condition2}.csv'
    
    stats_file_perchan = results_dir / 'statistics' / f'stats_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    if not stats_file_perchan.exists():
        # Try without _z suffix or with _z suffix
        if args.metric.endswith('_z'):
            metric_base = args.metric.replace('_z', '')
            stats_file_perchan = results_dir / 'statistics' / f'stats_{metric_base}_{args.condition1}_vs_{args.condition2}.csv'
        else:
            stats_file_perchan = results_dir / 'statistics' / f'stats_{args.metric}_z_{args.condition1}_vs_{args.condition2}.csv'
    
    stats_df = None
    stats_type = None
    if stats_file_agg.exists():
        stats_df = pd.read_csv(stats_file_agg)
        stats_type = 'aggregated'
        print(f"Loaded aggregated statistics from: {stats_file_agg.name}")
    elif stats_file_perchan.exists():
        stats_df = pd.read_csv(stats_file_perchan)
        stats_type = 'per-channel'
        print(f"Loaded per-channel statistics from: {stats_file_perchan.name}")
    else:
        print("Warning: No statistics file found, significance stars will not be shown")
    
    # Plot 1: Subject changes
    plot_subject_changes_per_band(df, args.metric, args.condition1, args.condition2, output_dir)
    
    # Plot 2: Violin plots (with significance stars)
    plot_violin_comparison(df, args.metric, args.condition1, args.condition2, output_dir, stats_df, stats_type)
    
    # Plot 3: Heatmap
    plot_heatmap_all_channels(df, args.metric, args.condition1, args.condition2, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
