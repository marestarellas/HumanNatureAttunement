"""
Create intuitive visualizations of mutual information changes between conditions.

Shows individual subject changes and group patterns for audio-EEG MI.
Note: MI is always non-negative (0 = independent, higher = stronger coupling).

Usage:
    python plot_mi_changes.py --metric mi_direct --condition1 VIZ --condition2 AUD
    python plot_mi_changes.py --metric mi_max_lagged --condition1 VIZ --condition2 AUD
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
    """Plot individual subject changes between conditions for each band."""
    bands = sorted(df['band'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = df[df['band'] == band].copy()
        
        # Calculate mean per subject-condition
        subject_means = band_df.groupby(['subject_id', 'condition'])[metric].mean().reset_index()
        
        # Pivot
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=metric)
        
        if condition1 not in pivot.columns or condition2 not in pivot.columns:
            continue
        
        # Calculate changes
        pivot['change'] = pivot[condition2] - pivot[condition1]
        
        # Plot individual trajectories
        subjects = sorted(pivot.index)
        colors = plt.cm.Set2(np.linspace(0, 1, len(subjects)))
        
        for i, subj in enumerate(subjects):
            if subj in pivot.index:
                viz_val = pivot.loc[subj, condition1]
                aud_val = pivot.loc[subj, condition2]
                
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
        ax.set_ylabel('Mutual Information', fontsize=11)
        ax.set_title(f'{band.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
        
        # Ensure y-axis starts at 0 (MI is always non-negative)
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
        
        # Add change statistics
        mean_change = pivot['change'].mean()
        std_change = pivot['change'].std()
        ax.text(0.5, 0.02, f'Δ = {mean_change:.4f} ± {std_change:.4f}',
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplots
    for idx in range(len(bands), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Individual Subject Changes: {condition1} → {condition2}\n{metric}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'subject_changes_{metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file.name}")


def plot_violin_comparison(df, metric, condition1, condition2, output_dir, stats_df=None, stats_type=None):
    """Violin plots showing distribution of MI per condition and band."""
    bands = sorted(df['band'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_df = df[df['band'] == band].copy()
        
        # Violin plot
        parts = ax.violinplot(
            [band_df[band_df['condition'] == condition1][metric].values,
             band_df[band_df['condition'] == condition2][metric].values],
            positions=[0, 1], widths=0.6, showmeans=True, showextrema=True
        )
        
        # Color violins
        for pc, color in zip(parts['bodies'], ['#1f77b4', '#ff7f0e']):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        # Add individual points (subject means)
        subject_means = band_df.groupby(['subject_id', 'condition'])[metric].mean().reset_index()
        
        for cond, pos in [(condition1, 0), (condition2, 1)]:
            cond_data = subject_means[subject_means['condition'] == cond][metric].values
            x_jitter = np.random.normal(pos, 0.04, len(cond_data))
            ax.scatter(x_jitter, cond_data, alpha=0.8, s=50, c='black', 
                      edgecolors='white', linewidths=1.5)
        
        # Add connecting lines for paired data
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=metric)
        if condition1 in pivot.columns and condition2 in pivot.columns:
            for subj in pivot.index:
                if not np.isnan(pivot.loc[subj, condition1]) and not np.isnan(pivot.loc[subj, condition2]):
                    ax.plot([0, 1], 
                           [pivot.loc[subj, condition1], pivot.loc[subj, condition2]], 
                           'k-', alpha=0.2, linewidth=1)
        
        # Add AVERAGE LINE across all participants
        if condition1 in pivot.columns and condition2 in pivot.columns:
            avg_vals = [pivot[condition1].mean(), pivot[condition2].mean()]
            ax.plot([0, 1], avg_vals, 'k-', linewidth=3, alpha=0.8, zorder=6, label='Average')
            ax.plot([0, 1], avg_vals, 'ko', markersize=12, markerfacecolor='gold',
                   markeredgecolor='black', markeredgewidth=2, zorder=7)
        
        # Styling
        ax.set_xticks([0, 1])
        ax.set_xticklabels([condition1, condition2], fontsize=11)
        ax.set_ylabel('Mutual Information', fontsize=11)
        ax.set_title(f'{band.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-limits starting at 0 (MI is always non-negative)
        # Each band gets its own y-axis scale
        band_max = band_df[metric].max()
        y_margin = band_max * 0.1
        ax.set_ylim(0, band_max + y_margin)
        
        # Add statistics
        mean1 = band_df[band_df['condition'] == condition1][metric].mean()
        mean2 = band_df[band_df['condition'] == condition2][metric].mean()
        
        sig_text = f'Mean diff: {mean2 - mean1:.4f}'
        if stats_df is not None:
            if stats_type == 'aggregated':
                # Use aggregated statistics (one p-value per band)
                band_stats = stats_df[stats_df['band'] == band]
                if len(band_stats) > 0:
                    p_val = band_stats['p_value'].iloc[0]
                    n_obs = band_stats['n_obs'].iloc[0] if 'n_obs' in band_stats.columns else 'N/A'
                    
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = 'n.s.'
                    
                    sig_text = f'Mean diff: {mean2 - mean1:.4f} {stars}\n(p={p_val:.4f}, n={n_obs})'
            else:
                # Use per-channel statistics (original behavior)
                band_stats = stats_df[stats_df['band'] == band]
                if len(band_stats) > 0:
                    n_sig = np.sum(band_stats['p_fdr'] < 0.05)
                    n_total = len(band_stats)
                    min_p = band_stats['p_fdr'].min()
                    
                    if min_p < 0.001:
                        stars = '***'
                    elif min_p < 0.01:
                        stars = '**'
                    elif min_p < 0.05:
                        stars = '*'
                    else:
                        stars = 'n.s.'
                    
                    sig_text = f'Mean diff: {mean2 - mean1:.4f} {stars}\n({n_sig}/{n_total} channels FDR<0.05)'
        
        ax.text(0.5, 0.98, sig_text,
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add legend to first subplot
    if len(bands) > 0:
        axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Remove empty subplots
    for idx in range(len(bands), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Distribution Comparison: {condition1} vs {condition2}\n{metric}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'violin_comparison_{metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file.name}")


def plot_heatmap_all_channels(df, metric, condition1, condition2, output_dir):
    """Heatmap showing MI change for all channels and bands."""
    # Calculate mean MI per channel-band-condition
    means = df.groupby(['channel', 'band', 'condition'])[metric].mean().reset_index()
    
    # Pivot
    pivot = means.pivot_table(index='channel', columns=['band', 'condition'], values=metric)
    
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
               cbar_kws={'label': f'Δ MI ({condition2} - {condition1})'}, 
               linewidths=0.5, ax=ax, annot=False)
    
    # Add directional labels to colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.text(1.5, 1.02, condition2, transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    cbar.ax.text(1.5, -0.02, condition1, transform=cbar.ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold', color='darkblue')
    
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'Change in Mutual Information: {condition1} → {condition2}\n{metric}', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / f'heatmap_changes_{metric}_{condition1}_vs_{condition2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MI changes between conditions')
    parser.add_argument('--metric', type=str, required=True,
                       choices=['mydirect', 'mi_max_lagged'],
                       help='Metric to visualize')
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
    
    # Load data
    results_file = results_dir / 'audio_eeg_mutual_information_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    df = df[df['condition'].isin([args.condition1, args.condition2])].copy()
    
    print("="*60)
    print("MUTUAL INFORMATION CHANGE VISUALIZATIONS")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Comparison: {args.condition1} vs {args.condition2}")
    print(f"Subjects: {sorted(df['subject_id'].unique())}")
    print(f"Channels: {len(df['channel'].unique())}")
    
    # Create output directory
    output_dir = results_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Load statistics - try aggregated first (better for violin plots), fallback to per-channel
    stats_file_agg = results_dir / 'statistics' / f'stats_aggregated_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    stats_file_perchan = results_dir / 'statistics' / f'stats_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    
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
    
    print("\nGenerating plots...")
    
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
