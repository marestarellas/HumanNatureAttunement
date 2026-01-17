"""
Create topomaps showing percentage difference between RS2 and RS1 for EEG features.

Generates two consolidated figures:
1. PSD features (absolute and relative power across frequency bands)
2. Complexity/entropy features

Usage:
    python plot_resting_state_topomaps.py --subjects 2 3 4 5 6
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mne
import mne.channels
import warnings
warnings.filterwarnings('ignore')

# Standard 32 channel names for MNE 10-20 system
STANDARD_32_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'O1', 'Oz', 'O2', 'PO10',
    'AF7', 'AF8'
]


def load_subject_data(subject_id, condition, data_dir):
    """Load feature data for one subject and condition."""
    subject_folder = f'sub-{subject_id:02d}'
    feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
    
    if not feature_file.exists():
        print(f"  WARNING: File not found: {feature_file}")
        return None
    
    df = pd.read_csv(feature_file)
    df['subject_id'] = subject_id
    df['condition'] = condition
    
    return df


def load_all_data(subjects, conditions, data_dir):
    """Load and combine data from multiple subjects and conditions."""
    print(f"\nLoading data for {len(subjects)} subjects, conditions: {conditions}")
    
    dfs = []
    for subject_id in subjects:
        for condition in conditions:
            df = load_subject_data(subject_id, condition, data_dir)
            if df is not None:
                print(f"  Subject {subject_id:02d}, {condition}: {len(df)} observations")
                dfs.append(df)
    
    if not dfs:
        raise ValueError("No data loaded!")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(df_combined)} observations")
    print(f"Channels: {len(df_combined['channel'].unique())}")
    
    return df_combined


def compute_percentage_difference(df, features):
    """
    Compute percentage difference: ((RS2 - RS1) / RS1) * 100
    Averages across subjects and time windows for each channel.
    Uses robust calculation to handle small denominators.
    """
    # Average within subject × channel × condition (across time)
    df_avg = df.groupby(['subject_id', 'channel', 'condition'])[features].mean().reset_index()
    
    # Separate RS1 and RS2
    df_rs1 = df_avg[df_avg['condition'] == 'RS1'].set_index(['subject_id', 'channel'])
    df_rs2 = df_avg[df_avg['condition'] == 'RS2'].set_index(['subject_id', 'channel'])
    
    # Compute percentage difference for each subject
    pct_diff_list = []
    for subj in df_rs1.index.get_level_values('subject_id').unique():
        rs1_subj = df_rs1.loc[subj][features]
        rs2_subj = df_rs2.loc[subj][features]
        
        # Standard percentage difference: ((RS2 - RS1) / RS1) * 100
        # Add small epsilon only to prevent division by zero, not using abs()
        # This preserves the sign of the baseline
        pct_diff = ((rs2_subj - rs1_subj) / (rs1_subj + 1e-10)) * 100
        
        # Cap extreme values at ±500% to prevent outliers from dominating
        pct_diff = pct_diff.clip(-500, 500)
        
        pct_diff['subject_id'] = subj
        pct_diff['channel'] = rs1_subj.index
        pct_diff_list.append(pct_diff)
    
    df_pct_diff = pd.concat(pct_diff_list, ignore_index=True)
    
    # Print diagnostic info
    print("\nPercentage difference statistics:")
    for feat in features:
        vals = df_pct_diff[feat].values
        print(f"  {feat}: mean={np.mean(vals):.1f}%, median={np.median(vals):.1f}%, "
              f"range=[{np.min(vals):.1f}, {np.max(vals):.1f}]%")
    
    # Average across subjects for each channel
    df_channel_avg = df_pct_diff.groupby('channel')[features].mean().reset_index()
    
    return df_channel_avg


def plot_topomap_grid(df_channel_avg, features, feature_labels, title, output_file, n_cols=3):
    """
    Create a grid of topomaps for multiple features using MNE standard approach.
    
    Parameters
    ----------
    df_channel_avg : pd.DataFrame
        Dataframe with columns: channel, feature1, feature2, ...
    features : list
        List of feature column names
    feature_labels : list
        List of labels for each feature (for subplot titles)
    title : str
        Main figure title
    output_file : Path
        Output file path
    n_cols : int
        Number of columns in the grid
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Get number of channels from data
    n_channels = len(df_channel_avg)
    mne_ch_names = STANDARD_32_CHANNELS[:n_channels]
    
    # Create MNE Info object
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    # Collect all values to determine global colormap range
    # Use percentile-based limits to avoid outliers dominating the scale
    all_values = []
    for feat in features:
        all_values.extend(df_channel_avg[feat].values)
    
    # Use 95th percentile of absolute values to set color scale
    # This makes the visualization more informative by not letting extreme outliers dominate
    if len(all_values) > 0:
        abs_values = np.abs(all_values)
        global_vmax = np.percentile(abs_values, 95)
        # Ensure non-zero
        if global_vmax < 1e-6:
            global_vmax = np.max(abs_values) if np.max(abs_values) > 0 else 1e-6
    else:
        global_vmax = 1e-6
    
    print(f"\nPlotting {n_features} features with {n_channels} channels")
    print(f"  Color scale range: ±{global_vmax:.1f}% (95th percentile)")
    print(f"  Actual data range: {np.min(all_values):.1f}% to {np.max(all_values):.1f}%")
    
    # Plot each feature
    for idx, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[idx]
        
        # Get values for this feature
        values = df_channel_avg[feat].values
        
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
        
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    # Add single colorbar for the whole figure
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('% Change (RS2 vs RS1)', fontsize=11, rotation=270, labelpad=20)
    
    # Add directional labels at colorbar extremes
    cbar.ax.text(1.5, 1.02, 'RS2 >', transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    cbar.ax.text(1.5, -0.02, '< RS1', transform=cbar.ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold', color='darkblue')
    
    # Overall title
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create topomaps of RS2 vs RS1 percentage differences')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                      help='Subject IDs to include (default: 2 3 4 5 6)')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                      help='Root data directory')
    parser.add_argument('--output-dir', type=Path, 
                      default=Path('../results/resting_state_topomaps'),
                      help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Conditions to compare
    conditions = ['RS1', 'RS2']
    
    # Load data
    df = load_all_data(args.subjects, conditions, args.data_dir)
    
    # Define feature groups
    psd_features_abs = ['delta_abs', 'theta_abs', 'alpha_abs', 
                        'low_beta_abs', 'high_beta_abs', 'gamma1_abs']
    psd_features_rel = ['delta_rel', 'theta_rel', 'alpha_rel',
                        'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
    
    entropy_features = ['lzc', 'perm_entropy', 'spectral_entropy',
                       'svd_entropy', 'sample_entropy']
    
    # Check which features exist
    psd_abs_available = [f for f in psd_features_abs if f in df.columns]
    psd_rel_available = [f for f in psd_features_rel if f in df.columns]
    entropy_available = [f for f in entropy_features if f in df.columns]
    
    print(f"\nAvailable features:")
    print(f"  PSD absolute: {psd_abs_available}")
    print(f"  PSD relative: {psd_rel_available}")
    print(f"  Entropy: {entropy_available}")
    
    # Frequency ranges for labels
    freq_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma1': (30, 50)
    }
    
    # ===== PSD FEATURES =====
    if psd_abs_available or psd_rel_available:
        print("\n" + "="*80)
        print("COMPUTING PSD PERCENTAGE DIFFERENCES")
        print("="*80)
        
        # Combine absolute and relative
        all_psd_features = psd_abs_available + psd_rel_available
        df_pct_psd = compute_percentage_difference(df, all_psd_features)
        
        # Create labels
        psd_labels = []
        for feat in psd_abs_available:
            band_name = feat.replace('_abs', '')
            freq_range = freq_ranges.get(band_name, (0, 0))
            psd_labels.append(f'{band_name.replace("_", "-")}\n{freq_range[0]}-{freq_range[1]} Hz\n(absolute)')
        
        for feat in psd_rel_available:
            band_name = feat.replace('_rel', '')
            freq_range = freq_ranges.get(band_name, (0, 0))
            psd_labels.append(f'{band_name.replace("_", "-")}\n{freq_range[0]}-{freq_range[1]} Hz\n(relative)')
        
        # Plot
        output_file = args.output_dir / 'topomap_psd_RS2_vs_RS1.png'
        plot_topomap_grid(
            df_pct_psd,
            all_psd_features,
            psd_labels,
            'PSD Features: % Change (RS2 vs RS1)',
            output_file,
            n_cols=3
        )
    
    # ===== ENTROPY/COMPLEXITY FEATURES =====
    if entropy_available:
        print("\n" + "="*80)
        print("COMPUTING COMPLEXITY/ENTROPY PERCENTAGE DIFFERENCES")
        print("="*80)
        
        df_pct_entropy = compute_percentage_difference(df, entropy_available)
        
        # Create labels
        entropy_labels = []
        label_map = {
            'lzc': 'Lempel-Ziv\nComplexity',
            'perm_entropy': 'Permutation\nEntropy',
            'spectral_entropy': 'Spectral\nEntropy',
            'svd_entropy': 'SVD\nEntropy',
            'sample_entropy': 'Sample\nEntropy'
        }
        for feat in entropy_available:
            entropy_labels.append(label_map.get(feat, feat))
        
        # Plot
        output_file = args.output_dir / 'topomap_complexity_RS2_vs_RS1.png'
        plot_topomap_grid(
            df_pct_entropy,
            entropy_available,
            entropy_labels,
            'Complexity Features: % Change (RS2 vs RS1)',
            output_file,
            n_cols=3
        )
    
    print("\n" + "="*80)
    print(f"All topomaps saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
