"""
Compare EEG features between RS1 and RS2 resting-state periods.

Shows distribution of features across all channels and subjects with:
- Violin plots showing full distribution for RS1 and RS2
- Individual subject means as points
- Paired connections between conditions per subject
- Average trend line across all participants
- Solid statistics using linear mixed model with crossed random effects + FDR correction
- Modern color scheme and design

Usage:
    python plot_resting_state_features.py --subjects 2 3 4 5 6
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

# Modern color palette for resting states
COLORS = {
    'RS1': '#2E86AB',      # Deep blue
    'RS2': '#A23B72',      # Deep magenta
}

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("husl")


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


def linear_mixed_model(df, feature, conditions):
    """
    Perform statistical test using linear mixed model with crossed random effects.
    
    Data preparation:
    1. Average within each subject × channel × condition (removes temporal pseudoreplication)
    
    Model: feature_mean ~ C(condition) + (1|subject_id) + (1|channel)
    
    This accounts for:
    - Repeated measures across conditions (within-subject design)
    - Subject-level random effects (between-subject variability)
    - Channel-level random effects (between-channel variability)
    - Crossed random effects structure (subjects and channels are both random factors)
    
    Final n = subjects × channels per condition (e.g., 5 subjects × 32 channels = 160)
    MATCHES the correlation analysis approach.
    
    Returns LR-statistic and p-value.
    """
    import statsmodels.formula.api as smf
    from scipy import stats as scipy_stats
    
    # Get column names (handle both relative and absolute features)
    feature_col = feature
    if feature_col not in df.columns:
        print(f"WARNING: Feature '{feature}' not found in dataframe")
        return np.nan, np.nan
    
    # Step 1: Average within subject × channel × condition to remove temporal pseudoreplication
    df_avg = df.groupby(['subject_id', 'channel', 'condition'])[feature_col].mean().reset_index()
    df_avg.columns = ['subject_id', 'channel', 'condition', 'feature_mean']
    
    # Filter to only the two conditions we're comparing
    df_avg = df_avg[df_avg['condition'].isin(conditions)]
    
    # Convert to categorical
    df_avg['condition'] = pd.Categorical(df_avg['condition'], categories=conditions)
    
    n_subjects = df_avg['subject_id'].nunique()
    n_channels = df_avg['channel'].nunique()
    n_per_condition = len(df_avg[df_avg['condition'] == conditions[0]])
    
    print(f"  {feature}: {n_subjects} subj × {n_channels} chan = {n_per_condition} obs per condition")
    
    # Check for NaN values
    if df_avg['feature_mean'].isna().any():
        print(f"  WARNING: {df_avg['feature_mean'].isna().sum()} NaN values found, dropping...")
        df_avg = df_avg.dropna(subset=['feature_mean'])
    
    # Check if we have enough variance to fit model
    if df_avg['feature_mean'].std() < 1e-10:
        print(f"  WARNING: No variance in {feature}, skipping")
        return np.nan, np.nan
    
    try:
        # Null model (no condition effect)
        formula_null = "feature_mean ~ 1"
        model_null = smf.mixedlm(
            formula_null, 
            df_avg, 
            groups=df_avg["subject_id"],
            re_formula="1",
            vc_formula={"channel": "0 + C(channel)"}
        )
        result_null = model_null.fit(method='powell', maxiter=1000, reml=False)
        
        # Full model (with condition effect)
        formula_full = "feature_mean ~ C(condition)"
        model_full = smf.mixedlm(
            formula_full,
            df_avg,
            groups=df_avg["subject_id"],
            re_formula="1",
            vc_formula={"channel": "0 + C(channel)"}
        )
        result_full = model_full.fit(method='powell', maxiter=1000, reml=False)
        
        # Likelihood ratio test
        lr_stat = -2 * (result_null.llf - result_full.llf)
        p_value = scipy_stats.chi2.sf(lr_stat, df=1)
        
        return lr_stat, p_value
        
    except Exception as e:
        print(f"  ERROR fitting model for {feature}: {e}")
        return np.nan, np.nan


def post_hoc_pairwise_wilcoxon(df, feature, conditions):
    """
    Post-hoc pairwise comparisons using Wilcoxon signed-rank test.
    For two conditions, we only need one comparison.
    
    Uses subject-averaged data (averaged across channels and time).
    """
    # Get column names
    feature_col = feature
    if feature_col not in df.columns:
        return None, None
    
    # Average within subject × condition (across channels and time)
    df_subj_avg = df.groupby(['subject_id', 'condition'])[feature_col].mean().reset_index()
    
    # Get data for each condition
    data_rs1 = df_subj_avg[df_subj_avg['condition'] == conditions[0]][feature_col].values
    data_rs2 = df_subj_avg[df_subj_avg['condition'] == conditions[1]][feature_col].values
    
    # Ensure we have paired data
    if len(data_rs1) != len(data_rs2):
        print(f"  WARNING: Unequal sample sizes for {feature}")
        return None, None
    
    # Wilcoxon signed-rank test (paired)
    try:
        stat, p = stats.wilcoxon(data_rs1, data_rs2, alternative='two-sided')
        return stat, p
    except Exception as e:
        print(f"  WARNING: Wilcoxon test failed for {feature}: {e}")
        return None, None


def plot_violin_with_subjects(df, feature, conditions, omnibus_p, posthoc_results, output_file):
    """
    Create violin plot comparing RS1 vs RS2 for a single feature.
    
    Shows:
    - Full distribution as violin
    - Subject means as connected points
    - Overall mean as horizontal line
    - Statistical results in title
    """
    feature_col = feature
    
    # Prepare data for plotting (subject × channel averages)
    df_avg = df.groupby(['subject_id', 'channel', 'condition'])[feature_col].mean().reset_index()
    df_avg.columns = ['subject_id', 'channel', 'condition', 'feature_mean']
    
    # Also get subject means (averaged across channels) for connecting lines
    df_subj_mean = df.groupby(['subject_id', 'condition'])[feature_col].mean().reset_index()
    df_subj_mean.columns = ['subject_id', 'condition', 'subject_mean']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Violin plot with all subject × channel data
    parts = ax.violinplot(
        [df_avg[df_avg['condition'] == cond]['feature_mean'].values for cond in conditions],
        positions=[0, 1],
        widths=0.6,
        showmeans=False,
        showextrema=False
    )
    
    # Color violins
    for i, (pc, cond) in enumerate(zip(parts['bodies'], conditions)):
        pc.set_facecolor(COLORS[cond])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Plot subject means as connected points
    for subj in df_subj_mean['subject_id'].unique():
        subj_data = df_subj_mean[df_subj_mean['subject_id'] == subj]
        x_vals = [0, 1]
        y_vals = [subj_data[subj_data['condition'] == cond]['subject_mean'].values[0] 
                  for cond in conditions]
        ax.plot(x_vals, y_vals, 'o-', color='gray', alpha=0.3, linewidth=1, markersize=5)
    
    # Add mean line (average across all subjects × channels)
    means = [df_avg[df_avg['condition'] == cond]['feature_mean'].mean() for cond in conditions]
    ax.plot([0, 1], means, 'D-', color='gold', linewidth=3, markersize=10, 
            label='Mean', zorder=10, markeredgecolor='black', markeredgewidth=1)
    
    # Labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(conditions, fontsize=12, fontweight='bold')
    ax.set_ylabel(feature, fontsize=12, fontweight='bold')
    
    # Statistical results in title
    n_subjects = df['subject_id'].nunique()
    n_channels = df['channel'].nunique()
    n_per_condition = n_subjects * n_channels
    
    title = f'{feature}\n'
    title += f'({n_subjects} subj × {n_channels} chan = {n_per_condition} obs)\n'
    
    if np.isfinite(omnibus_p):
        if omnibus_p < 0.001:
            title += f'LMM: p < 0.001***'
        elif omnibus_p < 0.01:
            title += f'LMM: p = {omnibus_p:.3f}**'
        elif omnibus_p < 0.05:
            title += f'LMM: p = {omnibus_p:.3f}*'
        else:
            title += f'LMM: p = {omnibus_p:.3f} (ns)'
    else:
        title += 'LMM: p = NaN'
    
    # Add post-hoc result if available
    if posthoc_results is not None:
        stat, p = posthoc_results
        if p is not None and np.isfinite(p):
            if p < 0.001:
                title += f'\nWilcoxon: p < 0.001***'
            elif p < 0.01:
                title += f'\nWilcoxon: p = {p:.3f}**'
            elif p < 0.05:
                title += f'\nWilcoxon: p = {p:.3f}*'
            else:
                title += f'\nWilcoxon: p = {p:.3f} (ns)'
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare EEG features between RS1 and RS2')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                      help='Subject IDs to include (default: 2 3 4 5 6)')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                      help='Root data directory')
    parser.add_argument('--output-dir', type=Path, 
                      default=Path('../results/resting_state_features'),
                      help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Conditions to compare
    conditions = ['RS1', 'RS2']
    
    # Load data
    df = load_all_data(args.subjects, conditions, args.data_dir)
    
    # Feature groups
    psd_features = [
        'delta_abs', 'theta_abs', 'alpha_abs', 
        'low_beta_abs', 'high_beta_abs', 'gamma1_abs',
        'delta_rel', 'theta_rel', 'alpha_rel',
        'low_beta_rel', 'high_beta_rel', 'gamma1_rel'
    ]
    
    entropy_features = [
        'lzc', 'perm_entropy', 'spectral_entropy',
        'svd_entropy', 'sample_entropy'
    ]
    
    all_features = psd_features + entropy_features
    
    # Check which features exist in the data
    available_features = [f for f in all_features if f in df.columns]
    print(f"\nAvailable features: {len(available_features)}")
    print(f"PSD: {[f for f in psd_features if f in available_features]}")
    print(f"Entropy: {[f for f in entropy_features if f in available_features]}")
    
    if not available_features:
        print("ERROR: No features found in data!")
        return
    
    # Run statistical tests
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: RS1 vs RS2")
    print("="*80)
    print("\nUsing linear mixed model with crossed random effects:")
    print("  Model: feature_mean ~ C(condition) + (1|subject_id) + (1|channel)")
    print("  - Accounts for repeated measures (within-subject design)")
    print("  - Accounts for subject variability (random effect)")
    print("  - Accounts for channel variability (random effect)")
    print("  - n = subjects × channels per condition")
    
    results = []
    
    for feature in available_features:
        print(f"\n{feature}:")
        
        # Omnibus test (LMM with LRT)
        lr_stat, omnibus_p = linear_mixed_model(df, feature, conditions)
        
        # Post-hoc pairwise comparison
        posthoc_stat, posthoc_p = post_hoc_pairwise_wilcoxon(df, feature, conditions)
        
        results.append({
            'feature': feature,
            'lr_stat': lr_stat,
            'omnibus_p': omnibus_p,
            'posthoc_stat': posthoc_stat,
            'posthoc_p': posthoc_p
        })
        
        # Generate plot
        output_file = args.output_dir / f'{feature}_RS1_vs_RS2.png'
        plot_violin_with_subjects(df, feature, conditions, omnibus_p, 
                                 (posthoc_stat, posthoc_p), output_file)
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # FDR correction on omnibus p-values
    valid_mask = df_results['omnibus_p'].notna()
    if valid_mask.sum() > 0:
        _, pvals_corrected, _, _ = multipletests(
            df_results.loc[valid_mask, 'omnibus_p'], 
            method='fdr_bh'
        )
        df_results.loc[valid_mask, 'omnibus_p_fdr'] = pvals_corrected
    else:
        df_results['omnibus_p_fdr'] = np.nan
    
    # FDR correction on post-hoc p-values
    valid_mask_ph = df_results['posthoc_p'].notna()
    if valid_mask_ph.sum() > 0:
        _, pvals_corrected_ph, _, _ = multipletests(
            df_results.loc[valid_mask_ph, 'posthoc_p'],
            method='fdr_bh'
        )
        df_results.loc[valid_mask_ph, 'posthoc_p_fdr'] = pvals_corrected_ph
    else:
        df_results['posthoc_p_fdr'] = np.nan
    
    # Sort by omnibus p-value
    df_results = df_results.sort_values('omnibus_p')
    
    # Save results
    results_file = args.output_dir / 'statistical_results_RS1_vs_RS2.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\nSaved statistical results to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS (sorted by p-value)")
    print("="*80)
    print("\nSignificant features (FDR-corrected p < 0.05):")
    sig_features = df_results[df_results['omnibus_p_fdr'] < 0.05]
    if len(sig_features) > 0:
        print(sig_features[['feature', 'omnibus_p', 'omnibus_p_fdr', 'posthoc_p', 'posthoc_p_fdr']].to_string(index=False))
    else:
        print("  None")
    
    print("\nTrending features (0.05 <= FDR-corrected p < 0.10):")
    trend_features = df_results[(df_results['omnibus_p_fdr'] >= 0.05) & 
                                (df_results['omnibus_p_fdr'] < 0.10)]
    if len(trend_features) > 0:
        print(trend_features[['feature', 'omnibus_p', 'omnibus_p_fdr', 'posthoc_p', 'posthoc_p_fdr']].to_string(index=False))
    else:
        print("  None")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
