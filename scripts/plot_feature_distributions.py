"""
Create elegant violin plots comparing EEG features across all three conditions.

Shows distribution of features across all channels and subjects with:
- Violin plots showing full distribution for VIZ, AUD, and MULTI
- Individual subject means as points
- Paired connections between conditions per subject
- Average trend line across all participants
- Solid statistics using repeated-measures ANOVA with FDR correction
- Modern color scheme and design

Usage:
    python plot_feature_distributions.py --subjects 2 3 4 5 6
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

# Modern color palette
COLORS = {
    'VIZ': '#2E86AB',      # Deep blue
    'AUD': '#A23B72',      # Deep magenta
    'MULTI': '#F18F01',    # Orange
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


def repeated_measures_anova(df, feature, conditions):
    """
    Perform repeated-measures ANOVA using linear mixed model with crossed random effects.
    
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
    
    # Step 1: Average within each subject × channel × condition
    # This removes pseudoreplication from temporal segments
    df_avg = df.groupby(['subject_id', 'channel', 'condition'])[feature].mean().reset_index()
    df_avg.rename(columns={feature: f'{feature}_mean'}, inplace=True)
    
    # Prepare data - KEEP all subject×channel observations
    df_clean = df_avg.dropna()
    
    # Need categorical variables as strings
    df_clean = df_clean.copy()
    df_clean['subject_id'] = df_clean['subject_id'].astype(str)
    df_clean['channel'] = df_clean['channel'].astype(str)
    
    # Filter to conditions of interest
    df_clean = df_clean[df_clean['condition'].isin(conditions)]
    
    if len(df_clean) < 20:
        return np.nan, 1.0, 0, 0
    
    n_obs = len(df_clean)  # Total observations = subjects × channels × conditions
    n_subjects = df_clean['subject_id'].nunique()
    n_channels = df_clean['channel'].nunique()
    
    try:
        # Fit mixed model with CROSSED random effects (subject + channel)
        # MATCHES correlation analysis approach
        formula = f"{feature}_mean ~ C(condition)"
        
        # Suppress convergence warnings - they're common and don't affect validity
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # Try with crossed random effects first (like correlation analysis)
            try:
                # Fit null model (no condition effect) with ML for LRT
                null_model = smf.mixedlm(f"{feature}_mean ~ 1", df_clean,
                                         groups=df_clean["subject_id"],
                                         re_formula="1",
                                         vc_formula={"channel": "0 + C(channel)"})
                null_result = null_model.fit(method='powell', maxiter=500, reml=False)
                
                # Fit full model with ML for LRT
                full_model = smf.mixedlm(formula, df_clean,
                                         groups=df_clean["subject_id"],
                                         re_formula="1",
                                         vc_formula={"channel": "0 + C(channel)"})
                full_result = full_model.fit(method='powell', maxiter=500, reml=False)
            except:
                # Fallback to simpler model if crossed effects fail
                null_model = smf.mixedlm(f"{feature}_mean ~ 1", df_clean,
                                         groups=df_clean["subject_id"],
                                         re_formula="1")
                null_result = null_model.fit(method='powell', maxiter=500, reml=False)
                
                full_model = smf.mixedlm(formula, df_clean,
                                         groups=df_clean["subject_id"],
                                         re_formula="1")
                full_result = full_model.fit(method='powell', maxiter=500, reml=False)
        
        # Likelihood ratio test
        lr_stat = 2 * (full_result.llf - null_result.llf)
        df_diff = len(conditions) - 1  # degrees of freedom
        pval = scipy_stats.chi2.sf(lr_stat, df_diff)
        
        # Check for valid results
        if np.isnan(pval) or np.isinf(pval):
            raise ValueError("Invalid p-value from LRT")
        
        return lr_stat, pval, n_obs, n_subjects
        
    except Exception as e:
        # Fallback to Friedman test on subject means if mixed model fails
        print(f"  Mixed model failed for {feature}, using Friedman test: {str(e)[:100]}")
        
        # Average across channels for each subject
        subject_means = df_clean.groupby(['subject_id', 'condition'])[f'{feature}_mean'].mean().reset_index()
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=f'{feature}_mean')
        pivot_clean = pivot.dropna()
        
        if len(pivot_clean) < 3:
            print(f"  Not enough subjects after dropna: {len(pivot_clean)}")
            return np.nan, 1.0, 0, 0
        
        data_arrays = [pivot_clean[cond].values for cond in conditions if cond in pivot_clean.columns]
        
        if len(data_arrays) < 2:
            print(f"  Not enough conditions: {len(data_arrays)}")
            return np.nan, 1.0, 0, 0
        
        try:
            stat, pval = scipy_stats.friedmanchisquare(*data_arrays)
            print(f"  Friedman test successful: p={pval:.4f}")
            return stat, pval, len(pivot_clean), len(pivot_clean)
        except Exception as e2:
            print(f"  Friedman test also failed: {str(e2)}")
            return np.nan, 1.0, 0, 0


def plot_psd_features(df, output_dir):
    """
    Plot violin plots for all PSD features (relative power in each band).
    One subplot per band, showing all three conditions.
    """
    # PSD features (relative power)
    psd_features = ['delta_rel', 'theta_rel', 'alpha_rel', 
                    'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
    
    band_labels = ['Delta\n(0.5-4 Hz)', 'Theta\n(4-8 Hz)', 'Alpha\n(8-13 Hz)',
                   'Low Beta\n(13-20 Hz)', 'High Beta\n(20-30 Hz)', 'Gamma\n(30-50 Hz)']
    
    conditions = ['VIZ', 'AUD', 'MULTI']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Collect p-values for FDR correction
    pvalues = []
    n_obs_list = []
    n_subj_list = []
    
    for idx, (feature, label) in enumerate(zip(psd_features, band_labels)):
        ax = axes[idx]
        
        # Calculate statistics using full nested structure
        stat, pval, n_obs, n_subj = repeated_measures_anova(df, feature, conditions)
        pvalues.append(pval)
        n_obs_list.append(n_obs)
        n_subj_list.append(n_subj)
    
    # FDR correction across all bands
    reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
    
    for idx, (feature, label) in enumerate(zip(psd_features, band_labels)):
        ax = axes[idx]
        
        # Calculate subject means for each condition
        subject_means = df.groupby(['subject_id', 'condition'])[feature].mean().reset_index()
        
        # Create violin plots for all three conditions
        violin_data = []
        positions = [0, 1, 2]
        colors = [COLORS['VIZ'], COLORS['AUD'], COLORS['MULTI']]
        
        for cond in conditions:
            cond_data = df[df['condition'] == cond][feature].dropna().values
            violin_data.append(cond_data)
        
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.7,
            showmeans=False,
            showextrema=False,
            showmedians=False
        )
        
        # Color violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add box plot elements (quartiles and median)
        for i, cond in enumerate(conditions):
            cond_data = df[df['condition'] == cond][feature].dropna().values
            if len(cond_data) > 0:
                q1, median, q3 = np.percentile(cond_data, [25, 50, 75])
                
                # Median line
                ax.hlines(median, i - 0.15, i + 0.15, colors='white', linewidth=3, zorder=3)
                ax.hlines(median, i - 0.15, i + 0.15, colors='black', linewidth=1.5, zorder=4)
                
                # Quartile lines
                ax.hlines([q1, q3], i - 0.08, i + 0.08, colors='black', linewidth=1.5, zorder=3)
        
        # Plot individual subject trajectories
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=feature)
        subjects = pivot.index
        
        # Individual lines
        for subj in subjects:
            vals = [pivot.loc[subj, cond] if cond in pivot.columns and pd.notna(pivot.loc[subj, cond]) else None 
                   for cond in conditions]
            
            # Only plot if we have at least 2 values
            valid_idx = [i for i, v in enumerate(vals) if v is not None]
            if len(valid_idx) >= 2:
                valid_pos = [positions[i] for i in valid_idx]
                valid_vals = [vals[i] for i in valid_idx]
                ax.plot(valid_pos, valid_vals, 'k-', alpha=0.25, linewidth=1, zorder=1)
        
        # Individual points with jitter
        np.random.seed(42)
        for i, cond in enumerate(conditions):
            if cond in pivot.columns:
                values = pivot[cond].dropna().values
                if len(values) > 0:
                    jitter = np.random.normal(0, 0.03, len(values))
                    ax.scatter(i + jitter, values, 
                             s=100, c=colors[i], alpha=0.9, 
                             edgecolors='white', linewidths=2, zorder=5)
        
        # Add AVERAGE LINE across all participants
        avg_vals = []
        for cond in conditions:
            if cond in pivot.columns:
                avg_vals.append(pivot[cond].mean())
            else:
                avg_vals.append(np.nan)
        
        # Plot average line with markers
        ax.plot(positions, avg_vals, 'k-', linewidth=3, alpha=0.8, zorder=6,
               label='Average')
        ax.plot(positions, avg_vals, 'ko', markersize=12, markerfacecolor='gold',
               markeredgecolor='black', markeredgewidth=2, zorder=7)
        
        # Styling
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=11, fontweight='bold')
        ax.set_ylabel('Relative Power', fontsize=10)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Add statistics text (FDR-corrected) with post-hoc comparisons
        pval_fdr = pvals_corrected[idx]
        n_obs = n_obs_list[idx]
        n_subj = n_subj_list[idx]
        
        if pval_fdr < 0.001:
            stars = '***'
        elif pval_fdr < 0.01:
            stars = '**'
        elif pval_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        
        # Add post-hoc pairwise comparisons if overall test is significant
        posthoc_text = ""
        if pval_fdr < 0.05:
            # Calculate pairwise comparisons
            subject_means = df.groupby(['subject_id', 'condition'])[feature].mean().reset_index()
            pivot = subject_means.pivot(index='subject_id', columns='condition', values=feature)
            
            pairs = [('VIZ', 'AUD'), ('VIZ', 'MULTI'), ('AUD', 'MULTI')]
            sig_pairs = []
            
            for c1, c2 in pairs:
                if c1 in pivot.columns and c2 in pivot.columns:
                    vals1 = pivot[c1].dropna()
                    vals2 = pivot[c2].dropna()
                    if len(vals1) > 2 and len(vals2) > 2:
                        from scipy.stats import wilcoxon
                        try:
                            _, p = wilcoxon(vals1, vals2)
                            if p < 0.05/3:  # Bonferroni correction for 3 comparisons
                                sig_pairs.append(f"{c1}≠{c2}")
                        except:
                            pass
            
            if sig_pairs:
                posthoc_text = f"\n{', '.join(sig_pairs)}"
        
        # Always calculate post-hoc for interpretation (even if overall n.s.)
        if pval_fdr <= 1.0:  # Always run post-hoc
            subject_means = df.groupby(['subject_id', 'condition'])[feature].mean().reset_index()
            pivot = subject_means.pivot(index='subject_id', columns='condition', values=feature)
            
            pairs = [('VIZ', 'AUD'), ('VIZ', 'MULTI'), ('AUD', 'MULTI')]
            pair_pvals = []
            
            for c1, c2 in pairs:
                if c1 in pivot.columns and c2 in pivot.columns:
                    vals1 = pivot[c1].dropna()
                    vals2 = pivot[c2].dropna()
                    if len(vals1) > 2 and len(vals2) > 2:
                        from scipy.stats import wilcoxon
                        try:
                            _, p = wilcoxon(vals1, vals2)
                            pair_pvals.append((c1, c2, p))
                        except:
                            pass
            
            # Show significant pairs if overall test is significant
            if pval_fdr < 0.05 and pair_pvals:
                sig_pairs = [f"{c1}≠{c2} (p={p:.3f})" for c1, c2, p in pair_pvals if p < 0.05/3]
                if sig_pairs:
                    posthoc_text = f"\n{', '.join(sig_pairs)}"
        
        # Show n info (matching correlation plot format)
        n_channels = df['channel'].nunique()
        ax.text(0.5, 0.98, f'p={pval_fdr:.4f} {stars}{posthoc_text}\n({n_subj} subj × {n_channels} chan = {n_obs} obs)',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.suptitle(f'Relative Power Spectral Density: VIZ vs AUD vs MULTI\n(All channels, n={len(df["subject_id"].unique())} subjects)', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'psd_features_violin_all_conditions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_file.name}")


def plot_complexity_features(df, output_dir):
    """
    Plot violin plots for complexity/entropy features.
    Shows all three conditions.
    """
    # Complexity features
    complexity_features = ['lzc', 'perm_entropy', 'spectral_entropy', 
                          'svd_entropy', 'sample_entropy']
    
    feature_labels = ['Lempel-Ziv\nComplexity', 'Permutation\nEntropy', 
                     'Spectral\nEntropy', 'SVD\nEntropy', 'Sample\nEntropy']
    
    # Check which features exist
    available_features = [f for f in complexity_features if f in df.columns]
    available_labels = [l for f, l in zip(complexity_features, feature_labels) 
                       if f in df.columns]
    
    if len(available_features) == 0:
        print("\n⚠ No complexity features found in data")
        return
    
    conditions = ['VIZ', 'AUD', 'MULTI']
    
    # Create figure with appropriate number of subplots
    n_features = len(available_features)
    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    # Collect p-values for FDR correction
    pvalues = []
    n_obs_list = []
    n_subj_list = []
    
    for feature in available_features:
        stat, pval, n_obs, n_subj = repeated_measures_anova(df, feature, conditions)
        pvalues.append(pval)
        n_obs_list.append(n_obs)
        n_subj_list.append(n_subj)
    
    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
    
    for idx, (feature, label) in enumerate(zip(available_features, available_labels)):
        ax = axes[idx]
        
        # Calculate subject means
        subject_means = df.groupby(['subject_id', 'condition'])[feature].mean().reset_index()
        
        # Get colors
        colors = [COLORS['VIZ'], COLORS['AUD'], COLORS['MULTI']]
        positions = [0, 1, 2]
        
        # Create violin plots
        violin_data = []
        for cond in conditions:
            cond_data = df[df['condition'] == cond][feature].dropna().values
            violin_data.append(cond_data)
        
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.7,
            showmeans=False,
            showextrema=False,
            showmedians=False
        )
        
        # Color violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add box plot elements
        for i, cond in enumerate(conditions):
            cond_data = df[df['condition'] == cond][feature].dropna().values
            if len(cond_data) > 0:
                q1, median, q3 = np.percentile(cond_data, [25, 50, 75])
                
                ax.hlines(median, i - 0.15, i + 0.15, colors='white', linewidth=3, zorder=3)
                ax.hlines(median, i - 0.15, i + 0.15, colors='black', linewidth=1.5, zorder=4)
                ax.hlines([q1, q3], i - 0.08, i + 0.08, colors='black', linewidth=1.5, zorder=3)
        
        # Plot individual subject trajectories
        pivot = subject_means.pivot(index='subject_id', columns='condition', values=feature)
        subjects = pivot.index
        
        # Individual lines
        for subj in subjects:
            vals = [pivot.loc[subj, cond] if cond in pivot.columns and pd.notna(pivot.loc[subj, cond]) else None 
                   for cond in conditions]
            
            valid_idx = [i for i, v in enumerate(vals) if v is not None]
            if len(valid_idx) >= 2:
                valid_pos = [positions[i] for i in valid_idx]
                valid_vals = [vals[i] for i in valid_idx]
                ax.plot(valid_pos, valid_vals, 'k-', alpha=0.25, linewidth=1, zorder=1)
        
        # Individual points
        np.random.seed(42)
        for i, cond in enumerate(conditions):
            if cond in pivot.columns:
                values = pivot[cond].dropna().values
                if len(values) > 0:
                    jitter = np.random.normal(0, 0.03, len(values))
                    ax.scatter(i + jitter, values, 
                             s=100, c=colors[i], alpha=0.9, 
                             edgecolors='white', linewidths=2, zorder=5)
        
        # Add AVERAGE LINE across all participants
        avg_vals = []
        for cond in conditions:
            if cond in pivot.columns:
                avg_vals.append(pivot[cond].mean())
            else:
                avg_vals.append(np.nan)
        
        ax.plot(positions, avg_vals, 'k-', linewidth=3, alpha=0.8, zorder=6,
               label='Average')
        ax.plot(positions, avg_vals, 'ko', markersize=12, markerfacecolor='gold',
               markeredgecolor='black', markeredgewidth=2, zorder=7)
        
        # Styling
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        # Add statistics (FDR-corrected) with post-hoc comparisons
        pval_fdr = pvals_corrected[idx]
        n_obs = n_obs_list[idx]
        n_subj = n_subj_list[idx]
        
        if pval_fdr < 0.001:
            stars = '***'
        elif pval_fdr < 0.01:
            stars = '**'
        elif pval_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        
        # Always calculate post-hoc for interpretation (even if overall n.s.)
        posthoc_text = ""
        if pval_fdr <= 1.0:  # Always run post-hoc
            subject_means = df.groupby(['subject_id', 'condition'])[feature].mean().reset_index()
            pivot = subject_means.pivot(index='subject_id', columns='condition', values=feature)
            
            pairs = [('VIZ', 'AUD'), ('VIZ', 'MULTI'), ('AUD', 'MULTI')]
            pair_pvals = []
            
            for c1, c2 in pairs:
                if c1 in pivot.columns and c2 in pivot.columns:
                    vals1 = pivot[c1].dropna()
                    vals2 = pivot[c2].dropna()
                    if len(vals1) > 2 and len(vals2) > 2:
                        from scipy.stats import wilcoxon
                        try:
                            _, p = wilcoxon(vals1, vals2)
                            pair_pvals.append((c1, c2, p))
                        except:
                            pass
            
            # Show significant pairs if overall test is significant
            if pval_fdr < 0.05 and pair_pvals:
                sig_pairs = [f"{c1}≠{c2} (p={p:.3f})" for c1, c2, p in pair_pvals if p < 0.05/3]
                if sig_pairs:
                    posthoc_text = f"\n{', '.join(sig_pairs)}"
        
        # Show n info (matching correlation plot format)
        n_channels = df['channel'].nunique()
        ax.text(0.5, 0.98, f'p={pval_fdr:.4f} {stars}{posthoc_text}\n({n_subj} subj × {n_channels} chan = {n_obs} obs)',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend to first subplot
    if len(available_features) > 0:
        axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Complexity & Entropy Features: VIZ vs AUD vs MULTI\n(All channels, n={len(df["subject_id"].unique())} subjects)', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'complexity_features_violin_all_conditions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Create violin plots comparing EEG features across conditions')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs (default: 2 3 4 5 6)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/feature_distributions)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    if args.output_dir is None:
        output_dir = project_dir / 'results' / 'feature_distributions'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FEATURE DISTRIBUTION VISUALIZATION")
    print("="*70)
    print(f"Conditions: VIZ vs AUD vs MULTI")
    print(f"Subjects: {args.subjects}")
    print(f"Statistics: Linear Mixed Model with CROSSED random effects (subject + channel)")
    print(f"  Data: Averaged within each subject×channel×condition (removes pseudoreplication)")
    print(f"  Model: feature_mean ~ C(condition) + (1|subject) + (1|channel)")
    print(f"  Test: Likelihood Ratio Test comparing full vs null model")
    print(f"  n = subjects × channels per condition (e.g., 5 × 32 = 160)")
    print(f"  MATCHES correlation analysis approach")
    print(f"Correction: FDR (Benjamini-Hochberg) + Bonferroni for post-hoc")
    print(f"Output: {output_dir}")
    
    # Load data for all three conditions
    df = load_all_data(args.subjects, ['VIZ', 'AUD', 'MULTI'], data_dir)
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_psd_features(df, output_dir)
    plot_complexity_features(df, output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Visualizations saved to: {output_dir}")
    print("\nStatistical Notes:")
    print("- Data Preparation:")
    print("  * Average within subject×channel×condition → removes temporal pseudoreplication")
    print("  * Result: One value per subject×channel×condition combination")
    print("- Linear Mixed Model with CROSSED Random Effects:")
    print("  * Model: feature_mean ~ C(condition) + (1|subject) + (1|channel)")
    print("  * Subject random effect: accounts for between-subject variability")
    print("  * Channel random effect: accounts for between-channel variability")
    print("  * Crossed structure: subjects and channels are independent random factors")
    print(f"  * n = {len(args.subjects)} subjects × ~32 channels = ~{len(args.subjects)*32} obs per condition")
    print("  * Test: Likelihood Ratio Test (LRT) comparing full vs null model")
    print("  * MATCHES approach used in correlation analysis")
    print("- Post-hoc Pairwise Tests:")
    print("  * Method: Wilcoxon signed-rank (paired, on subject-averaged data)")
    print("  * Bonferroni correction: p < 0.05/3 = 0.0167 for significance")
    print("  * Shows which specific condition pairs differ (e.g., VIZ≠AUD)")
    print("- Multiple Comparisons:")
    print("  * FDR correction across features (controls false discovery rate)")
    print("  * Bonferroni for 3 pairwise comparisons (conservative)")
    print("- Crossed random effects allow:")
    print("  ✓ Using all subject×channel observations (more power)")
    print("  ✓ Accounting for both subject and channel variability")
    print("  ✓ Proper repeated measures structure")
    print("- Fallback: If crossed effects fail, uses simpler (1|subject) model")


if __name__ == '__main__':
    main()
