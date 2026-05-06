"""
Script to diagnose why correlated features show opposite GLM effects.

Checks:
1. Overall correlations (already done)
2. Within-subject correlations (what the mixed model sees)
3. Bivariate condition effects before random effects
4. Variance inflation factors (VIF) if features modeled together

Usage:
    python diagnose_entropy_effects.py --cond1 VIZ --cond2 MULTI --subjects 2 3 4 5 6
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_subject_data(subject_id, condition, data_dir):
    """Load feature data for one subject and condition."""
    subject_folder = f'sub-{subject_id:02d}'
    feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
    
    if not feature_file.exists():
        return None
    
    df = pd.read_csv(feature_file)
    df['subject_id'] = subject_id
    return df


def load_all_data(subjects, conditions, data_dir):
    """Load and combine data from multiple subjects and conditions."""
    dfs = []
    for subject_id in subjects:
        for condition in conditions:
            df = load_subject_data(subject_id, condition, data_dir)
            if df is not None:
                dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else None


def check_within_subject_correlations(df, features):
    """Check correlations within each subject separately."""
    
    print("\n" + "="*60)
    print("WITHIN-SUBJECT CORRELATIONS")
    print("="*60)
    print("(Mixed models primarily use within-subject variation)")
    
    subjects = sorted(df['subject_id'].unique())
    
    # Store correlations for each subject
    subject_corrs = []
    
    for subject_id in subjects:
        df_subj = df[df['subject_id'] == subject_id][features].dropna()
        
        if len(df_subj) < 3:
            continue
        
        corr_subj = df_subj.corr()
        subject_corrs.append(corr_subj)
        
        print(f"\nSubject {subject_id}:")
        print(f"  N observations: {len(df_subj)}")
        
        # Show key correlations
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                corr_val = corr_subj.loc[feat1, feat2]
                print(f"  {feat1} vs {feat2}: {corr_val:.3f}")
    
    # Average within-subject correlation
    if subject_corrs:
        avg_within_corr = np.mean([c.values for c in subject_corrs], axis=0)
        avg_within_df = pd.DataFrame(avg_within_corr, 
                                     columns=features, 
                                     index=features)
        
        print("\n" + "-"*60)
        print("AVERAGE WITHIN-SUBJECT CORRELATION:")
        print(avg_within_df.to_string())
        
        return avg_within_df
    
    return None


def check_bivariate_condition_effects(df, features, cond1, cond2):
    """Check simple bivariate t-tests for each feature."""
    
    print("\n" + "="*60)
    print("BIVARIATE CONDITION EFFECTS (Simple t-tests)")
    print("="*60)
    print(f"Comparing: {cond1} vs {cond2}")
    print("(Ignoring subject structure)")
    
    df_two = df[df['condition'].isin([cond1, cond2])].copy()
    
    results = []
    
    for feature in features:
        data_cond1 = df_two[df_two['condition'] == cond1][feature].dropna()
        data_cond2 = df_two[df_two['condition'] == cond2][feature].dropna()
        
        if len(data_cond1) > 0 and len(data_cond2) > 0:
            t_stat, p_val = stats.ttest_ind(data_cond1, data_cond2)
            
            mean_diff = data_cond2.mean() - data_cond1.mean()
            
            results.append({
                'feature': feature,
                'mean_cond1': data_cond1.mean(),
                'mean_cond2': data_cond2.mean(),
                'mean_diff': mean_diff,
                't_stat': t_stat,
                'p_value': p_val,
                'direction': 'positive' if mean_diff > 0 else 'negative'
            })
            
            print(f"\n{feature}:")
            print(f"  Mean {cond1}: {data_cond1.mean():.6f}")
            print(f"  Mean {cond2}: {data_cond2.mean():.6f}")
            print(f"  Difference ({cond2} - {cond1}): {mean_diff:.6f}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_val:.4f}")
            print(f"  Effect direction: {results[-1]['direction']}")
    
    return pd.DataFrame(results)


def check_within_subject_condition_effects(df, features, cond1, cond2):
    """Check condition effects within each subject."""
    
    print("\n" + "="*60)
    print("WITHIN-SUBJECT CONDITION EFFECTS")
    print("="*60)
    print(f"Comparing: {cond1} vs {cond2}")
    
    df_two = df[df['condition'].isin([cond1, cond2])].copy()
    subjects = sorted(df_two['subject_id'].unique())
    
    results = []
    
    for subject_id in subjects:
        df_subj = df_two[df_two['subject_id'] == subject_id]
        
        print(f"\nSubject {subject_id}:")
        
        for feature in features:
            data_cond1 = df_subj[df_subj['condition'] == cond1][feature].dropna()
            data_cond2 = df_subj[df_subj['condition'] == cond2][feature].dropna()
            
            if len(data_cond1) > 0 and len(data_cond2) > 0:
                mean_diff = data_cond2.mean() - data_cond1.mean()
                direction = 'positive' if mean_diff > 0 else 'negative'
                
                results.append({
                    'subject_id': subject_id,
                    'feature': feature,
                    'mean_diff': mean_diff,
                    'direction': direction
                })
                
                print(f"  {feature}: {mean_diff:+.6f} ({direction})")
    
    results_df = pd.DataFrame(results)
    
    # Summary: how consistent are directions across subjects?
    if len(results_df) > 0:
        print("\n" + "-"*60)
        print("CONSISTENCY ACROSS SUBJECTS:")
        
        for feature in features:
            feature_results = results_df[results_df['feature'] == feature]
            n_positive = (feature_results['direction'] == 'positive').sum()
            n_negative = (feature_results['direction'] == 'negative').sum()
            
            print(f"\n{feature}:")
            print(f"  Subjects with positive effect: {n_positive}")
            print(f"  Subjects with negative effect: {n_negative}")
            print(f"  Consistency: {max(n_positive, n_negative) / len(feature_results) * 100:.1f}%")
    
    return results_df


def plot_condition_effects_comparison(bivariate_results, output_dir):
    """Plot comparison of bivariate effects."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = bivariate_results['feature'].values
    effects = bivariate_results['mean_diff'].values
    colors = ['red' if x > 0 else 'blue' for x in effects]
    
    ax.barh(features, effects, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Difference (Cond2 - Cond1)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Bivariate Condition Effects\n(Before accounting for random effects)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'bivariate_effects_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def main():
    """Main function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Diagnose entropy effect directions')
    parser.add_argument('--cond1', type=str, required=True, help='First condition')
    parser.add_argument('--cond2', type=str, required=True, help='Second condition')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs to include')
    parser.add_argument('--channel', type=str, default='EEG-ch1',
                       help='Channel to analyze (default: EEG-ch1)')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'results' / f'entropy_diagnostics_{args.cond1}_vs_{args.cond2}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ENTROPY EFFECTS DIAGNOSTIC")
    print("="*60)
    print(f"Conditions: {args.cond1} vs {args.cond2}")
    print(f"Subjects: {args.subjects}")
    print(f"Channel: {args.channel}")
    
    # Load data
    df = load_all_data(args.subjects, [args.cond1, args.cond2], data_dir)
    
    if df is None:
        print("ERROR: No data loaded!")
        return
    
    # Filter to one channel for clarity
    df = df[df['channel'] == args.channel].copy()
    
    print(f"\nTotal observations: {len(df)}")
    
    # Define entropy features
    entropy_features = ['lzc', 'perm_entropy', 'spectral_entropy', 'svd_entropy', 'sample_entropy']
    entropy_features = [f for f in entropy_features if f in df.columns]
    
    print(f"Entropy features: {entropy_features}")
    
    # 1. Check overall correlation
    print("\n" + "="*60)
    print("OVERALL CORRELATIONS (Between-subject + within-subject)")
    print("="*60)
    corr_overall = df[entropy_features].corr()
    print(corr_overall.to_string())
    
    # 2. Check within-subject correlations
    avg_within_corr = check_within_subject_correlations(df, entropy_features)
    
    # 3. Bivariate condition effects (ignoring subject structure)
    bivariate_results = check_bivariate_condition_effects(df, entropy_features, args.cond1, args.cond2)
    
    # 4. Within-subject condition effects
    within_subject_results = check_within_subject_condition_effects(df, entropy_features, args.cond1, args.cond2)
    
    # 5. Plot comparison
    if bivariate_results is not None and len(bivariate_results) > 0:
        plot_condition_effects_comparison(bivariate_results, output_dir)
    
    # 6. Key insight
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("\nIf you see opposite effect directions despite positive correlations:")
    print("  1. Features may be positively correlated OVERALL but show different")
    print("     within-subject changes in response to conditions")
    print("  2. Mixed models use WITHIN-subject variation, not between-subject")
    print("  3. Check if within-subject correlations differ from overall correlations")
    print("  4. Individual subjects may show inconsistent direction patterns")
    print("\nRecommendation:")
    print("  - If subjects show inconsistent directions, consider subject-level moderators")
    print("  - If within-subject correlations are weak/negative, features capture")
    print("    different aspects of complexity despite overall positive correlation")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
