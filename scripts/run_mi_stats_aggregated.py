"""
Aggregated statistical analysis for mutual information - one model per band.

Uses all channels and subjects together with nested random effects for better power.
This complements the per-channel analysis in run_mi_stats.py.

Model: MI ~ condition + (1|subject) + (1|channel)
n = 128 observations per condition (4 subjects × 32 channels)

Usage:
    python run_mi_stats_aggregated.py --metric mi_direct --condition1 VIZ --condition2 AUD
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


def load_mi_results(results_dir, condition1, condition2):
    """Load MI results for two conditions."""
    results_file = results_dir / 'audio_eeg_mutual_information_results.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    df = df[df['condition'].isin([condition1, condition2])].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for conditions {condition1}, {condition2}")
    
    print(f"Loaded {len(df)} records")
    print(f"  Subjects: {sorted(df['subject_id'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Bands: {sorted(df['band'].unique())}")
    print(f"  Channels: {len(df['channel'].unique())}")
    
    return df


def run_aggregated_mixed_model(df, metric, band, condition1, condition2):
    """
    Run mixed-effects model with all channels and subjects.
    
    Model: metric ~ C(condition) + (1|subject_id) + (1|channel)
    
    This tests the overall effect of condition across all channels,
    accounting for both subject and channel variability.
    """
    band_df = df[df['band'] == band].copy()
    band_df['subject_id'] = band_df['subject_id'].astype(str)
    band_df['channel'] = band_df['channel'].astype(str)
    
    # Check variance
    if band_df[metric].std() < 1e-10:
        return None
    
    n_obs = len(band_df)
    n_subjects = band_df['subject_id'].nunique()
    n_channels = band_df['channel'].nunique()
    
    # Calculate descriptive statistics
    cond1_data = band_df[band_df['condition'] == condition1][metric]
    cond2_data = band_df[band_df['condition'] == condition2][metric]
    mean_diff = cond2_data.mean() - cond1_data.mean()
    
    formula = f"{metric} ~ C(condition)"
    
    try:
        # Try with both random effects
        model = smf.mixedlm(formula, band_df, groups=band_df["subject_id"], 
                           re_formula="1", vc_formula={"channel": "0 + C(channel)"})
        result = model.fit(method='powell', maxiter=2000, reml=True)
        
        # Extract coefficient for condition
        coef_key = None
        for key in result.params.index:
            if 'condition' in key and condition2 in key:
                coef_key = key
                break
        if coef_key is None:
            for key in result.params.index:
                if 'condition' in key and condition1 in key:
                    coef_key = key
                    break
        
        if coef_key is None:
            coef = np.nan
            pvalue = np.nan
        else:
            coef = result.params[coef_key]
            pvalue = result.pvalues[coef_key]
        
        return {
            'band': band,
            'test': 'mixed-lm-aggregated',
            'coefficient': coef,
            'p_value': pvalue,
            'mean_diff': mean_diff,
            'mean_cond1': cond1_data.mean(),
            'mean_cond2': cond2_data.mean(),
            'n_obs': n_obs,
            'n_subjects': n_subjects,
            'n_channels': n_channels,
            'model_type': 'mixed_aggregated',
            'converged': result.converged
        }
        
    except Exception as e:
        print(f"    Warning - {band}: Mixed model failed ({str(e)[:50]}), trying simpler model...")
        
        # Fallback: simpler model with only subject random effect
        try:
            model = smf.mixedlm(formula, band_df, groups=band_df["subject_id"])
            result = model.fit(method='powell', maxiter=1000, reml=True)
            
            # Extract coefficient
            coef_key = None
            for key in result.params.index:
                if 'condition' in key:
                    coef_key = key
                    break
            
            if coef_key is None:
                coef = np.nan
                pvalue = np.nan
            else:
                coef = result.params[coef_key]
                pvalue = result.pvalues[coef_key]
            
            return {
                'band': band,
                'test': 'mixed-lm-simple',
                'coefficient': coef,
                'p_value': pvalue,
                'mean_diff': mean_diff,
                'mean_cond1': cond1_data.mean(),
                'mean_cond2': cond2_data.mean(),
                'n_obs': n_obs,
                'n_subjects': n_subjects,
                'n_channels': n_channels,
                'model_type': 'mixed_simple',
                'converged': result.converged
            }
            
        except Exception as e2:
            print(f"    Error - {band}: Both models failed: {str(e2)[:100]}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Aggregated statistical analysis for MI (one model per band)'
    )
    parser.add_argument('--metric', type=str, required=True,
                       choices=['mi_direct', 'mi_max_lagged'],
                       help='Metric to analyze')
    parser.add_argument('--condition1', type=str, required=True,
                       help='First condition')
    parser.add_argument('--condition2', type=str, required=True,
                       help='Second condition')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory')
    
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
    
    print("="*60)
    print("AGGREGATED MIXED-EFFECTS ANALYSIS")
    print("="*60)
    print(f"Metric: {args.metric}")
    print(f"Conditions: {args.condition1} vs {args.condition2}")
    print()
    
    # Load data
    df = load_mi_results(results_dir, args.condition1, args.condition2)
    
    # Run aggregated model for each band
    print("\nRunning aggregated mixed-effects models...")
    print("Model: metric ~ C(condition) + (1|subject) + (1|channel)")
    print()
    
    bands = sorted(df['band'].unique())
    results_list = []
    
    for band in bands:
        print(f"  {band}...", end=' ')
        result = run_aggregated_mixed_model(df, args.metric, band, 
                                           args.condition1, args.condition2)
        if result is not None:
            results_list.append(result)
            
            # Display result
            p_str = f"p={result['p_value']:.4f}"
            if result['p_value'] < 0.001:
                p_str = "p<0.001 ***"
            elif result['p_value'] < 0.01:
                p_str = f"p={result['p_value']:.4f} **"
            elif result['p_value'] < 0.05:
                p_str = f"p={result['p_value']:.4f} *"
            
            print(f"{p_str}, coef={result['coefficient']:.4f}, "
                  f"Δ={result['mean_diff']:.4f}, n={result['n_obs']}")
        else:
            print("FAILED")
    
    if len(results_list) == 0:
        print("\nERROR: No valid results obtained")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results
    stats_dir = results_dir / 'statistics'
    stats_dir.mkdir(exist_ok=True)
    
    output_file = stats_dir / f'stats_aggregated_{args.metric}_{args.condition1}_vs_{args.condition2}.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    n_sig_05 = np.sum(results_df['p_value'] < 0.05)
    n_sig_01 = np.sum(results_df['p_value'] < 0.01)
    n_sig_001 = np.sum(results_df['p_value'] < 0.001)
    n_total = len(results_df)
    
    print(f"Significant bands (p < 0.05): {n_sig_05}/{n_total}")
    print(f"Significant bands (p < 0.01): {n_sig_01}/{n_total}")
    print(f"Significant bands (p < 0.001): {n_sig_001}/{n_total}")
    print()
    print("Per-band results:")
    print(results_df[['band', 'p_value', 'coefficient', 'mean_diff', 'n_obs', 'model_type']].to_string(index=False))
    print()
    print(f"Results saved to: {output_file}")
    print()
    print("NOTE: These aggregated statistics test the OVERALL effect")
    print("      across all channels, providing better statistical power.")
    print("      Use these for the violin plots that show distributions")
    print("      across all channels.")
    print()
    print("      The per-channel statistics in run_mi_stats.py show")
    print("      SPATIAL specificity and are used for topomaps.")


if __name__ == '__main__':
    main()
