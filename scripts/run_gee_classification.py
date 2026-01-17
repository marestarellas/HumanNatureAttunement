"""
Script to run GEE logistic regression to predict condition from EEG features.

This script uses Generalized Estimating Equations (GEE) with:
- Binary outcome: condition (0/1)
- Multiple predictors: ALL features (PSD + entropy)
- Clustered by subject (accounts for repeated measures)
- One model per electrode
- Shows which features best predict condition

GEE provides population-averaged effects and accounts for within-subject
correlation without requiring full mixed-effects machinery.

Usage:
    python scripts/run_gee_classification.py --cond1 VIZ --cond2 MULTI
    python scripts/run_gee_classification.py --cond1 AUD --cond2 VIZ --features alpha_rel theta_rel lzc
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, accuracy_score
import mne


# Available features
PSD_FEATURES = ['delta_abs', 'theta_abs', 'alpha_abs', 'low_beta_abs', 'high_beta_abs', 'gamma1_abs',
                'delta_rel', 'theta_rel', 'alpha_rel', 'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
ENTROPY_FEATURES = ['lzc', 'perm_entropy', 'spectral_entropy', 'svd_entropy', 'sample_entropy']
ALL_FEATURES = PSD_FEATURES + ENTROPY_FEATURES


def load_subject_data(subject_id, condition, data_dir):
    """Load feature data for one subject and condition."""
    subject_folder = f'sub-{subject_id:02d}'
    feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
    
    if not feature_file.exists():
        print(f"  Warning: {feature_file} not found")
        return None
    
    df = pd.read_csv(feature_file)
    df['subject_id'] = subject_id
    
    return df


def load_all_data(subjects, conditions, data_dir):
    """Load and combine data from multiple subjects and conditions."""
    print(f"\nLoading data for subjects {subjects}")
    print(f"Conditions: {conditions}")
    
    dfs = []
    
    for subject_id in subjects:
        print(f"\n  Subject {subject_id:02d}:")
        for condition in conditions:
            df = load_subject_data(subject_id, condition, data_dir)
            if df is not None:
                print(f"    {condition}: {len(df)} observations")
                dfs.append(df)
    
    if not dfs:
        raise ValueError("No data loaded!")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal combined data: {len(df_combined)} observations")
    print(f"Subjects: {sorted(df_combined['subject_id'].unique())}")
    print(f"Conditions: {sorted(df_combined['condition'].unique())}")
    print(f"Channels: {len(df_combined['channel'].unique())}")
    
    return df_combined


def run_gee_model(df_channel, features, cond1, cond2):
    """
    Run GEE logistic regression for one channel.
    
    Parameters
    ----------
    df_channel : pd.DataFrame
        Data for one channel
    features : list
        List of feature names to use as predictors
    cond1, cond2 : str
        Condition names
    
    Returns
    -------
    result_dict : dict
        Model results including coefficients, p-values, AUC
    """
    # Create binary outcome (0 = cond1, 1 = cond2)
    df_channel = df_channel.copy()
    df_channel['y'] = (df_channel['condition'] == cond2).astype(int)
    
    # Check if any features are entirely NaN for this channel
    for feat in features:
        if df_channel[feat].isnull().all():
            print(f"    Feature '{feat}' is entirely NaN for this channel")
            return None
    
    # Drop rows with missing values in features or outcome and reset index
    df_clean = df_channel[['y', 'subject_id'] + features].dropna(how='any').reset_index(drop=True)
    
    # Check for any remaining NaNs or infs
    if df_clean[features].isnull().any().any():
        nan_cols = df_clean[features].columns[df_clean[features].isnull().any()].tolist()
        print(f"    Warning: NaNs still present in {nan_cols} after dropna")
        return None
    
    if np.isinf(df_clean[features].values).any():
        print(f"    Warning: Inf values detected in features")
        return None
    
    if len(df_clean) < 20:
        print(f"    After removing NaNs: only {len(df_clean)} observations (need ≥20)")
        return None
    
    # Check if we have both conditions
    if df_clean['y'].nunique() < 2:
        print(f"    Warning: Only one condition present after cleaning")
        return None
    
    # Check if we have multiple subjects
    if df_clean['subject_id'].nunique() < 2:
        print(f"    Warning: Only one subject after cleaning (need ≥2 for GEE)")
        return None
    
    # Prepare data
    y = df_clean['y'].values
    X = df_clean[features].values
    
    # Check for zero variance features
    feature_vars = X.var(axis=0)
    if np.any(feature_vars == 0):
        zero_var_feats = [features[i] for i in range(len(features)) if feature_vars[i] == 0]
        print(f"    Warning: Zero variance in features: {zero_var_feats}")
        return None
    
    # Standardize features for better convergence
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero (shouldn't happen after check above)
    X_scaled = (X - X_mean) / X_std
    
    # Final check for NaNs after standardization
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print(f"    Warning: NaN/Inf after standardization")
        return None
    
    # Add intercept
    X_scaled = sm.add_constant(X_scaled)
    
    # Check constant for NaN
    if np.isnan(X_scaled).any():
        print(f"    Warning: NaN in design matrix after adding constant")
        return None
    
    # Subject groups for GEE (must be integer or string, and same length as data)
    groups = df_clean['subject_id'].astype(str).values
    
    # Sort data by groups for GEE (important!)
    sort_idx = np.argsort(groups)
    y = y[sort_idx]
    X_scaled = X_scaled[sort_idx]
    groups = groups[sort_idx]
    
    try:
        # Try Exchangeable correlation structure first
        model = GEE(y, X_scaled, groups=groups, 
                   family=Binomial(),
                   cov_struct=Exchangeable())
        
        result = model.fit(maxiter=100)
        
        # Get coefficients and p-values
        coefs = result.params[1:]  # Exclude intercept
        pvals = result.pvalues[1:]  # Exclude intercept
        
        # Predict probabilities for AUC
        y_pred_prob = result.predict()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y, y_pred_prob)
        accuracy = accuracy_score(y, y_pred)
        
        # Build result dictionary
        result_dict = {
            'intercept': result.params[0],
            'converged': result.converged,
            'n_obs': len(df_clean),
            'auc': auc,
            'accuracy': accuracy,
            'cov_struct': 'exchangeable'
        }
        
        # Add feature coefficients and p-values
        for i, feature in enumerate(features):
            result_dict[f'{feature}_coef'] = coefs[i]
            result_dict[f'{feature}_pval'] = pvals[i]
        
        return result_dict
        
    except ValueError as e:
        if "NaN" in str(e) or "nan" in str(e):
            # Try simpler Independence covariance structure as fallback
            try:
                model = GEE(y, X_scaled, groups=groups, 
                           family=Binomial(),
                           cov_struct=Independence())
                
                result = model.fit(maxiter=100)
                
                # Get coefficients and p-values
                coefs = result.params[1:]
                pvals = result.pvalues[1:]
                
                # Predict probabilities
                y_pred_prob = result.predict()
                y_pred = (y_pred_prob > 0.5).astype(int)
                
                # Calculate metrics
                auc = roc_auc_score(y, y_pred_prob)
                accuracy = accuracy_score(y, y_pred)
                
                # Build result dictionary
                result_dict = {
                    'intercept': result.params[0],
                    'converged': result.converged,
                    'n_obs': len(df_clean),
                    'auc': auc,
                    'accuracy': accuracy,
                    'cov_struct': 'independence'  # Mark as fallback
                }
                
                # Add feature coefficients and p-values
                for i, feature in enumerate(features):
                    result_dict[f'{feature}_coef'] = coefs[i]
                    result_dict[f'{feature}_pval'] = pvals[i]
                
                print(f"    Note: Used Independence structure (Exchangeable failed)")
                return result_dict
                
            except Exception as e2:
                print(f"    Error: NaN in model (Exchangeable and Independence both failed)")
                print(f"    Details: {str(e2)[:100]}")
                return None
        else:
            print(f"    Error: {str(e)[:100]}")
            return None
    except Exception as e:
        print(f"    Error ({type(e).__name__}): {str(e)[:100]}")
        return None


def run_gee_all_channels(df_combined, features, cond1, cond2):
    """
    Run GEE models for all channels.
    
    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined data from all subjects
    features : list
        Features to use as predictors
    cond1, cond2 : str
        Condition names to compare
    
    Returns
    -------
    results_df : pd.DataFrame
        Results for all channels
    """
    print(f"\n{'='*70}")
    print(f"GEE LOGISTIC REGRESSION: Predicting condition from features")
    print(f"{'='*70}")
    print(f"Outcome: {cond1} (0) vs {cond2} (1)")
    print(f"Predictors: {', '.join(features)}")
    print(f"Accounting for subject clustering")
    print(f"{'='*70}")
    
    # Filter to only the two conditions
    df_two_conds = df_combined[df_combined['condition'].isin([cond1, cond2])].copy()
    
    channels = sorted(df_two_conds['channel'].unique())
    results = []
    
    for i, channel in enumerate(channels, 1):
        print(f"\n[{i}/{len(channels)}] {channel}...")
        
        # Get data for this channel
        df_channel = df_two_conds[df_two_conds['channel'] == channel].copy()
        
        # Check for NaNs in features before running model
        nan_counts = df_channel[features].isnull().sum()
        if nan_counts.sum() > 0:
            print(f"    NaN counts before cleaning: {dict(nan_counts[nan_counts > 0])}")
        
        # Run GEE model
        result = run_gee_model(df_channel, features, cond1, cond2)
        
        if result is not None:
            result['channel'] = channel
            auc = result['auc']
            acc = result['accuracy']
            print(f"  ✓ Converged: {result['converged']}, AUC={auc:.3f}, Acc={acc:.3f}")
            results.append(result)
        else:
            print(f"  ✗ Failed")
    
    if len(results) == 0:
        print("\nNo successful models!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"Completed: {len(results_df)} channels")
    print(f"Mean AUC: {results_df['auc'].mean():.3f} ± {results_df['auc'].std():.3f}")
    print(f"Mean Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
    
    # Apply FDR correction for each feature across channels
    for feature in features:
        pval_col = f'{feature}_pval'
        if pval_col in results_df.columns:
            pvals = results_df[pval_col].values
            _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            results_df[f'{feature}_pval_fdr'] = pvals_corrected
            
            n_sig = (pvals < 0.05).sum()
            n_sig_fdr = (pvals_corrected < 0.05).sum()
            print(f"\n{feature}:")
            print(f"  Significant channels (uncorrected): {n_sig}/{len(results_df)}")
            print(f"  Significant channels (FDR): {n_sig_fdr}/{len(results_df)}")
    
    return results_df


def plot_feature_importance_topomap(results_df, feature, cond1, cond2, 
                                    output_dir, alpha=0.05, use_fdr=True):
    """
    Plot topomap showing feature coefficients from GEE models.
    
    Positive coefficients = feature predicts cond2 (vs cond1)
    Negative coefficients = feature predicts cond1 (vs cond2)
    """
    print(f"\nPlotting topomap for {feature}...")
    
    coef_col = f'{feature}_coef'
    pval_col = f'{feature}_pval_fdr' if use_fdr else f'{feature}_pval'
    
    if coef_col not in results_df.columns:
        print(f"  Warning: {coef_col} not in results")
        return
    
    # Standard 32 channel names
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    n_channels = len(results_df)
    mne_ch_names = standard_32_channels[:n_channels]
    
    # Create MNE Info
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    # Get coefficients and p-values
    coefs = results_df[coef_col].values
    pvals = results_df[pval_col].values
    
    # Significance mask
    sig_mask = pvals < alpha
    
    # Mask non-significant
    masked_coefs = coefs.copy()
    masked_coefs[~sig_mask] = 0
    
    # Determine colormap
    if np.any(sig_mask):
        sig_coefs = masked_coefs[sig_mask]
        has_positive = np.any(sig_coefs > 0)
        has_negative = np.any(sig_coefs < 0)
        
        if has_positive and has_negative:
            cmap = 'RdBu_r'
            max_abs = np.max(np.abs(masked_coefs))
            vlim = (-max_abs, max_abs) if max_abs > 0 else (-1e-6, 1e-6)
        elif has_positive:
            cmap = 'Reds'
            vlim = (0, np.max(masked_coefs) if np.max(masked_coefs) > 0 else 1e-6)
        else:
            cmap = 'Blues_r'
            vlim = (np.min(masked_coefs) if np.min(masked_coefs) < 0 else -1e-6, 0)
    else:
        cmap = 'RdBu_r'
        vlim = (-1e-6, 1e-6)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im, _ = mne.viz.plot_topomap(
        masked_coefs, info, axes=ax, show=False,
        cmap=cmap, vlim=vlim, contours=0, sensors=False
    )
    
    # Add significance markers
    if np.any(sig_mask):
        from mne.viz.topomap import _get_pos_outlines
        pos_xy, outlines = _get_pos_outlines(info, None, sphere=None)
        sig_positions = pos_xy[sig_mask]
        ax.scatter(sig_positions[:, 0], sig_positions[:, 1],
                  s=80, c='white', marker='o', edgecolors='black',
                  linewidths=1, zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('GEE Coefficient (standardized)', rotation=270, labelpad=20)
    
    # Title
    pval_label = 'FDR-corrected' if use_fdr else 'uncorrected'
    n_sig = sig_mask.sum()
    ax.set_title(f'{feature}\nPredicting: {cond1} (0) vs {cond2} (1)\n'
                f'Positive = predicts {cond2}, Negative = predicts {cond1}\n'
                f'{n_sig}/{n_channels} significant ({pval_label} p < {alpha})',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'gee_feature_{feature}_{cond1}_vs_{cond2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_model_performance_topomap(results_df, metric, cond1, cond2, output_dir):
    """
    Plot topomap showing model performance (AUC or accuracy) per channel.
    """
    print(f"\nPlotting {metric} topomap...")
    
    if metric not in results_df.columns:
        print(f"  Warning: {metric} not in results")
        return
    
    # Standard channels
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    n_channels = len(results_df)
    mne_ch_names = standard_32_channels[:n_channels]
    
    # Create MNE Info
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    # Get values
    values = results_df[metric].values
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    if metric == 'auc':
        vlim = (0.5, 1.0)
        cmap = 'RdYlGn'
    else:
        vlim = (0.5, 1.0)
        cmap = 'RdYlGn'
    
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False,
        cmap=cmap, vlim=vlim,
        contours=6, sensors=True
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.upper(), rotation=270, labelpad=20)
    
    # Title
    mean_val = values.mean()
    std_val = values.std()
    ax.set_title(f'GEE Model Performance: {metric.upper()}\n'
                f'{cond1} vs {cond2}\n'
                f'Mean = {mean_val:.3f} ± {std_val:.3f}',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'gee_{metric}_{cond1}_vs_{cond2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    
    parser = argparse.ArgumentParser(
        description='Run GEE logistic regression to predict condition from EEG features'
    )
    parser.add_argument('--cond1', type=str, required=True, 
                       help='First condition (coded as 0)')
    parser.add_argument('--cond2', type=str, required=True, 
                       help='Second condition (coded as 1)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs (default: 2 3 4 5 6)')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Features to use (default: all relative PSD + entropy)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance threshold (default: 0.05)')
    parser.add_argument('--no-fdr', action='store_true',
                       help='Disable FDR correction')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    if args.output_dir is None:
        output_dir = project_dir / 'results' / f'gee_classification_{args.cond1}_vs_{args.cond2}'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default features: all relative PSD + all entropy
    if args.features is None:
        features = ['delta_rel', 'theta_rel', 'alpha_rel', 
                   'low_beta_rel', 'high_beta_rel', 'gamma1_rel']
        # Try to add entropy features if they exist
        potential_entropy = ['lzc', 'perm_entropy', 'spectral_entropy', 
                            'svd_entropy', 'sample_entropy']
        # Will be validated below
        features.extend(potential_entropy)
    else:
        features = args.features
    
    # Validate
    invalid = [f for f in features if f not in ALL_FEATURES]
    if invalid:
        print(f"ERROR: Invalid features: {invalid}")
        print(f"Available: {ALL_FEATURES}")
        sys.exit(1)
    
    print("="*70)
    print("GEE LOGISTIC REGRESSION ANALYSIS")
    print("="*70)
    print(f"Predicting condition: {args.cond1} (0) vs {args.cond2} (1)")
    print(f"Subjects: {args.subjects}")
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Alpha: {args.alpha}")
    print(f"FDR correction: {'No' if args.no_fdr else 'Yes'}")
    print(f"Output: {output_dir}")
    
    # Load data
    df_combined = load_all_data(args.subjects, [args.cond1, args.cond2], data_dir)
    
    # Validate features exist in data
    available_features = [f for f in features if f in df_combined.columns]
    missing_features = [f for f in features if f not in df_combined.columns]
    
    if missing_features:
        print(f"\nWarning: Features not found in data (will skip): {missing_features}")
    
    if len(available_features) == 0:
        print("\nERROR: No valid features found in data!")
        print(f"Available columns: {list(df_combined.columns)}")
        sys.exit(1)
    
    features = available_features
    print(f"\nUsing {len(features)} features: {', '.join(features)}")
    
    # Run GEE for all channels
    results_df = run_gee_all_channels(df_combined, features, args.cond1, args.cond2)
    
    if len(results_df) == 0:
        print("\nNo results to save!")
        return
    
    # Save results
    csv_file = output_dir / f'gee_results_{args.cond1}_vs_{args.cond2}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\nSaved results: {csv_file}")
    
    # Plot model performance
    plot_model_performance_topomap(results_df, 'auc', args.cond1, args.cond2, output_dir)
    plot_model_performance_topomap(results_df, 'accuracy', args.cond1, args.cond2, output_dir)
    
    # Plot feature importance for each feature
    print("\nPlotting feature importance topomaps...")
    for feature in features:
        plot_feature_importance_topomap(
            results_df, feature, args.cond1, args.cond2,
            output_dir, args.alpha, use_fdr=not args.no_fdr
        )
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    use_fdr = not args.no_fdr
    pval_suffix = '_pval_fdr' if use_fdr else '_pval'
    pval_type = 'FDR-corrected' if use_fdr else 'uncorrected'
    
    print(f"\nFeature importance ({pval_type} p < {args.alpha}):")
    for feature in features:
        pval_col = f'{feature}{pval_suffix}'
        if pval_col in results_df.columns:
            n_sig = (results_df[pval_col] < args.alpha).sum()
            mean_coef = results_df[f'{feature}_coef'].mean()
            print(f"  {feature:20s}: {n_sig:2d}/{len(results_df)} channels "
                  f"(mean coef = {mean_coef:+.3f})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
