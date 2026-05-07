"""Quick diagnostic: count NaN cells in extracted EEG band-power feature CSVs.

Loads ``features_<CONDITION>.csv`` for a chosen subject set and reports:
  - per-subject / per-condition channels with any NaN in the band columns,
  - global NaN counts per feature,
  - the list of channels that have at least one NaN anywhere.

Used after a fresh ``extract_eeg_features.py`` run to spot bad channels
that should be excluded or re-cleaned.

Usage:
    python scripts/analysis/_check_gee_data.py
    python scripts/analysis/_check_gee_data.py --subjects 2 3 4 5 6 \\
        --conditions VIZ AUD MULTI --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURES = [
    "delta_rel", "theta_rel", "alpha_rel",
    "low_beta_rel", "high_beta_rel", "gamma1_rel",
]
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                   help="Subject IDs to scan (default: 2 3 4 5 6).")
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD"],
                   help="Conditions to scan (default: VIZ AUD).")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data",
                   help=f"Data root containing processed/sub-*/ (default: {ROOT / 'data'}).")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                   help=f"Feature columns to inspect (default: {DEFAULT_FEATURES}).")
    return p.parse_args()


def main():
    args = parse_args()
    print("Checking NaN patterns in feature files...\n")

    all_dfs = []
    for subject_id in args.subjects:
        for condition in args.conditions:
            subject_folder = f"sub-{subject_id:02d}"
            feature_file = (args.data_dir / "processed" / subject_folder
                            / "tables" / f"features_{condition}.csv")
            if not feature_file.exists():
                print(f"  SKIP {subject_folder}/{condition}: missing {feature_file.name}")
                continue

            df = pd.read_csv(feature_file)
            df["subject_id"] = subject_id
            all_dfs.append(df)

            nan_by_channel = (df.groupby("channel")[args.features]
                                .apply(lambda x: x.isnull().sum()))
            channels_with_nans = nan_by_channel[nan_by_channel.sum(axis=1) > 0]
            if len(channels_with_nans) > 0:
                print(f"Subject {subject_id}, {condition}:")
                print(f"  Channels with NaNs: {list(channels_with_nans.index)}")
                print(f"  NaN counts:\n{channels_with_nans}\n")

    if not all_dfs:
        print("\nNo feature CSVs found. Run scripts/features/extract_eeg_features.py first.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total observations: {len(df_all)}")
    print(f"Subjects: {sorted(df_all['subject_id'].unique())}")
    print(f"Channels: {sorted(df_all['channel'].unique())}")

    print("\nNaN counts by feature (across all data):")
    print(df_all[args.features].isnull().sum())

    print("\nChannels with NaN in any feature:")
    nan_by_channel = (df_all.groupby("channel")[args.features]
                            .apply(lambda x: x.isnull().any().any()))
    problem_channels = nan_by_channel[nan_by_channel].index.tolist()
    print(f"{len(problem_channels)} channels: {problem_channels}")

    print("\nFor problem channels, detailed NaN counts:")
    for channel in problem_channels[:5]:
        ch_df = df_all[df_all["channel"] == channel]
        print(f"\n{channel}:")
        print(ch_df[args.features].isnull().sum())


if __name__ == "__main__":
    main()
