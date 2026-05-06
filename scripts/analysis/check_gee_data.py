"""Quick diagnostic script to check NaN patterns in GEE data."""

import pandas as pd
from pathlib import Path

# Load one subject's data to check
data_dir = Path(__file__).parent.parent / 'data'
subjects = [2, 3, 4, 5, 6]
conditions = ['VIZ', 'AUD']

features = ['delta_rel', 'theta_rel', 'alpha_rel', 
           'low_beta_rel', 'high_beta_rel', 'gamma1_rel']

print("Checking NaN patterns in feature files...\n")

all_dfs = []
for subject_id in subjects:
    for condition in conditions:
        subject_folder = f'sub-{subject_id:02d}'
        feature_file = data_dir / 'processed' / subject_folder / 'tables' / f'features_{condition}.csv'
        
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            df['subject_id'] = subject_id
            all_dfs.append(df)
            
            # Check NaNs per channel
            nan_by_channel = df.groupby('channel')[features].apply(lambda x: x.isnull().sum())
            channels_with_nans = nan_by_channel[nan_by_channel.sum(axis=1) > 0]
            
            if len(channels_with_nans) > 0:
                print(f"Subject {subject_id}, {condition}:")
                print(f"  Channels with NaNs: {list(channels_with_nans.index)}")
                print(f"  NaN counts:\n{channels_with_nans}\n")

# Combine all
df_all = pd.concat(all_dfs, ignore_index=True)

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
print(f"Total observations: {len(df_all)}")
print(f"Subjects: {sorted(df_all['subject_id'].unique())}")
print(f"Channels: {sorted(df_all['channel'].unique())}")

print("\nNaN counts by feature (across all data):")
nan_counts = df_all[features].isnull().sum()
print(nan_counts)

print("\nChannels with NaN in any feature:")
nan_by_channel = df_all.groupby('channel')[features].apply(lambda x: x.isnull().any().any())
problem_channels = nan_by_channel[nan_by_channel].index.tolist()
print(f"{len(problem_channels)} channels: {problem_channels}")

print("\nFor problem channels, detailed NaN counts:")
for channel in problem_channels[:5]:  # Show first 5
    ch_df = df_all[df_all['channel'] == channel]
    print(f"\n{channel}:")
    print(ch_df[features].isnull().sum())
