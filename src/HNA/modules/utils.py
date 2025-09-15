import os
import pandas as pd
import numpy as np
import soundfile as sf  # pip install soundfile
import json

def load_data(subject, modality, data_dir='../data'):
    """
    Load modality data for a given subject.
    For EEG: loads EEG.csv and D in.csv, returns a single df with a 'triggers' column (D in-ch2).
    For physio: returns a single df with columns ['ecg', 'respiration', 'triggers', 'condition_triggers'].
    """
    # Format subject folder name
    if str(subject).startswith('sub-'):
        subj_folder = str(subject)
    else:
        subj_folder = f'sub-{int(subject):02d}'

    if modality.lower() == 'eeg':
        folder = os.path.join(data_dir, subj_folder, 'eeg')
        eeg_path = os.path.join(folder, 'EEG.csv')
        din_path = os.path.join(folder, 'D in.csv')
        eeg_df = pd.read_csv(eeg_path) if os.path.exists(eeg_path) else None
        din_df = pd.read_csv(din_path) if os.path.exists(din_path) else None

        if eeg_df is not None:
            if din_df is not None and 'D in-ch2' in din_df.columns:
                # Add triggers column by index, align lengths (pad with NaN)
                triggers = din_df['D in-ch2'].rename('triggers').reset_index(drop=True)
                eeg_df = eeg_df.reset_index(drop=True)
                # If lengths differ, pad with NaN
                n = max(len(eeg_df), len(triggers))
                eeg_df = eeg_df.reindex(range(n))
                triggers = triggers.reindex(range(n))
                eeg_df['eeg_triggers'] = triggers
            else:
                print(f"'D in-ch2' column not found in D in.csv or file missing.")
                eeg_df['eeg_triggers'] = float('nan')
            return eeg_df
        else:
            print(f"EEG.csv not found in {folder}")
            return None

    elif modality.lower() == 'physio':
        folder = os.path.join(data_dir, subj_folder, 'physio')  # adjust if physio files are elsewhere

        channel_files = {
            'ecg': ('ExG [1].csv', 'ExG [1]-ch1'),
            'respiration': ('Analog AUX [1].csv', 'Analog AUX [1]-ch1'),
            'physio_triggers': ('D out.csv', 'D out-ch1'),
            'condition_triggers': ('ExG [2].csv', 'ExG [2]-ch1')
        }

        dfs = []
        for key, (fname, colname) in channel_files.items():
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                if colname in df.columns:
                    df = df[[colname]].rename(columns={colname: key})
                else:
                    print(f"Column '{colname}' not found in {fname}. Filling with NaN.")
                    df = pd.DataFrame({key: [float('nan')]*len(df)})
            else:
                print(f"File not found: {fpath}. Filling column '{key}' with NaN.")
                if dfs:
                    df = pd.DataFrame({key: [float('nan')]*len(dfs[0])})
                else:
                    df = pd.DataFrame({key: []})
            dfs.append(df.reset_index(drop=True))

        # Merge all columns by index
        from functools import reduce
        if dfs:
            physio_df = reduce(lambda left, right: pd.concat([left, right], axis=1), dfs)
            return physio_df
        else:
            print(f"None of the physio files found in {folder}")
            return None

    elif modality.lower() == 'audio':

        folder = os.path.join(data_dir, subj_folder, 'audio')
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                path = os.path.join(folder, file)
                audio, sr = sf.read(path)
                return audio, sr
        print(f"No audio (.wav) file found in {folder}")
        return None

    else:
        raise ValueError("modality must be 'eeg', 'physio', or 'audio'")


def align_by_first_triggers(physio_df, eeg_df):
    """
    Aligns physio and eeg DataFrames by their respective first trigger event.
    Both DataFrames must have a 'triggers' column.
    Cuts both DataFrames at the later first trigger index so they remain aligned.
    Returns the sliced DataFrames.
    """
    # Find the first nonzero trigger in each DataFrame
    physio_idx = physio_df['physio_triggers'].ne(0).idxmax()
    eeg_idx = eeg_df['eeg_triggers'].ne(0).idxmax()
    print(f"Physio first trigger index: {physio_idx}, EEG first trigger index: {eeg_idx}")
    physio_aligned = physio_df.iloc[physio_idx:].reset_index(drop=True)
    eeg_aligned = eeg_df.iloc[eeg_idx:].reset_index(drop=True)
    return physio_aligned, eeg_aligned

def find_last_high_indices(merged_data, threshold=2000):
    """
    Find the last index in each consecutive run of merged_data['condition_triggers'] > threshold.
    Returns a list of indices.
    """
    high_indices = merged_data.index[merged_data['condition_triggers'] > threshold].tolist()
    high_indices = np.array(high_indices)
    if high_indices.size > 0:
        breaks = np.where(np.diff(high_indices) > 1)[0]
        last_in_run = np.append(breaks, len(high_indices) - 1)
        high_indices = high_indices[last_in_run].tolist()
    return high_indices

def annotate_conditions(merged_data, condition_indices, condition_list):
    """
    Adds a 'condition_names' column to merged_data.
    - Each of the first N-10 indices in condition_indices is labeled 'AUDIO_SYNC'.
    - The last 10 are mapped to their event label.
    - All other rows are 0.

    Parameters
    ----------
    merged_data : pd.DataFrame
    condition_indices : list of int
        Event indices (can be more than 10; last 10 are mapped to event labels).
    condition_list : list of str
        Three condition names, e.g. ["MULTI", "AUD", "VIZ"].

    Returns
    -------
    merged_data : pd.DataFrame
        With new column 'condition_names'.
    """
    assert len(condition_list) == 3, "Expected 3 condition names in condition_list."
    assert len(condition_indices) >= 10, "Need at least 10 indices."

    # Labels for last 10
    last_labels = [
        'RS1_start', 'RS1_stop',
        f"{condition_list[0]}_start", f"{condition_list[0]}_stop",
        f"{condition_list[1]}_start", f"{condition_list[1]}_stop",
        f"{condition_list[2]}_start", f"{condition_list[2]}_stop",
        'RS2_start', 'RS2_stop'
    ]

    merged_data = merged_data.copy()
    merged_data['condition_names'] = 0

    # AUDIO_SYNC for all indices before the last 10
    for idx in condition_indices[:-10]:
        if 0 <= idx < len(merged_data):
            merged_data.at[idx, 'condition_names'] = 'AUDIO_SYNC'

    # Event label for the last 10
    for idx, label in zip(condition_indices[-10:], last_labels):
        if 0 <= idx < len(merged_data):
            merged_data.at[idx, 'condition_names'] = label

    return merged_data


# ---- helpers ----
def parse_mmss(value):
    """Accepts 'mm:ss', 'h:mm:ss', or a numeric seconds value; returns seconds (float)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = int(parts[0]), float(parts[1])
        return m * 60 + sec
    elif len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + sec
    else:
        raise ValueError(f"Unrecognized time format: {value!r}")

def get_sync_seconds_for_subject(json_path, subj):
    """Robustly resolve the subject key in the JSON and return sync time (in seconds)."""
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Try several key variants: sub02, sub-02, 02, sub2, sub-2, 2
    s02 = f"{int(subj):02d}"
    s2  = str(int(subj))
    candidates = [f"sub{s02}", f"sub-{s02}", s02, f"sub{s2}", f"sub-{s2}", s2]

    # direct matches first
    for k in candidates:
        if k in cfg:
            return parse_mmss(cfg[k])

    # case-insensitive fallback
    low = {k.lower(): k for k in cfg.keys()}
    for k in candidates:
        lk = k.lower()
        if lk in low:
            return parse_mmss(cfg[low[lk]])

    raise KeyError(f"No sync time for subject {subj}. Tried keys: {candidates}")

def load_audio_for_subject(subj, data_dir):
    """Loads the first .wav in sub-XX/audio/. Returns (mono_np_array, sr, original_path or None)."""
    folder = os.path.join(data_dir, f"sub-{int(subj):02d}", "audio")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Audio folder not found: {folder}")

    wav_path = None
    for fn in os.listdir(folder):
        if fn.lower().endswith(".wav"):
            wav_path = os.path.join(folder, fn)
            break
    if wav_path is None:
        raise FileNotFoundError(f"No .wav file in {folder}")

    audio, sr = sf.read(wav_path)
    # Make mono like your snippet (left channel)
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio, sr, wav_path

def get_condition_segments(df, conditions):
    """Return a dict of start/stop indices for each condition"""
    indices = {}
    for cond in conditions:
        # Find all indices labeled as this condition
        cond_idx = df.index[df['condition_names'] == cond].tolist()
        if cond_idx:
            # Assume one continuous segment per condition
            indices[cond] = (cond_idx[0])
        else:
            indices[cond] = None
    return indices