"""Diagnostic: inspect raw condition_triggers for problem subjects.

Loads raw EEG/physio for a subject, aligns on first triggers, then prints
a histogram of values and shows what find_last_high_indices yields for
several thresholds. Plots the condition_triggers channel.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from HNA.utils import (  # noqa: E402
    load_data,
    align_by_first_triggers,
    find_last_high_indices,
)


def investigate(subj: str, data_dir: Path, sampling_rate: int = 256, plot_path: Path | None = None):
    print(f"\n========== {subj} ==========")
    eeg = load_data(subj, "eeg", data_dir=str(data_dir))
    physio = load_data(subj, "physio", data_dir=str(data_dir))
    if eeg is None or physio is None:
        print("  Missing data; abort.")
        return

    physio_a, eeg_a = align_by_first_triggers(physio, eeg)
    merged = pd.concat([physio_a, eeg_a], axis=1)
    print(f"  merged rows: {len(merged)}  duration: {len(merged)/sampling_rate:.1f} s")

    ct = merged["condition_triggers"]
    valid = ct.dropna()
    print(f"  condition_triggers: dtype={ct.dtype} non-NaN={len(valid)}/{len(ct)}")
    print(f"    min={valid.min():.1f}  max={valid.max():.1f}  mean={valid.mean():.2f}  std={valid.std():.2f}")
    pcts = [50, 90, 95, 99, 99.5, 99.9, 99.99]
    qs = np.percentile(valid.values, pcts)
    print(f"    percentiles {pcts}: {[f'{q:.0f}' for q in qs]}")

    print("\n  trigger detection at multiple thresholds:")
    for thr in [1500, 1800, 1900, 1950, 2000, 2200, 2500, 3000]:
        idx = find_last_high_indices(merged, threshold=thr)
        if idx:
            spans = np.diff(idx) / sampling_rate
            print(f"    thr={thr:>5}  n_runs={len(idx):>3}  "
                  f"first={idx[0]/sampling_rate:.1f}s  last={idx[-1]/sampling_rate:.1f}s  "
                  f"min_gap={spans.min():.2f}s" if len(idx) > 1 else
                  f"    thr={thr:>5}  n_runs={len(idx)}  "
                  f"first={idx[0]/sampling_rate:.1f}s")
        else:
            print(f"    thr={thr:>5}  n_runs=0 (no samples above threshold)")

    if plot_path is not None:
        t = np.arange(len(merged)) / sampling_rate
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, ct.values, lw=0.4, color='steelblue')
        ax.axhline(2000, color='red', linestyle='--', alpha=0.5, label='thr=2000')
        ax.axhline(1950, color='orange', linestyle='--', alpha=0.5, label='thr=1950')
        ax.set_title(f"{subj}: condition_triggers channel")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n  Plot saved: {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If given, save a PNG per subject here.")
    args = p.parse_args()

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in args.subjects:
        plot_path = (args.out_dir / f"sub-{int(s):02d}_triggers.png") if args.out_dir else None
        investigate(s, args.data_dir, plot_path=plot_path)
