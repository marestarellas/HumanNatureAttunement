#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-audio band-matched PLV across the three multisensory conditions:
6 bands x 3 violins per band. Stats inset come from the LMM results.

Reuses the layout from
``scripts/figures/preliminary/Fig15_eeg_audio_correlation_3cond.py``.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# Import the shared layout
sys.path.insert(0, str(Path(__file__).parent))
from Fig15_eeg_audio_correlation_3cond import plot_grid, CONDITIONS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-csv", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling"
                                / "eeg_audio_phase_coupling.csv")
    p.add_argument("--stats-csv", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling"
                                / "eeg_audio_phase_coupling_stats.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures"
                                / "Fig18_eeg_audio_plv_3cond")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phase_csv)
    df = df[df["condition"].isin(CONDITIONS)]
    per_sub = (df.groupby(["subject_id", "condition", "band"], as_index=False)
                  ["plv"].mean())
    stats = pd.DataFrame()
    if args.stats_csv.exists():
        sd = pd.read_csv(args.stats_csv)
        stats = sd[sd["metric"] == "plv"]
    plot_grid(per_sub, stats,
                value_col="plv",
                value_label="PLV (mean across 32 channels)",
                output_path=args.out)


if __name__ == "__main__":
    main()
