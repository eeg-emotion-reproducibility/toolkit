"""DEAP label statistics.

Computes, from the metadata distributed with DEAP:
  * the majority-class baseline of binary valence under fixed thresholds from 4.0 to 6.0;
  * the share of valence ratings within +/-1.0 of the threshold 5.0;
  * the disagreement between stimulus-assigned and self-reported binary labels at threshold 5.0,
    on valence and arousal, where the stimulus-side label of a clip is the average rating of the
    online cohort that rated it.

Usage:
    python deap_labels.py --metadata-dir /path/to/DEAP/metadata
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = [4.0, 4.5, 5.0, 5.5, 6.0]
DEFAULT_THRESHOLD = 5.0


def majority_baselines(valence, thresholds):
    rows = []
    for t in thresholds:
        high = int((valence >= t).sum())
        low = len(valence) - high
        rows.append((t, high, low, "High" if high >= low else "Low", 100.0 * max(high, low) / len(valence)))
    return rows


def stimulus_vs_self_report(ratings, video_list, threshold):
    clips = video_list.dropna(subset=["Experiment_id"]).copy()
    clips["Experiment_id"] = clips["Experiment_id"].astype(int)
    results = {}
    for dim in ["Valence", "Arousal"]:
        stim_mean = dict(zip(clips["Experiment_id"], clips[f"AVG_{dim}"]))
        stim = np.array([stim_mean[int(e)] >= threshold for e in ratings["Experiment_id"]])
        self_report = ratings[dim].to_numpy(dtype=float) >= threshold
        results[dim] = (int((stim != self_report).sum()), len(self_report))
    return results


def raters_per_clip(online, video_list):
    clips = video_list.dropna(subset=["Experiment_id"])
    counts = online[online["Online_id"].isin(clips["Online_id"])].groupby("Online_id").size()
    return int(counts.min()), int(counts.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--participant-ratings", default="participant_ratings.xls")
    parser.add_argument("--video-list", default="video_list.xls")
    parser.add_argument("--online-ratings", default="online_ratings.xls")
    args = parser.parse_args()

    ratings = pd.read_excel(args.metadata_dir / args.participant_ratings)
    video_list = pd.read_excel(args.metadata_dir / args.video_list)
    online = pd.read_excel(args.metadata_dir / args.online_ratings)
    valence = ratings["Valence"].to_numpy(dtype=float)
    n = len(valence)

    print(f"DEAP: {n} trials, {ratings['Participant_id'].nunique()} subjects")
    print("\nMajority-class baseline of binary valence (high = rating >= threshold)")
    print(f"{'Threshold':>10} {'High':>6} {'Low':>6} {'Majority':>9} {'Baseline (%)':>13}")
    for t, high, low, majority, baseline in majority_baselines(valence, THRESHOLDS):
        print(f"{'>= ' + str(t):>10} {high:>6} {low:>6} {majority:>9} {baseline:>13.1f}")

    near = int(((valence >= DEFAULT_THRESHOLD - 1.0) & (valence <= DEFAULT_THRESHOLD + 1.0)).sum())
    print(f"\nValence ratings within +/-1.0 of {DEFAULT_THRESHOLD}: {near}/{n} = {100.0 * near / n:.1f}%")

    lo, hi = raters_per_clip(online, video_list)
    print(f"\nStimulus vs self-report at threshold {DEFAULT_THRESHOLD} "
          f"(stimulus side: online-cohort average, {lo}-{hi} raters per clip)")
    for dim, (k, total) in stimulus_vs_self_report(ratings, video_list, DEFAULT_THRESHOLD).items():
        print(f"  {dim}: labels disagree on {k}/{total} = {100.0 * k / total:.1f}% of trials")


if __name__ == "__main__":
    main()
