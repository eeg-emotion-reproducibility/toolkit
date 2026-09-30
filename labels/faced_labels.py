"""FACED label statistics.

Computes, from the post-stimulus ratings stored in each subject's After_remarks.mat:
  * the share of valence ratings within +/-1.0 of the scale midpoint 3.5 (0-7 scale);
  * the cross-tabulation of stimulus-assigned and self-reported binary labels on the 24
    non-neutral clips, normalised within each stimulus class, for valence and arousal,
    together with the overall disagreement;
  * the mutual information between the two binary labels and the entropy of the
    self-reported label.

Stimulus valence follows the dataset's category grouping: anger, disgust, fear and sadness
are negative; amusement, inspiration, joy and tenderness are positive; neutral clips are
excluded. FACED defines no arousal classes for its stimuli, so stimulus arousal is assigned
through the circumplex model of affect: anger, disgust, fear, amusement and joy are high;
sadness, tenderness and inspiration are low. Self-reports are binarised at the midpoint 3.5.

Usage:
    python faced_labels.py --faced-dir /path/to/FACED/Data
where --faced-dir holds one folder per subject (sub000, sub001, ...) with After_remarks.mat.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

AROUSAL_COL, VALENCE_COL = 8, 9
MIDPOINT = 3.5

CATEGORY_OF_CLIP = {}
for first, last, category in [(1, 3, "anger"), (4, 6, "disgust"), (7, 9, "fear"), (10, 12, "sadness"),
                              (13, 16, "neutral"), (17, 19, "amusement"), (20, 22, "inspiration"),
                              (23, 25, "joy"), (26, 28, "tenderness")]:
    for clip in range(first, last + 1):
        CATEGORY_OF_CLIP[clip] = category

STIMULUS_VALENCE = {"anger": 0, "disgust": 0, "fear": 0, "sadness": 0,
                    "amusement": 1, "inspiration": 1, "joy": 1, "tenderness": 1}
STIMULUS_AROUSAL = {"anger": 1, "disgust": 1, "fear": 1, "amusement": 1, "joy": 1,
                    "sadness": 0, "tenderness": 0, "inspiration": 0}


def load_ratings(faced_dir):
    trials = []
    for subject in sorted(p for p in faced_dir.iterdir() if p.name.startswith("sub")):
        remarks = loadmat(subject / "After_remarks.mat")["After_remark"]
        for i in range(remarks.shape[0]):
            record = remarks[i, 0]
            score = record["score"]
            if score.size == 0:
                continue
            trials.append((CATEGORY_OF_CLIP[int(record["vid"][0, 0])],
                           float(score[0, VALENCE_COL]), float(score[0, AROUSAL_COL])))
    return trials


def entropy(labels):
    n = len(labels)
    return -sum(c / n * np.log2(c / n) for c in Counter(labels).values())


def mutual_information(a, b):
    return entropy(a) + entropy(b) - entropy(list(zip(a, b)))


def report_axis(name, stimulus, self_report, class_names):
    counts = np.zeros((2, 2), dtype=int)
    for s, r in zip(stimulus, self_report):
        counts[s, r] += 1
    rates = counts / counts.sum(axis=1, keepdims=True)
    print(f"\n{name}: P(self-report | stimulus), {len(stimulus)} non-neutral trials")
    print(f"{'':>16} {'SR=' + class_names[0]:>10} {'SR=' + class_names[1]:>10}")
    for i in range(2):
        print(f"{'Stim=' + class_names[i]:>16} {100 * rates[i, 0]:>9.1f}% {100 * rates[i, 1]:>9.1f}%")
    disagree = int(counts[0, 1] + counts[1, 0])
    print(f"Overall disagreement: {disagree}/{len(stimulus)} = {100 * disagree / len(stimulus):.1f}%")
    print(f"Mutual information: {mutual_information(stimulus, self_report):.4f} bits; "
          f"self-report entropy: {entropy(self_report):.4f} bits")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--faced-dir", type=Path, required=True)
    args = parser.parse_args()

    trials = load_ratings(args.faced_dir)
    valence = np.array([t[1] for t in trials])
    near = int(((valence >= MIDPOINT - 1.0) & (valence <= MIDPOINT + 1.0)).sum())
    print(f"FACED: {len(trials)} rated trials")
    print(f"Valence ratings within +/-1.0 of {MIDPOINT}: {near}/{len(valence)} = {100 * near / len(valence):.1f}%")

    non_neutral = [t for t in trials if t[0] != "neutral"]
    report_axis("Valence",
                [STIMULUS_VALENCE[t[0]] for t in non_neutral],
                [int(t[1] >= MIDPOINT) for t in non_neutral], ("Neg", "Pos"))
    report_axis("Arousal (stimulus classes from the circumplex mapping)",
                [STIMULUS_AROUSAL[t[0]] for t in non_neutral],
                [int(t[2] >= MIDPOINT) for t in non_neutral], ("Low", "High"))


if __name__ == "__main__":
    main()
