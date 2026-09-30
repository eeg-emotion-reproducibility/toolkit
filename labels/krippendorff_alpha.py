"""Inter-subject agreement of self-reported emotion ratings (Krippendorff's alpha).

Subjects are treated as raters and stimuli as units. For each dataset the script reports
alpha on the continuous valence and arousal ratings (interval level) and on binary valence
(nominal level), binarised at the scale midpoint: DEAP >= 5 (1-9 SAM), FACED >= 3.5
(0-7 scale), DREAMER > 3 (1-5 SAM). Any subset of the three datasets can be given.

Usage:
    python krippendorff_alpha.py --faced-dir /path/to/FACED/Data \
        --deap-metadata-dir /path/to/DEAP/metadata --dreamer-mat /path/to/DREAMER.mat
"""
import argparse
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.io import loadmat

FACED_AROUSAL_COL, FACED_VALENCE_COL = 8, 9


def faced_matrices(faced_dir):
    subjects = sorted(p for p in faced_dir.iterdir() if p.name.startswith("sub"))
    valence = np.full((len(subjects), 28), np.nan)
    arousal = np.full((len(subjects), 28), np.nan)
    for s, subject in enumerate(subjects):
        remarks = loadmat(subject / "After_remarks.mat")["After_remark"]
        for i in range(remarks.shape[0]):
            record = remarks[i, 0]
            score = record["score"]
            if score.size == 0:
                continue
            clip = int(record["vid"][0, 0]) - 1
            valence[s, clip] = score[0, FACED_VALENCE_COL]
            arousal[s, clip] = score[0, FACED_AROUSAL_COL]
    return valence, arousal, np.where(np.isnan(valence), np.nan, (valence >= 3.5).astype(float))


def deap_matrices(metadata_dir, filename):
    ratings = pd.read_excel(metadata_dir / filename)
    subjects = sorted(ratings["Participant_id"].unique())
    valence = np.full((len(subjects), 40), np.nan)
    arousal = np.full((len(subjects), 40), np.nan)
    for s, subject in enumerate(subjects):
        rows = ratings[ratings["Participant_id"] == subject]
        clips = rows["Experiment_id"].to_numpy(dtype=int) - 1
        valence[s, clips] = rows["Valence"].to_numpy(dtype=float)
        arousal[s, clips] = rows["Arousal"].to_numpy(dtype=float)
    return valence, arousal, (valence >= 5.0).astype(float)


def dreamer_matrices(mat_path):
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["DREAMER"].Data
    valence = np.stack([s.ScoreValence for s in data]).astype(float)
    arousal = np.stack([s.ScoreArousal for s in data]).astype(float)
    return valence, arousal, (valence > 3).astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--faced-dir", type=Path)
    parser.add_argument("--deap-metadata-dir", type=Path)
    parser.add_argument("--deap-participant-ratings", default="participant_ratings.xls")
    parser.add_argument("--dreamer-mat", type=Path)
    args = parser.parse_args()

    datasets = []
    if args.faced_dir:
        datasets.append(("FACED", faced_matrices(args.faced_dir)))
    if args.deap_metadata_dir:
        datasets.append(("DEAP", deap_matrices(args.deap_metadata_dir, args.deap_participant_ratings)))
    if args.dreamer_mat:
        datasets.append(("DREAMER", dreamer_matrices(args.dreamer_mat)))
    if not datasets:
        parser.error("give at least one of --faced-dir, --deap-metadata-dir, --dreamer-mat")

    print(f"{'Dataset':<8} {'Subjects x Stimuli':>18} {'Valence':>8} {'Arousal':>8} {'Binary valence':>15}")
    for name, (valence, arousal, binary) in datasets:
        a_val = krippendorff.alpha(reliability_data=valence, level_of_measurement="interval")
        a_aro = krippendorff.alpha(reliability_data=arousal, level_of_measurement="interval")
        a_bin = krippendorff.alpha(reliability_data=binary, level_of_measurement="nominal")
        shape = f"{valence.shape[0]} x {valence.shape[1]}"
        print(f"{name:<8} {shape:>18} {a_val:>8.3f} {a_aro:>8.3f} {a_bin:>15.3f}")


if __name__ == "__main__":
    main()
