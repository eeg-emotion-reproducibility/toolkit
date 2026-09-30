"""Correlation between the DE features of the two documented FACED preprocessing pipelines.

For each subject, computes the Pearson correlation between the DE features of the FACED
release (Chen et al., 2023) and the DE features extracted with the same procedure from
DAEST's preprocessed signal (daest_de_features.py), per band and over all bands, on the
30 channels the two share (the release's first 30 channels; DAEST drops the two mastoids).
Reports the mean and SD over subjects (numpy default, ddof = 0).

Usage:
    python cross_pipeline_correlation.py --chen-de-dir /path/to/FACED/EEG_Features/DE \
        --daest-de-dir /path/to/daest_de --out results/cross_pipeline_correlation.json
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.stats import pearsonr

N_SUBS = 123
N_CHANNELS = 30
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BAND_NAMES)


def load_de_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return np.asarray(obj.get("data", list(obj.values())[0]))
    return np.asarray(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chen-de-dir", required=True)
    parser.add_argument("--daest-de-dir", required=True)
    parser.add_argument("--out", default="cross_pipeline_correlation.json")
    args = parser.parse_args()

    per_sub_per_band = np.full((N_SUBS, N_BANDS), np.nan)
    per_sub_overall = np.full(N_SUBS, np.nan)
    skipped = []

    for sub in range(N_SUBS):
        chen_path = os.path.join(args.chen_de_dir, f"sub{sub:03d}.pkl")
        daest_path = os.path.join(args.daest_de_dir, f"sub{sub:03d}.pkl")
        if not os.path.exists(chen_path):
            chen_path = chen_path + ".pkl"
        if not os.path.exists(chen_path) or not os.path.exists(daest_path):
            skipped.append(sub)
            continue

        chen_de = load_de_pkl(chen_path)[:, :N_CHANNELS, :, :]
        daest_de = load_de_pkl(daest_path)
        if chen_de.shape != daest_de.shape:
            print(f"sub{sub:03d}: shape mismatch {chen_de.shape} vs {daest_de.shape}, skipped")
            skipped.append(sub)
            continue

        for b in range(N_BANDS):
            a = chen_de[:, :, :, b].flatten()
            d = daest_de[:, :, :, b].flatten()
            if a.std() < 1e-12 or d.std() < 1e-12:
                continue
            per_sub_per_band[sub, b], _ = pearsonr(a, d)

        a_all = chen_de.flatten()
        d_all = daest_de.flatten()
        if a_all.std() > 1e-12 and d_all.std() > 1e-12:
            per_sub_overall[sub], _ = pearsonr(a_all, d_all)

    valid_subs = [s for s in range(N_SUBS) if s not in skipped]
    print(f"Subjects: {len(valid_subs)}/{N_SUBS}; skipped: {skipped or 'none'}")
    print(f"{'Band':<10} {'Mean r':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
    band_means = []
    for b in range(N_BANDS):
        vals = per_sub_per_band[valid_subs, b]
        vals = vals[~np.isnan(vals)]
        band_means.append(np.mean(vals))
        print(f"{BAND_NAMES[b]:<10} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    overall = per_sub_overall[valid_subs]
    overall = overall[~np.isnan(overall)]
    print(f"{'All bands':<10} {np.mean(overall):>8.4f} {np.std(overall):>8.4f} "
          f"{np.min(overall):>8.4f} {np.max(overall):>8.4f}")

    results = {
        "n_subjects": len(valid_subs),
        "n_channels": N_CHANNELS,
        "overall_mean_r": float(np.mean(overall)),
        "overall_std_r": float(np.std(overall)),
        "per_band_mean_r": {BAND_NAMES[b]: float(band_means[b]) for b in range(N_BANDS)},
        "per_subject_overall_r": {f"sub{s:03d}": float(per_sub_overall[s])
                                  for s in valid_subs if not np.isnan(per_sub_overall[s])},
        "skipped_subjects": skipped,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
