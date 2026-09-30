"""Spectral content below 4 Hz in DEAP's preprocessed files.

DEAP's preprocessed data (data_preprocessed_python) are band-pass filtered at 4-45 Hz, so a
"delta" band (1-4 Hz) extracted from these files contains only filter roll-off. For each
subject the script averages the Welch power spectral density over the 40 trials and 32 EEG
channels, reports the mean power in five bands and the delta-to-theta power ratio, and can
plot the spectra of four subjects.

Usage:
    python deap_delta_band.py --data-dir /path/to/data_preprocessed_python [--figure psd.pdf]
"""
import argparse
import os
import pickle

import numpy as np
from scipy.signal import welch

FS = 128
BANDS = {
    "delta (1-4 Hz)": (1, 4),
    "theta (4-8 Hz)": (4, 8),
    "alpha (8-14 Hz)": (8, 14),
    "beta (14-31 Hz)": (14, 31),
    "gamma (31-45 Hz)": (31, 45),
}


def subject_psd(data_dir, subject):
    with open(os.path.join(data_dir, f"s{subject:02d}.dat"), "rb") as f:
        data = pickle.load(f, encoding="latin1")
    eeg = data["data"][:, :32, :]
    freqs, psd = welch(eeg.reshape(-1, eeg.shape[-1]), fs=FS, nperseg=256, noverlap=128)
    return freqs, psd.mean(axis=0)


def band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return psd[mask].mean()


def plot_spectra(spectra, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    for idx, (ax, (subject, (freqs, psd))) in enumerate(zip(axes.flatten(), spectra.items())):
        ax.semilogy(freqs, psd, "k-", linewidth=0.8)
        for (name, (lo, hi)), color in zip(BANDS.items(), colors):
            mask = (freqs >= lo) & (freqs <= hi)
            ax.fill_between(freqs[mask], psd[mask], alpha=0.3, color=color, label=name)
        ax.axvline(x=4, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Subject {subject}", fontsize=9)
        ax.set_xlim(0, 50)
        if idx >= 2:
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("PSD (uV^2/Hz)", fontsize=9)
    axes[0, 0].legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 33)))
    parser.add_argument("--figure", help="save the spectra of --figure-subjects to this file")
    parser.add_argument("--figure-subjects", type=int, nargs=4, default=[1, 10, 20, 32])
    args = parser.parse_args()

    spectra = {s: subject_psd(args.data_dir, s) for s in args.subjects}
    powers = {name: np.array([band_power(f, p, lo, hi) for f, p in spectra.values()])
              for name, (lo, hi) in BANDS.items()}

    theta = powers["theta (4-8 Hz)"].mean()
    print(f"{len(spectra)} subjects; mean PSD per band and ratio to theta")
    for name, values in powers.items():
        print(f"{name:<18} {values.mean():>10.4f} {values.mean() / theta:>8.4f}")
    ratios = powers["delta (1-4 Hz)"] / powers["theta (4-8 Hz)"]
    print(f"Per-subject delta/theta ratio: mean {ratios.mean():.4f}, SD {ratios.std():.4f}, "
          f"range {ratios.min():.4f}-{ratios.max():.4f}")

    if args.figure:
        missing = [s for s in args.figure_subjects if s not in spectra]
        spectra.update({s: subject_psd(args.data_dir, s) for s in missing})
        plot_spectra({s: spectra[s] for s in args.figure_subjects}, args.figure)
        print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
