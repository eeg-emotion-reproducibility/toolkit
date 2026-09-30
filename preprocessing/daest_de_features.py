"""Five-band differential entropy (DE) features from DAEST's preprocessed FACED recordings.

Applies the DE procedure of the FACED release (Chen et al., 2023) to the signal released by
DAEST (Shen et al., 2025), so that the two preprocessing pipelines can be compared with the
feature extraction held fixed:
  * the last 30 s of each of the 28 clips are taken, as DAEST's data loader does;
  * each band is band-pass filtered over the whole 30 s (mne.filter.filter_data), the signal is
    cut into 1-s windows, and DE = 0.5 * log(2 * pi * e * variance) per window.
The output uses the layout of the FACED release's DE features: one pickle per subject
(sub000.pkl, ...) holding {"data": array of shape (28 clips, 30 channels, 30 windows, 5 bands)}.

Usage:
    python daest_de_features.py --src /path/to/DAEST/preprocessed --dst /path/to/output
"""
import argparse
import os
import pickle

import mne
import numpy as np
import scipy.io as sio

FS = 125
N_VIDEOS = 28
N_CHANNELS = 30
T_SECONDS = 30
WINDOW_SEC = 1
BANDS_5 = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 30),
    "gamma": (30, 47),
}


def slice_last_30s_per_video(data, n_samp_one_sec, t=T_SECONDS):
    """Return the last t seconds of each clip, shape (clips, channels, t * FS).

    data: (channels, samples), all clips concatenated; n_samp_one_sec: clip durations in seconds.
    """
    n_points = n_samp_one_sec * FS
    n_points_cum = np.cumsum(n_points).astype(int)
    start_points = n_points_cum - t * FS
    win_len = t * FS
    out = np.zeros((N_VIDEOS, data.shape[0], win_len), dtype=np.float32)
    for vid in range(N_VIDEOS):
        s = int(start_points[vid])
        e = s + win_len
        out[vid] = data[:, s:e]
    return out


def compute_de_per_second(eeg, fs=FS, bands=BANDS_5):
    """DE per 1-s window and band for one clip, shape (windows, channels, bands)."""
    n_ch, n_samp = eeg.shape
    win = int(WINDOW_SEC * fs)
    n_seg = n_samp // win
    n_b = len(bands)
    out = np.zeros((n_seg, n_ch, n_b), dtype=np.float32)
    eeg64 = eeg.astype(np.float64)
    for bi, (_, (lo, hi)) in enumerate(bands.items()):
        filtered = mne.filter.filter_data(
            eeg64, sfreq=fs, l_freq=lo, h_freq=hi, verbose=False
        )
        trimmed = filtered[:, : n_seg * win].reshape(n_ch, n_seg, win)
        var = trimmed.var(axis=2)
        de = 0.5 * np.log(2 * np.pi * np.e * np.maximum(var, 1e-10))
        out[:, :, bi] = de.T.astype(np.float32)
    return out


def process_subject(src_path, sub_id):
    """DE features of one subject, shape (28, 30, 30, 5)."""
    fn = os.path.join(src_path, f"sub{sub_id:03d}.mat")
    if not os.path.exists(fn):
        for cand in sorted(os.listdir(src_path)):
            if cand.startswith(f"sub{sub_id:03d}"):
                fn = os.path.join(src_path, cand)
                break
    mat = sio.loadmat(fn)
    data = mat["data_all_cleaned"]
    n_samp_one = mat["n_samples_one"][0].astype(int)
    assert data.shape[0] == N_CHANNELS, (
        f"sub{sub_id}: expected {N_CHANNELS} channels, got {data.shape[0]}"
    )
    assert n_samp_one.shape[0] == N_VIDEOS, (
        f"sub{sub_id}: expected {N_VIDEOS} videos, got {n_samp_one.shape[0]}"
    )
    last30 = slice_last_30s_per_video(data, n_samp_one)
    de_all = np.zeros((N_VIDEOS, N_CHANNELS, T_SECONDS, len(BANDS_5)), dtype=np.float32)
    for vid in range(N_VIDEOS):
        de = compute_de_per_second(last30[vid])
        de_all[vid] = de.transpose(1, 0, 2)
    return de_all


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True,
                        help="folder with DAEST's preprocessed recordings (subNNN*.mat)")
    parser.add_argument("--dst", required=True, help="output folder for subNNN.pkl")
    parser.add_argument("--n_subs", type=int, default=123)
    parser.add_argument("--start", type=int, default=0, help="first subject index")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for s in range(args.start, args.n_subs):
        out_fn = os.path.join(args.dst, f"sub{s:03d}.pkl")
        if os.path.exists(out_fn):
            print(f"sub{s:03d}: exists, skipped")
            continue
        try:
            de = process_subject(args.src, s)
        except Exception as exc:
            print(f"sub{s:03d}: failed ({exc})")
            continue
        with open(out_fn, "wb") as f:
            pickle.dump({"data": de}, f)
        print(f"sub{s:03d}: DE {de.shape} -> {out_fn}")


if __name__ == "__main__":
    main()
