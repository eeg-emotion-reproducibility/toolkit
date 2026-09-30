"""FACED differential-entropy (DE) features for DGCNN.

Input: FACED's released processed data, ``Processed_data/subNNN.pkl`` (NNN =
000..122), each an array of shape (28 trials, 32 channels, 7500 samples) at
250 Hz. All channels are kept and no resampling is applied.

For every non-overlapping window (default 1 s = 250 samples) the power
spectral density is estimated with Welch's method and, for each of five bands
(delta 1-4, theta 4-8, alpha 8-14, beta 14-31, gamma 31-50 Hz),
DE = 0.5 * log(mean band power).

Output: ``<output-dir>/subNNN.npy`` of shape (28, 30, 32, 5) float32
(trials x windows x channels x bands) and ``EXTRACTION_META.json``.

Usage:
    python faced_features.py --input-dir /path/to/FACED/Processed_data \
        --output-dir /path/to/faced_de
"""

import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
from scipy.signal import welch

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 14.0),
    'beta':  (14.0, 31.0),
    'gamma': (31.0, 50.0),
}


def compute_de_per_window(trial_data, sfreq, window_samples):
    """Per-window DE for one trial.

    Args:
        trial_data: (n_channels, n_samples)
        sfreq: sampling frequency (Hz)
        window_samples: window length in samples

    Returns:
        de: (n_windows, n_channels, n_bands) float32
    """
    n_channels, n_samples = trial_data.shape
    n_windows = n_samples // window_samples
    n_bands = len(BANDS)

    de = np.zeros((n_windows, n_channels, n_bands), dtype=np.float32)

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        seg = trial_data[:, start:end]

        # nperseg = min(window, 1 s) gives about 1 Hz resolution.
        nperseg = min(window_samples, int(sfreq))
        noverlap = nperseg // 2
        freqs, psd = welch(seg, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)

        for b, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
            band_mask = (freqs >= fmin) & (freqs < fmax)
            if band_mask.sum() == 0:
                de[w, :, b] = 0.0
            else:
                band_power = np.mean(psd[:, band_mask], axis=1)
                de[w, :, b] = 0.5 * np.log(np.maximum(band_power, 1e-12))

    return de


def extract_one_subject(pkl_path, sfreq, window_sec):
    """DE features for one subject's pickle.

    Returns:
        de: (n_trials, n_windows, n_channels, n_bands) float32
        meta: dict with shape information
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)  # (28, 32, 7500)

    if not isinstance(data, np.ndarray):
        raise TypeError(f'{pkl_path}: expected ndarray, got {type(data)}')
    if data.ndim != 3:
        raise ValueError(f'{pkl_path}: expected (trials, channels, samples), got shape {data.shape}')

    n_trials, n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    n_windows_per_trial = n_samples // window_samples

    if n_windows_per_trial < 1:
        raise ValueError(f'{pkl_path}: trial too short for {window_sec}s @ {sfreq}Hz '
                         f'(n_samples={n_samples}, window_samples={window_samples})')

    de_all = np.zeros((n_trials, n_windows_per_trial, n_channels, len(BANDS)),
                      dtype=np.float32)
    for t in range(n_trials):
        de_all[t] = compute_de_per_window(data[t], sfreq, window_samples)

    meta = {
        'n_trials': n_trials,
        'n_windows_per_trial': n_windows_per_trial,
        'n_channels': n_channels,
        'n_bands': len(BANDS),
        'window_samples': window_samples,
        'sfreq': sfreq,
        'input_shape': list(data.shape),
        'output_shape': list(de_all.shape),
    }
    return de_all, meta


def main():
    p = argparse.ArgumentParser(description='FACED DE features (Welch PSD, 5 bands).')
    p.add_argument('--input-dir', required=True,
                   help='FACED Processed_data directory (subNNN.pkl).')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sfreq', type=float, default=250.0)
    p.add_argument('--window-sec', type=float, default=1.0)
    p.add_argument('--n-subs', type=int, default=123)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'FACED DE extraction ({args.sfreq} Hz, {args.window_sec} s windows)')

    subject_shapes = {}
    sample_sha = None
    t0 = time.time()

    for sub_id in range(args.n_subs):
        pkl_path = os.path.join(args.input_dir, f'sub{sub_id:03d}.pkl')
        if not os.path.isfile(pkl_path):
            print(f'[warn] sub {sub_id:03d}: {pkl_path} missing; skipping')
            continue

        out_path = os.path.join(args.output_dir, f'sub{sub_id:03d}.npy')
        if os.path.isfile(out_path):
            print(f'[skip] sub {sub_id:03d}: {out_path} already exists')
            continue

        de, meta = extract_one_subject(pkl_path, args.sfreq, args.window_sec)
        np.save(out_path, de)
        subject_shapes[f'sub{sub_id:03d}'] = meta['output_shape']

        # Hash of the first extracted subject, for comparing re-extractions.
        if sample_sha is None:
            sample_sha = hashlib.sha256(de.tobytes()).hexdigest()[:16]

        if sub_id % 10 == 0 or sub_id == args.n_subs - 1:
            print(f'  sub {sub_id:03d}: shape={de.shape} ({time.time() - t0:.1f}s)')

    meta_record = {
        'sfreq': args.sfreq,
        'window_sec': args.window_sec,
        'window_samples': int(args.window_sec * args.sfreq),
        'bands': {k: list(v) for k, v in BANDS.items()},
        'n_subjects_extracted': len(subject_shapes),
        'subject_shapes': subject_shapes,
        'sample_sha256_first_subject': sample_sha,
    }
    meta_path = os.path.join(args.output_dir, 'EXTRACTION_META.json')
    with open(meta_path, 'w') as f:
        json.dump(meta_record, f, indent=2)
    print(f'{len(subject_shapes)} subjects extracted in {time.time() - t0:.1f}s; meta: {meta_path}')


if __name__ == '__main__':
    main()
