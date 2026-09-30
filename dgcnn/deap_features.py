"""DEAP differential-entropy (DE) features for DGCNN.

Input: one pickle per subject, ``sub_XX.pkl`` (XX = 00..31), holding an array
of shape (n_trials, 32, 15000): 60-s trials at 250 Hz, 32 EEG channels, in the
order the trials were played to the subject (playback order). These are
re-preprocessed from DEAP's raw BDF recordings (see README_DGCNN.md). An
optional ``sub_XX_meta.json`` with ``valid_playback_indices`` marks which
playback positions the trials occupy when a subject has fewer than 40 trials.

Pipeline:
  1. DE per channel and band (5 bands, 1-47 Hz) in non-overlapping windows of
     ``--window-sec`` seconds, written to ``de_raw.npy`` (playback order)
     together with ``valid_trials.npy`` (subjects x playback positions).
  2. For every leave-one-subject-out fold:
       running normalisation per subject (decay 0.990), initialised with the
       mean and variance of the fold's training subjects;
       reorder playback -> canonical video order (participant_ratings.csv);
       forward Kalman (LDS) smoothing within each trial;
       per-subject z-score.
     written to ``fold_loso/de_final_fold{k}.npy`` (``--selection test``) or
     ``fold_loso_source/de_final_fold{k}.npy`` (``--selection source``). With
     ``source`` the validation subjects used by ``train_deap.py`` are also
     excluded from the normalisation statistics.

Missing trials are kept as NaN end to end and skipped by every step.

Usage:
    python deap_features.py --input-dir /path/to/deap_250hz \
        --ratings /path/to/participant_ratings.csv \
        --output-dir /path/to/de_features_4s --window-sec 4 --selection test
"""

import argparse
import json
import logging
import os
import pickle

import mne
import numpy as np

from common import pick_validation_subjects

mne.set_log_level("ERROR")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

FS = 250
N_SUBS = 32
N_TRIALS = 40
TRIAL_SECONDS = 60
N_CHANNELS = 32
DE_WINDOW_SEC = 1
DE_WINDOW_SAMPLES = FS * DE_WINDOW_SEC
N_TIMESTEPS_PER_TRIAL = TRIAL_SECONDS // DE_WINDOW_SEC
N_TIMESTEPS_TOTAL = N_TRIALS * N_TIMESTEPS_PER_TRIAL
FREQS = [[1, 4], [4, 8], [8, 14], [14, 30], [30, 47]]
N_BANDS = len(FREQS)
N_FEATURES = N_CHANNELS * N_BANDS
DECAY_RATE = 0.990


def set_window(window_sec):
    """Set the DE window length (1 s -> 60 windows per trial, 4 s -> 15)."""
    global DE_WINDOW_SEC, DE_WINDOW_SAMPLES, N_TIMESTEPS_PER_TRIAL, N_TIMESTEPS_TOTAL
    DE_WINDOW_SEC = window_sec
    DE_WINDOW_SAMPLES = FS * DE_WINDOW_SEC
    N_TIMESTEPS_PER_TRIAL = TRIAL_SECONDS // DE_WINDOW_SEC
    N_TIMESTEPS_TOTAL = N_TRIALS * N_TIMESTEPS_PER_TRIAL


def extract_de_all_subjects(input_dir):
    """Extract DE features for all subjects.

    Returns:
        de_flat: (32, N_TIMESTEPS_TOTAL, 160), playback order, band-major
            feature layout [b0_ch0 .. b0_ch31, b1_ch0, ..]. Missing playback
            positions are NaN.
        valid_trials: (32, 40) bool, True at recovered playback positions.
    """
    de = np.full(
        (N_SUBS, N_CHANNELS, N_TIMESTEPS_TOTAL, N_BANDS),
        np.nan, dtype=np.float64,
    )
    valid_trials = np.zeros((N_SUBS, N_TRIALS), dtype=bool)

    for sub in range(N_SUBS):
        pkl_path = os.path.join(input_dir, f'sub_{sub:02}.pkl')
        meta_path = os.path.join(input_dir, f'sub_{sub:02}_meta.json')
        if not os.path.exists(pkl_path):
            log.warning(f"  sub_{sub:02}.pkl not found; all trials left as NaN")
            continue

        with open(pkl_path, 'rb') as f:
            data_sub = pickle.load(f)  # (n_valid, 32, 15000)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            valid_playback_indices = meta.get(
                'valid_playback_indices', list(range(data_sub.shape[0]))
            )
        else:
            valid_playback_indices = list(range(data_sub.shape[0]))

        if len(valid_playback_indices) != data_sub.shape[0]:
            raise RuntimeError(
                f"sub_{sub:02}: metadata lists {len(valid_playback_indices)} "
                f"valid trials but pkl contains {data_sub.shape[0]}"
            )

        missing = sorted(set(range(N_TRIALS)) - set(valid_playback_indices))
        msg = f" (missing playback positions: {missing})" if missing else ""
        log.info(
            f"  DE extraction: sub {sub:02} shape={data_sub.shape}, "
            f"valid={len(valid_playback_indices)}/{N_TRIALS}{msg}"
        )

        for band_idx, (low, high) in enumerate(FREQS):
            for local_idx, playback_pos in enumerate(valid_playback_indices):
                data_trial = data_sub[local_idx, :, :]  # (32, 15000)

                data_filt = mne.filter.filter_data(
                    data_trial.astype(np.float64), FS,
                    l_freq=low, h_freq=high,
                    verbose=False,
                )

                # (32, N_TIMESTEPS_PER_TRIAL, DE_WINDOW_SAMPLES)
                data_filt = data_filt.reshape(
                    N_CHANNELS, -1, DE_WINDOW_SAMPLES
                )

                # DE of a Gaussian signal: 0.5 * log(2 * pi * e * variance)
                de_trial = 0.5 * np.log(
                    2 * np.pi * np.exp(1) * np.var(data_filt, axis=2)
                )

                start = playback_pos * N_TIMESTEPS_PER_TRIAL
                end = (playback_pos + 1) * N_TIMESTEPS_PER_TRIAL
                de[sub, :, start:end, band_idx] = de_trial

        for p in valid_playback_indices:
            valid_trials[sub, p] = True

    de_flat = de.transpose(0, 2, 3, 1).reshape(
        N_SUBS, N_TIMESTEPS_TOTAL, N_FEATURES
    )
    return de_flat, valid_trials


def load_deap_playback_order(ratings_csv):
    """Per-subject playback order from DEAP's participant_ratings.csv.

    Returns:
        playback_to_canonical: (32, 40) int, the canonical video index
            (0-based Experiment_id) shown at each playback position.
    """
    import pandas as pd
    df = pd.read_csv(ratings_csv)

    playback_to_canonical = np.zeros((N_SUBS, N_TRIALS), dtype=int)

    for pid in df['Participant_id'].unique():
        sub_idx = pid - 1
        if sub_idx >= N_SUBS:
            continue
        sub_df = df[df['Participant_id'] == pid].sort_values('Trial')
        exp_ids = sub_df['Experiment_id'].values
        playback_to_canonical[sub_idx, :] = exp_ids - 1

    return playback_to_canonical


def reorder_to_canonical(data, playback_to_canonical):
    """Reorder (n_subs, n_timesteps, n_features) from playback to canonical order."""
    n_subs, n_timesteps, n_features = data.shape
    data_canonical = np.zeros_like(data)

    for sub in range(n_subs):
        order = playback_to_canonical[sub]
        data_trials = data[sub].reshape(N_TRIALS, N_TIMESTEPS_PER_TRIAL, n_features)

        canonical_trials = np.zeros_like(data_trials)
        for playback_pos, canonical_idx in enumerate(order):
            canonical_trials[canonical_idx] = data_trials[playback_pos]

        data_canonical[sub] = canonical_trials.reshape(n_timesteps, n_features)

    return data_canonical


def running_norm_session(data, data_mean, data_var, decay_rate, skip_nan=False):
    """Running normalisation of one subject's feature sequence.

    Blends the training-set mean/variance with the subject's running
    statistics; the weight on the training-set statistics decays by
    ``decay_rate`` per time step. With ``skip_nan`` a row containing NaN is
    left as NaN and does not advance the running state or the decay factor.
    """
    data_norm = np.full_like(data, np.nan) if skip_nan else np.zeros_like(data)
    running_sum = np.zeros(data.shape[-1])
    running_square = np.zeros(data.shape[-1])
    decay_factor = 1.0

    for t in range(data.shape[0]):
        x = data[t]
        if skip_nan and np.isnan(x).any():
            continue
        running_sum = running_sum + x
        running_mean = running_sum / (t + 1)
        running_square = running_square + x ** 2
        running_var = (
            (running_square - 2 * running_mean * running_sum) / (t + 1)
            + running_mean ** 2
        )

        curr_mean = decay_factor * data_mean + (1 - decay_factor) * running_mean
        curr_var = decay_factor * data_var + (1 - decay_factor) * np.maximum(running_var, 0)
        decay_factor *= decay_rate

        data_norm[t] = (x - curr_mean) / np.sqrt(curr_var + 1e-5)

    return data_norm


def lds_forward(sequence):
    """Forward Kalman filter (linear dynamical system) over one trial.

    Args:
        sequence: (n_timesteps, n_features)

    Returns:
        Smoothed sequence, same shape.
    """
    ave = np.mean(sequence, axis=0)
    u0 = ave
    X = sequence.T  # (n_features, n_timesteps)

    V0 = 0.01
    A = 1
    T = 0.0001
    C = 1
    sigma = 1

    m, n = X.shape
    P = np.zeros((m, n))
    u = np.zeros((m, n))
    V = np.zeros((m, n))
    K = np.zeros((m, n))

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones(m)
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones(m) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones(m) - K[:, i] * C) * P[:, i - 1]

    return u.T


def process_one_fold(de_data, playback_order, train_subs, fold_idx,
                     output_dir=None):
    """Running norm -> reorder -> LDS -> per-subject z-score for one fold.

    Args:
        de_data: (32, N_TIMESTEPS_TOTAL, 160) raw DE in playback order, NaN at
            missing trials. Non-finite values at non-missing positions are
            clamped to -30 in place, so the change persists for later folds.
        playback_order: (32, 40) playback -> canonical mapping.
        train_subs: subjects whose statistics initialise the running norm.
        fold_idx: fold number, used in the output file name.
        output_dir: directory for ``de_final_fold{fold_idx}.npy``.

    Returns:
        de_final: (32, N_TIMESTEPS_TOTAL, 160) in canonical order.
    """
    de_playback = de_data

    trialview = de_playback.reshape(
        N_SUBS, N_TRIALS, N_TIMESTEPS_PER_TRIAL, N_FEATURES
    )
    missing_trials_playback = np.isnan(trialview).all(axis=(2, 3))
    n_missing = int(missing_trials_playback.sum())
    if n_missing > 0:
        rows = np.argwhere(missing_trials_playback)
        preview = rows[:5].tolist()
        log.info(
            f"  Fold {fold_idx}: {n_missing} missing trial(s) in playback order "
            f"(first: {preview})"
        )

    missing_ts = np.repeat(
        missing_trials_playback, N_TIMESTEPS_PER_TRIAL, axis=1
    )
    missing_tsf = missing_ts[:, :, None]

    # Training-set statistics (NaN-aware).
    train_data = de_playback[train_subs]
    data_mean = np.nanmean(np.nanmean(train_data, axis=1), axis=0)
    data_var = np.nanmean(np.nanvar(train_data, axis=1), axis=0)

    # Clamp degenerate DE values (-inf from log(0), unexpected NaN/inf) but
    # keep the NaN that marks missing trials.
    bad = ~np.isfinite(de_playback) & ~missing_tsf
    n_bad = int(bad.sum())
    if n_bad > 0:
        log.info(
            f"  Fold {fold_idx}: clamping {n_bad} degenerate DE values to -30"
        )
        de_playback[bad] = -30

    log.info(f"  Fold {fold_idx}: running norm (decay={DECAY_RATE})")
    de_norm = np.full_like(de_playback, np.nan)
    for sub in range(N_SUBS):
        de_norm[sub] = running_norm_session(
            de_playback[sub], data_mean, data_var, DECAY_RATE, skip_nan=True,
        )

    de_norm = reorder_to_canonical(de_norm, playback_order)

    # LDS smoothing within each trial; all-NaN (missing) trials stay NaN.
    log.info(f"  Fold {fold_idx}: LDS (forward)")
    de_smooth = np.full_like(de_norm, np.nan)
    for sub in range(N_SUBS):
        for trial in range(N_TRIALS):
            start = trial * N_TIMESTEPS_PER_TRIAL
            end = (trial + 1) * N_TIMESTEPS_PER_TRIAL
            trial_data = de_norm[sub, start:end, :]
            if np.isnan(trial_data).all():
                continue
            de_smooth[sub, start:end, :] = lds_forward(trial_data)

    log.info(f"  Fold {fold_idx}: per-subject z-score")
    de_final = np.full_like(de_smooth, np.nan)
    for sub in range(N_SUBS):
        mean = np.nanmean(de_smooth[sub], axis=0)
        std = np.nanstd(de_smooth[sub], axis=0)
        std[std < 1e-8] = 1e-8
        de_final[sub] = (de_smooth[sub] - mean) / std

    expected_nan = n_missing * N_TIMESTEPS_PER_TRIAL * N_FEATURES
    actual_nan = int(np.isnan(de_final).sum())
    if actual_nan != expected_nan:
        log.warning(
            f"  Fold {fold_idx}: NaN count mismatch; expected {expected_nan} "
            f"(from {n_missing} missing trials), got {actual_nan}"
        )
    if np.isinf(de_final).any():
        log.warning(f"  Fold {fold_idx}: Inf detected in final output")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'de_final_fold{fold_idx}.npy')
        np.save(save_path, de_final)
        log.info(f"  Fold {fold_idx}: saved {de_final.shape} to {save_path}")

    return de_final


def main():
    parser = argparse.ArgumentParser(
        description='DEAP DE features: DE -> running norm -> LDS -> z-score')
    parser.add_argument('--input-dir', required=True,
                        help='Directory with sub_XX.pkl (and optional sub_XX_meta.json).')
    parser.add_argument('--ratings', required=True,
                        help="DEAP's participant_ratings.csv.")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--window-sec', type=int, default=1, choices=[1, 4],
                        help='DE window length: 1 s (60 windows per trial) or 4 s (15).')
    parser.add_argument('--selection', default='source', choices=['source', 'test'],
                        help='source: also exclude the validation subjects of '
                             'train_deap.py --selection source from the normalisation '
                             'statistics (writes fold_loso_source/); test: statistics '
                             'from all non-test subjects (writes fold_loso/).')
    parser.add_argument('--skip-de', action='store_true',
                        help='Reuse de_raw.npy and valid_trials.npy from --output-dir.')
    args = parser.parse_args()

    set_window(args.window_sec)
    os.makedirs(args.output_dir, exist_ok=True)

    de_raw_path = os.path.join(args.output_dir, 'de_raw.npy')
    valid_trials_path = os.path.join(args.output_dir, 'valid_trials.npy')
    if args.skip_de:
        log.info(f"Loading cached DE features from {de_raw_path}")
        de_data = np.load(de_raw_path)
        valid_trials = np.load(valid_trials_path)
    else:
        log.info(f"DE extraction ({N_BANDS} bands, {DE_WINDOW_SEC}s windows, "
                 f"{N_TIMESTEPS_PER_TRIAL} windows/trial)")
        de_data, valid_trials = extract_de_all_subjects(args.input_dir)
        np.save(de_raw_path, de_data)
        np.save(valid_trials_path, valid_trials)
        log.info(f"DE features saved: {de_data.shape} -> {de_raw_path}")
    log.info(f"Valid trials: {int(valid_trials.sum())}/{N_SUBS * N_TRIALS}")

    playback_order = load_deap_playback_order(args.ratings)

    sub_dir = 'fold_loso' if args.selection == 'test' else 'fold_loso_source'
    fold_dir = os.path.join(args.output_dir, sub_dir)
    validation_subjects = {}

    for fold in range(N_SUBS):
        pool = sorted(set(range(N_SUBS)) - {fold})
        if args.selection == 'source':
            val_subs = pick_validation_subjects(pool, fold)
            validation_subjects[str(fold)] = val_subs
            train_subs = sorted(set(pool) - set(val_subs))
        else:
            train_subs = pool
        process_one_fold(de_data, playback_order, train_subs, fold,
                         output_dir=fold_dir)

    if args.selection == 'source':
        with open(os.path.join(fold_dir, 'validation_subjects.json'), 'w') as f:
            json.dump(validation_subjects, f, indent=1)
    log.info(f"All {N_SUBS} folds complete. Output in {fold_dir}/")


if __name__ == '__main__':
    main()
