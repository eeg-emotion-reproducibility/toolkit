"""Re-preprocess DEAP from the raw BDF recordings.

Input: DEAP's raw recordings ``s01.bdf`` .. ``s32.bdf`` (``data_original``),
512 Hz, 48 channels (32 EEG, peripheral channels, trigger channel).

Output, per subject (XX = 00..31), read by ``deap_features.py``:
  ``sub_XX.pkl``        pickled float64 array (n_trials, 32, 15000): 60-s trials
                        at 250 Hz, in the order the trials were played to the
                        subject (playback order).
  ``sub_XX_meta.json``  per-trial bad channels, number of removed ICA
                        components and errors, and ``valid_playback_indices``,
                        the playback position (0..39) of each trial in the pkl.

Steps, per trial:
  1. Epoch 0-60 s from the trial-onset marker.
  2. Pick the 32 EEG channels by name (Twente list for s01-s22, Geneva list
     for s23-s32).
  3. Resample to 250 Hz.
  4. Band-pass 1-47 Hz (FIR, firwin), so the 1-4 Hz band is kept.
  5. Detect bad channels (amplitude criteria) and interpolate them; Fp1, Fp2,
     F7 and F8 are not interpolated in this pass.
  6. Common average reference, extended Infomax ICA (floor(0.78 x number of
     channels) components, random_state 42) and ICLabel; up to 4 components
     whose label is neither 'brain' nor 'other' are removed.
  7. Detect bad channels again and interpolate them (all channels eligible).
  8. Keep the first 15000 samples.

Usage:
    python deap_raw_preprocessing.py --input-dir /path/to/data_original \
        --output-dir /path/to/deap_250hz
"""

import argparse
import os
import json
import pickle
import logging

import numpy as np
import mne

mne.set_log_level("ERROR")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SFREQ_ORIGINAL = 512
TARGET_SFREQ = 250
BANDPASS_LOW = 1
BANDPASS_HIGH = 47
ICA_COMPONENT_RATIO = 0.78
ICA_MAX_EXCLUDE = 4
BAD_CHANNEL_THRESHOLDS = [(3, 0.4), (30, 0.01)]
N_SUBS = 32
N_TRIALS = 40
MIN_ACCEPTABLE_TRIALS = 30  # fewer recovered trials than this: raise
TRIAL_SECONDS = 60
N_CHANNELS = 32

# EEG channel names in the order of the two recording sites.
EEG_NAMES_TWENTE = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7',
    'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3',
    'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',
]

EEG_NAMES_GENEVA = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3',
    'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
    'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
]

# Frontal channels left out of the first interpolation pass (AF7 and AF8 are
# not DEAP channels and are dropped at the call site).
FRONTAL_EXCLUDE = ['Fp1', 'Fp2', 'F7', 'F8', 'AF7', 'AF8']


def deap_channel_names(sub_idx):
    """EEG channel names for a DEAP subject.

    Subjects 0-21 (s01-s22) were recorded in Twente, 22-31 (s23-s32) in Geneva.
    """
    return EEG_NAMES_TWENTE if sub_idx < 22 else EEG_NAMES_GENEVA


def detect_bad_channels(raw, thresholds=None):
    """Detect bad EEG channels with amplitude criteria.

    A channel is bad if, for any (multiplier, proportion) pair, more than
    ``proportion`` of its samples exceed ``multiplier`` times its median
    absolute amplitude.

    Returns:
        list of bad channel names
    """
    if thresholds is None:
        thresholds = [(3, 0.4), (30, 0.01)]

    data = raw.get_data(picks='eeg')
    total_samples = data.shape[1]
    bad_channels = set()

    for a, b in thresholds:
        for ch_idx, ch_data in enumerate(data):
            median = np.median(np.abs(ch_data))
            high_ratio = np.sum(np.abs(ch_data) > (a * median)) / total_samples
            if high_ratio > b:
                bad_channels.add(raw.info['ch_names'][ch_idx])

    return list(bad_channels)


def ica_denoise_iclabel(raw, component_ratio=0.78, max_exclude=4, random_state=42):
    """Common average reference, ICA and ICLabel-based component removal.

    The average reference (computed over the channels not marked bad) is
    applied here because ICLabel expects average-referenced data; the ICA is
    fitted on the channels not marked bad.
    Components are excluded when their ICLabel label is neither 'brain' nor
    'other', at most ``max_exclude`` of them, taken in component order.

    Returns:
        (n_excluded, cleaned_raw)
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    raw = raw.set_eeg_reference("average")
    n_components = int(np.floor(len(raw.ch_names) * component_ratio))
    ica = ICA(
        n_components=n_components,
        max_iter="auto",
        random_state=random_state,
        method='infomax',
        fit_params=dict(extended=True),
    )
    ica.fit(raw)
    raw.load_data()

    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels)
        if label not in ["brain", "other"]
    ]
    exclude_idx = exclude_idx[:max_exclude]

    raw_clean = raw.copy()
    ica.apply(raw_clean, exclude=exclude_idx)
    return len(exclude_idx), raw_clean


def find_events_deap(raw_bdf, sub_idx):
    """Find the trigger events of a DEAP recording.

    The trigger channel is looked up by the label it carries in each group of
    files: the default for subjects 0-22 (s01-s23), '' for 23-27 (s24-s28)
    and '-1' for 28-31 (s29-s32).
    """
    if sub_idx <= 22:
        events = mne.find_events(raw_bdf)
    elif sub_idx <= 27:
        events = mne.find_events(raw_bdf, stim_channel='')
    else:
        events = mne.find_events(raw_bdf, stim_channel='-1')
    return events


def extract_epochs_deap(raw_bdf, events):
    """Epoch the trials (0-60 s from the trial-onset marker).

    The onset code is 4, stored with the upper status bits set
    (1638144 + 4) in some recordings and without them in others; both codes
    are tried and the one that yields more epochs is kept. A recording with
    fewer than N_TRIALS markers is accepted down to MIN_ACCEPTABLE_TRIALS;
    identify_valid_playback_indices() then locates the missing trials.
    """
    best_epochs = None
    best_event_id = None
    best_n = 0

    for event_id in [1638144 + 4, 4]:
        try:
            epochs = mne.Epochs(
                raw_bdf, events,
                event_id=event_id,
                tmin=0, tmax=TRIAL_SECONDS,
                baseline=(0, 0),
                preload=True,
            )
            if len(epochs) > best_n:
                best_n = len(epochs)
                best_epochs = epochs
                best_event_id = event_id
        except Exception:
            continue

    if best_n == 0:
        raise RuntimeError(
            f"No usable events found with event_id ∈ {{1638148, 4}}"
        )
    if best_n < MIN_ACCEPTABLE_TRIALS:
        raise RuntimeError(
            f"Only {best_n} epochs recovered (below MIN_ACCEPTABLE_TRIALS={MIN_ACCEPTABLE_TRIALS}) "
            f"— likely systemic recording issue, not salvageable"
        )
    if best_n < N_TRIALS:
        log.warning(
            f"  Partial session: {best_n}/{N_TRIALS} trials "
            f"(event_id={best_event_id}) — identifying missing playback positions from gaps"
        )

    return best_epochs, best_event_id


def identify_valid_playback_indices(epochs, n_trials_expected=N_TRIALS):
    """Playback position (0-based) of each epoch.

    With all trials present this is 0..n_trials_expected-1. Otherwise the
    positions follow from the onset-to-onset intervals: the median interval is
    one trial, and an interval of k median intervals advances the position by
    round(k). The first marker is taken as position 0. In DEAP, s28.bdf holds
    37 onset markers; one long interval places its missing trials at playback
    positions 23-25.
    """
    n_observed = len(epochs)
    if n_observed == n_trials_expected:
        return list(range(n_trials_expected))

    sfreq = epochs.info['sfreq']
    times_s = epochs.events[:, 0] / sfreq
    gaps = np.diff(times_s)
    median_gap = float(np.median(gaps))

    positions = [0]
    for g in gaps:
        step = int(round(float(g) / median_gap))
        positions.append(positions[-1] + step)

    max_pos = positions[-1]
    if max_pos >= n_trials_expected:
        log.warning(
            f"  Gap analysis inferred last position {max_pos} "
            f"≥ n_trials_expected={n_trials_expected}; check for extra events"
        )
    if len(positions) != n_observed:
        log.error(
            f"  Position inference produced {len(positions)} positions "
            f"for {n_observed} observed events (should match)"
        )

    missing = sorted(set(range(max(max_pos + 1, n_trials_expected))) - set(positions))
    log.info(f"  Valid playback positions: {positions[:3]}...{positions[-3:]} "
             f"({n_observed} total); missing: {missing}")
    return positions


def process_one_subject(bdf_path, sub_idx):
    """Process one DEAP subject from its raw BDF file.

    Returns:
        (eeg_data, metadata): eeg_data has shape (n_trials, 32, 15000) at
        250 Hz in playback order.
    """
    metadata = {
        'subject': sub_idx,
        'trials': [],
        'status': 'success',
    }

    raw_bdf = mne.io.read_raw_bdf(bdf_path, preload=False)

    events = find_events_deap(raw_bdf, sub_idx)
    log.info(f"  Found {len(events)} events")

    epochs, event_id_used = extract_epochs_deap(raw_bdf, events)
    metadata['event_id_used'] = int(event_id_used)
    metadata['n_epochs'] = len(epochs)
    log.info(f"  Extracted {len(epochs)} epochs (event_id={event_id_used})")

    valid_playback_indices = identify_valid_playback_indices(epochs, N_TRIALS)
    metadata['valid_playback_indices'] = [int(i) for i in valid_playback_indices]
    metadata['n_valid_trials'] = len(valid_playback_indices)
    if len(valid_playback_indices) < N_TRIALS:
        missing = sorted(set(range(N_TRIALS)) - set(valid_playback_indices))
        metadata['missing_playback_indices'] = [int(i) for i in missing]
        log.warning(f"  Missing playback positions (0-idx): {missing}")

    eeg_ch_names = deap_channel_names(sub_idx)
    n_expected_samples = TARGET_SFREQ * TRIAL_SECONDS  # 15000

    eeg_trials = []
    for trial_idx in range(len(epochs)):
        trial_meta = {
            'trial': trial_idx,
            'bad_channels_1st': [],
            'bad_channels_2nd': [],
            'n_ica_excluded': 0,
            'valid': True,
        }

        try:
            trial_data = epochs[trial_idx].get_data()[0]  # (n_ch, n_samples)
            info = mne.create_info(
                ch_names=epochs.ch_names,
                sfreq=SFREQ_ORIGINAL,
                ch_types='eeg',
            )
            trial_raw = mne.io.RawArray(trial_data, info)

            # Select EEG channels (Twente or Geneva order)
            trial_raw.pick_channels(eeg_ch_names)

            trial_raw.resample(TARGET_SFREQ)

            trial_raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design='firwin')

            montage = mne.channels.make_standard_montage('standard_1020')
            trial_raw.set_montage(montage, on_missing='warn')

            # First interpolation pass. Fp1, Fp2, F7 and F8 are not
            # interpolated here; a flagged one stays marked bad, so it is left
            # out of the average reference and the ICA until the second pass
            # sets a new list of bad channels.
            bad_chs_1 = detect_bad_channels(trial_raw, BAD_CHANNEL_THRESHOLDS)
            trial_meta['bad_channels_1st'] = bad_chs_1
            if bad_chs_1:
                trial_raw.info['bads'] = bad_chs_1
                trial_raw.interpolate_bads(
                    reset_bads=True,
                    exclude=[ch for ch in FRONTAL_EXCLUDE if ch in trial_raw.ch_names],
                )

            n_excluded, trial_raw = ica_denoise_iclabel(
                trial_raw,
                component_ratio=ICA_COMPONENT_RATIO,
                max_exclude=ICA_MAX_EXCLUDE,
            )
            trial_meta['n_ica_excluded'] = n_excluded

            # Second interpolation pass, all channels eligible.
            bad_chs_2 = detect_bad_channels(trial_raw, BAD_CHANNEL_THRESHOLDS)
            trial_meta['bad_channels_2nd'] = bad_chs_2
            if bad_chs_2:
                trial_raw.info['bads'] = bad_chs_2
                trial_raw.interpolate_bads(reset_bads=True)

            # The average reference was applied inside ica_denoise_iclabel().

            trial_eeg = trial_raw.get_data()  # (32, n_samples)

            if trial_eeg.shape[1] >= n_expected_samples:
                trial_eeg = trial_eeg[:, :n_expected_samples]
            else:
                log.warning(
                    f"  Trial {trial_idx}: short "
                    f"({trial_eeg.shape[1]} < {n_expected_samples}), padding"
                )
                padded = np.zeros((N_CHANNELS, n_expected_samples))
                padded[:, :trial_eeg.shape[1]] = trial_eeg
                trial_eeg = padded

            eeg_trials.append(trial_eeg)

        except Exception as e:
            log.error(f"  Trial {trial_idx} failed: {e}")
            trial_meta['valid'] = False
            trial_meta['error'] = str(e)
            eeg_trials.append(np.zeros((N_CHANNELS, n_expected_samples)))

        metadata['trials'].append(trial_meta)

    # Trials without an onset marker are absent from the array; the positions
    # of the stored trials are in metadata['valid_playback_indices']. A trial
    # whose processing failed is stored as zeros and marked 'valid': False.
    eeg_data = np.array(eeg_trials)  # (n_trials, 32, 15000)
    return eeg_data, metadata


def main():
    parser = argparse.ArgumentParser(
        description='DEAP re-preprocessing from the raw BDF recordings'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help="Directory with DEAP's raw recordings s01.bdf .. s32.bdf.",
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory for sub_XX.pkl and sub_XX_meta.json. A subject whose '
             'sub_XX.pkl already exists here is skipped.',
    )
    parser.add_argument(
        '--start-sub', type=int, default=0,
        help='Start from this subject index (0-indexed, for resuming)',
    )
    parser.add_argument(
        '--end-sub', type=int, default=32,
        help='End at this subject index (exclusive)',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Processing subjects {args.start_sub} to {args.end_sub - 1}")
    log.info(f"Input: {args.input_dir}")
    log.info(f"Output: {args.output_dir}")

    for sub_idx in range(args.start_sub, min(args.end_sub, N_SUBS)):
        bdf_path = os.path.join(args.input_dir, f's{sub_idx + 1:02}.bdf')

        if not os.path.exists(bdf_path):
            log.error(f"[{sub_idx+1}/32] s{sub_idx+1:02}.bdf not found, skipping")
            continue

        out_pkl = os.path.join(args.output_dir, f'sub_{sub_idx:02}.pkl')
        if os.path.exists(out_pkl):
            log.info(f"[{sub_idx+1}/32] sub_{sub_idx:02}: already processed, skipping")
            continue

        log.info(f"[{sub_idx+1}/32] s{sub_idx+1:02}.bdf")

        try:
            eeg_data, metadata = process_one_subject(bdf_path, sub_idx)

            with open(out_pkl, 'wb') as f:
                pickle.dump(eeg_data, f)
            log.info(f"  Saved: {eeg_data.shape}")

        except Exception as e:
            log.error(f"  FAILED: {e}")
            metadata = {
                'subject': sub_idx,
                'status': f'error: {e}',
                'trials': [],
            }

        meta_path = os.path.join(args.output_dir, f'sub_{sub_idx:02}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    log.info("Done.")


if __name__ == '__main__':
    main()
