# DGCNN baseline

Code for the DGCNN (Song et al., 2018) results on DEAP, SEED and FACED.

| File | Purpose |
|---|---|
| `dgcnn.py` | DGCNN model (learnable adjacency, Chebyshev order 2, 64 hidden units, dropout 0.5) |
| `common.py` | validation-subject draw, best-validation-loss weights, metrics |
| `deap_features.py` | DEAP differential-entropy (DE) features, 5 bands, running normalisation, LDS smoothing, z-score |
| `train_deap.py` | DEAP leave-one-subject-out and subject-dependent 10-fold (trial or temporal split), label-permutation control |
| `train_seed.py` | SEED leave-one-subject-out, 3 classes |
| `faced_features.py` | FACED DE features (Welch PSD, 5 bands, 1-s windows) |
| `train_faced.py` | FACED binary valence, 10-fold cross-subject, stimulus or self-reported labels |
| `deap_raw_preprocessing.py` | DEAP raw BDF recordings to the 250-Hz trial pickles read by `deap_features.py` |

## Requirements

Python 3.10 and the packages the scripts import. The code was checked with:

| Package | Version |
|---|---|
| torch | 2.5.1 |
| torcheeg | 1.1.3, installed from source (`pip install -e ./torcheeg --no-deps`; PyPI has 1.1.2) |
| pytorch-lightning | 2.6.1 |
| torchmetrics | 1.9.0 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| mne | 1.11.0 (DEAP band-pass filtering) |

TorchEEG is used only for its `ClassifierTrainer` (Adam, cross-entropy) in
`train_deap.py` and `train_seed.py`.
`deap_features.py` uses MNE's default FIR filter settings, so a different MNE
version could change the DEAP features slightly.

Run the scripts from this folder (they import `dgcnn.py` and `common.py`).

## Data

**DEAP.** `participant_ratings.csv` from the DEAP release gives the labels
(valence or arousal >= 5) and each subject's playback order. The EEG input of
`deap_features.py` is one pickle per subject, `sub_XX.pkl` (XX = 00..31), with
an array (n_trials, 32, 15000): 60-s trials at 250 Hz in playback order. These
were re-preprocessed from DEAP's raw BDF recordings (1-47 Hz band-pass,
resampling to 250 Hz, ICA artefact removal, common average reference), so the
1-4 Hz band is present in the signal. They are **not** DEAP's preprocessed
Python files (128 Hz, 4-45 Hz band-pass), from which no delta band can be
extracted. `deap_raw_preprocessing.py` does the re-preprocessing (see DEAP input below). One
subject's recording holds 37 of the 40 trials; `sub_XX_meta.json` with
`valid_playback_indices` records their positions, and missing trials are
carried as NaN and skipped.

**SEED.** The released extracted-feature folder with 1-s windows
(`ExtractedFeatures_1s`): `<subject>_<date>.mat`, three sessions per subject,
keys `de_LDS1` .. `de_LDS15` of shape (62, T, 5), and `label.mat`.

**FACED.** `Processed_data/subNNN.pkl` (28 trials x 32 channels x 7500 samples,
250 Hz) for `faced_features.py`, and `Data/subNNN/After_remarks.mat` (valence
rating in column 9 of `score`, 0-7 scale) for self-reported labels.

## Feature files

`deap_features.py` writes `de_raw.npy`, `valid_trials.npy` and
`fold_loso/de_final_fold{k}.npy` (or `fold_loso_source/` plus
`validation_subjects.json` with `--selection source`). These names and array
layouts are those of the feature files behind the reported numbers, so existing
feature directories can be passed to `train_deap.py` unchanged.
`faced_features.py` writes `subNNN.npy` of shape (28, 30, 32, 5).

## Checkpoint selection

Every training script takes `--selection`:

* `source` (default): the validation data come only from the training data.
  Leave-one-subject-out and FACED 10-fold: 15 % of the fold's training subjects
  (seed 42 + fold index), excluded from training and, for DEAP, from the
  normalisation statistics (generate the features with
  `deap_features.py --selection source`). Subject-dependent DEAP: one ninth of
  each fold's training trials (trial split) or, with the temporal split, the
  window block after the test block. Early stopping (patience 20, at most 200
  epochs) on the validation loss; the weights with the lowest validation loss
  are evaluated. The test data are not used before the final evaluation.
* `test`: early stopping monitors the loss on the held-out test data and the
  weights at the stopping epoch are evaluated. The DEAP and SEED numbers in the
  paper were obtained this way.
* `fixed_epochs` (FACED only): 50 epochs, no validation set, final-epoch
  weights. The FACED numbers in the paper were obtained this way.

## Reproducing the reported configurations

Paths in capitals are placeholders.

DEAP leave-one-subject-out, valence (4-s windows), and the per-subject label
permutation control:

```
python deap_features.py --input-dir DEAP_250HZ --ratings RATINGS_CSV \
    --output-dir FEAT/deap_de_4s --window-sec 4 --selection test
python train_deap.py --eval-mode loso --target valence --window-sec 4 --selection test \
    --features-dir FEAT/deap_de_4s --ratings RATINGS_CSV --output-dir RUNS
python train_deap.py --eval-mode loso --target valence --window-sec 4 --selection test \
    --shuffle-labels --features-dir FEAT/deap_de_4s --ratings RATINGS_CSV --output-dir RUNS
```

DEAP subject-dependent 10-fold, trial-level and temporal split, valence and
arousal (1-s windows, 60 per trial; only `de_raw.npy` and `valid_trials.npy`
are used):

```
python deap_features.py --input-dir DEAP_250HZ --ratings RATINGS_CSV \
    --output-dir FEAT/deap_de_1s --window-sec 1 --selection test
for TARGET in valence arousal; do
  for SPLIT in trial temporal; do
    python train_deap.py --eval-mode sd --split $SPLIT --target $TARGET --window-sec 1 \
        --selection test --features-dir FEAT/deap_de_1s --ratings RATINGS_CSV --output-dir RUNS
  done
done
```

SEED leave-one-subject-out, 3 classes (batch size 256):

```
python train_seed.py --data-dir SEED_EXTRACTED_FEATURES_1S --output-dir RUNS --selection test
```

FACED binary valence, 10-fold, stimulus labels and self-reported labels
(threshold 3.0):

```
python faced_features.py --input-dir FACED/Processed_data --output-dir FEAT/faced_de
python train_faced.py --data-dir FEAT/faced_de --output-dir RUNS \
    --label-source stim --selection fixed_epochs
python train_faced.py --data-dir FEAT/faced_de --output-dir RUNS \
    --label-source self_report --ratings-dir FACED/Data --sam-threshold 3.0 --selection fixed_epochs
```

Drop `--selection ...` (or pass `--selection source`) to obtain the
source-validated result for the same configuration; for DEAP
leave-one-subject-out, generate the features with `--selection source` first.

## Outputs and randomness

Each run writes one JSON file to `--output-dir` with per-fold (per-subject for
the subject-dependent mode) accuracy and balanced accuracy, their mean and
standard deviation, and the full argument set. The standard deviation is taken
across folds (DEAP leave-one-subject-out, SEED), across subjects' 10-fold means
(DEAP subject-dependent), both with `ddof=0`, and across the 10 folds with
`ddof=1` (FACED); the field `std_ddof` records which. DEAP and SEED also save
per-fold predictions.

`train_deap.py` and `train_seed.py` seed once (42) and run all folds in one
process, so a fold's result depends on the folds before it; `train_faced.py`
reseeds (42) at the start of every fold. Results on a GPU can differ in the
last digits between runs because of non-deterministic kernels.

## DEAP input

`deap_raw_preprocessing.py` produces the `sub_XX.pkl` and `sub_XX_meta.json`
files that `deap_features.py` reads. It needs DEAP's raw recordings,
`s01.bdf` .. `s32.bdf` (the `data_original` folder of the DEAP release):

```
python deap_raw_preprocessing.py --input-dir DEAP_DATA_ORIGINAL --output-dir DEAP_250HZ
```

For each 60-s trial the script picks the 32 EEG channels, resamples to 250 Hz,
band-pass filters at 1-47 Hz, and interpolates bad channels twice, before and
after artefact removal. The artefact removal uses a common average reference,
extended Infomax ICA (seed 42) and ICLabel, and removes up to four components
that ICLabel labels neither brain nor other. A subject whose `sub_XX.pkl`
already exists in `--output-dir` is skipped, so start with an empty folder;
`--start-sub` and `--end-sub` select a range of subjects.

Requirements: `mne` (see the table above) and `mne-icalabel` with one of its
network back-ends (`pip install "mne-icalabel[torch]"` or
`"mne-icalabel[onnx]"`). The
resampling, filtering, interpolation and ICA come from MNE, so a different MNE
version can change the output.

The DEAP DGCNN results use this re-preprocessing of the raw recordings, not
DEAP's `data_preprocessed_python` files.
