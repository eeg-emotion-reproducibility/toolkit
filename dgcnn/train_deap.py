"""Train DGCNN on DEAP DE features (binary valence or arousal, threshold >= 5).

Evaluation modes:
  loso  leave-one-subject-out over the 32 subjects, on the per-fold features
        ``fold_loso[_source]/de_final_fold{k}.npy`` from deap_features.py.
  sd    subject-dependent 10-fold CV within each subject, on ``de_raw.npy``
        with a per-subject z-score.
          --split trial     whole trials are assigned to folds;
          --split temporal  every trial's windows are cut into 10 consecutive
                            blocks and fold k tests block k of every trial, so
                            train and test share trials (within-trial leakage).

Checkpoint selection (--selection):
  source (default)  loso: 15 % of the 31 training subjects (seed 42 + fold)
                    form the validation set and are excluded from training and
                    from the normalisation statistics (features written by
                    ``deap_features.py --selection source``).
                    sd/trial: one ninth of the training trials of each fold
                    (seed 42 + fold) form the validation set.
                    sd/temporal: the block following the test block in every
                    trial forms the validation set.
                    Early stopping on the validation loss; the weights with the
                    lowest validation loss are evaluated on the test data.
  test              early stopping monitors the test-data loss and the weights
                    at the stopping epoch are evaluated (the rule behind the
                    reported DEAP numbers).

--shuffle-labels permutes each subject's trial labels (permutation control).

Usage:
    python train_deap.py --eval-mode loso --features-dir /path/to/de_features_4s \
        --window-sec 4 --ratings /path/to/participant_ratings.csv \
        --output-dir runs/deap --selection test
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheeg.trainers import ClassifierTrainer

from common import (BestValLossWeights, balanced_accuracy, majority_baseline,
                    mean_sd, pick_validation_subjects, predict)
from dgcnn import DGCNN

N_SUBS = 32
N_TRIALS = 40
N_TIMESTEPS_PER_TRIAL = 60
N_TIMESTEPS_TOTAL = N_TRIALS * N_TIMESTEPS_PER_TRIAL
N_CHANNELS = 32
N_BANDS = 5
N_FEATURES = N_CHANNELS * N_BANDS

# 32 x 40 trials minus the three trials missing from one subject's recording.
EXPECTED_VALID_COUNT = N_SUBS * N_TRIALS - 3


def load_deap_labels(ratings_csv, target='valence'):
    """Binary labels (rating >= 5.0) from participant_ratings.csv.

    Returns:
        labels: (32, 40) int64 indexed by canonical video (Experiment_id - 1).
        raw_ratings: (32, 40) float.
    """
    import pandas as pd
    df = pd.read_csv(ratings_csv)

    raw_ratings = np.zeros((N_SUBS, N_TRIALS))

    for _, row in df.iterrows():
        sub = int(row['Participant_id']) - 1
        trial = int(row['Trial']) - 1
        exp_id = int(row['Experiment_id']) - 1

        col = 'Valence' if target == 'valence' else 'Arousal'
        raw_ratings[sub, exp_id] = float(row[col])

    labels = (raw_ratings >= 5.0).astype(np.int64)
    return labels, raw_ratings


def load_deap_playback_to_canonical(ratings_csv):
    """(32, 40) int: canonical video index shown at each playback position."""
    import pandas as pd
    df = pd.read_csv(ratings_csv)
    playback_to_canonical = np.zeros((N_SUBS, N_TRIALS), dtype=int)
    for pid in df['Participant_id'].unique():
        sub_idx = int(pid) - 1
        if sub_idx >= N_SUBS:
            continue
        sub_df = df[df['Participant_id'] == pid].sort_values('Trial')
        exp_ids = sub_df['Experiment_id'].values
        playback_to_canonical[sub_idx, :] = exp_ids - 1
    return playback_to_canonical


def load_valid_trials(features_dir, ratings_csv, order='canonical'):
    """(32, 40) bool mask of trials with EEG data, in playback or canonical order."""
    valid_trials_path = os.path.join(features_dir, 'valid_trials.npy')
    if not os.path.exists(valid_trials_path):
        raise FileNotFoundError(
            f"valid_trials.npy not found at {valid_trials_path}; "
            f"run deap_features.py first."
        )
    valid_playback = np.load(valid_trials_path)

    if order == 'playback':
        return valid_playback
    elif order == 'canonical':
        playback_order = load_deap_playback_to_canonical(ratings_csv)
        valid_canonical = np.zeros_like(valid_playback)
        for sub in range(N_SUBS):
            for p, c in enumerate(playback_order[sub]):
                valid_canonical[sub, c] = valid_playback[sub, p]
        return valid_canonical
    else:
        raise ValueError(f"order must be 'playback' or 'canonical', got {order!r}")


def reorder_playback_to_canonical(de_playback, playback_to_canonical):
    """Reorder (n_subs, n_timesteps, n_features) from playback to canonical order."""
    n_subs, n_timesteps, n_features = de_playback.shape
    de_canonical = np.zeros_like(de_playback)
    for sub in range(n_subs):
        order = playback_to_canonical[sub]
        trials = de_playback[sub].reshape(N_TRIALS, N_TIMESTEPS_PER_TRIAL, n_features)
        canonical_trials = np.zeros_like(trials)
        for pb_pos, canonical_idx in enumerate(order):
            canonical_trials[canonical_idx] = trials[pb_pos]
        de_canonical[sub] = canonical_trials.reshape(n_timesteps, n_features)
    return de_canonical


def de_to_samples(de_data, labels, valid_mask=None):
    """Per-window samples from (32, N_TIMESTEPS_TOTAL, 160) canonical-order features.

    Returns:
        X: (N, 32, 5) electrodes x bands; y: (N,) labels; subject_ids: (N,).
    """
    X_list, y_list, sub_list = [], [], []

    for sub in range(N_SUBS):
        for trial in range(N_TRIALS):
            if valid_mask is not None and not valid_mask[sub, trial]:
                continue
            start = trial * N_TIMESTEPS_PER_TRIAL
            end = (trial + 1) * N_TIMESTEPS_PER_TRIAL

            for t in range(start, end):
                # Band-major (160,) -> (5, 32) -> (32, 5)
                feat = de_data[sub, t, :].reshape(N_BANDS, N_CHANNELS).T
                if valid_mask is None and np.isnan(feat).any():
                    continue
                X_list.append(feat)
                y_list.append(labels[sub, trial])
                sub_list.append(sub)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subject_ids = np.array(sub_list, dtype=np.int64)

    return X, y, subject_ids


def build_model():
    return DGCNN(
        in_channels=N_BANDS,
        num_electrodes=N_CHANNELS,
        hid_channels=64,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
    )


def trial_windows(trials):
    return np.concatenate([
        np.arange(t * N_TIMESTEPS_PER_TRIAL, (t + 1) * N_TIMESTEPS_PER_TRIAL)
        for t in trials
    ])


def run_loso(args):
    labels, _ = load_deap_labels(args.ratings, target=args.target)
    if args.shuffle_labels:
        rng = np.random.RandomState(99)
        for sub in range(labels.shape[0]):
            rng.shuffle(labels[sub])
        print("Labels permuted within each subject (expect chance accuracy).")
    print(f"Labels loaded: {labels.shape}, class balance: {labels.mean():.3f}")

    valid_mask = load_valid_trials(args.features_dir, args.ratings, order='canonical')
    n_valid = int(valid_mask.sum())
    print(f"Valid trials (canonical): {n_valid}/{N_SUBS * N_TRIALS}")
    if n_valid != EXPECTED_VALID_COUNT:
        print(f"  WARNING: expected {EXPECTED_VALID_COUNT} valid trials for the full dataset.")

    source = args.selection == 'source'
    fold_dir = os.path.join(args.features_dir,
                            'fold_loso_source' if source else 'fold_loso')
    stored_val = None
    if source:
        with open(os.path.join(fold_dir, 'validation_subjects.json')) as f:
            stored_val = json.load(f)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    fold_results = []
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = (f"deap_loso_{args.target}{'_shuffled' if args.shuffle_labels else ''}"
                f"_{args.selection}_{run_stamp}")
    preds_dir = os.path.join(args.output_dir, 'predictions', run_name)
    os.makedirs(preds_dir, exist_ok=True)

    for fold in range(N_SUBS):
        fold_path = os.path.join(fold_dir, f'de_final_fold{fold}.npy')
        if not os.path.exists(fold_path):
            print(f"  Fold {fold}: MISSING, skipping")
            continue

        de_data = np.load(fold_path)
        X, y, subject_ids = de_to_samples(de_data, labels, valid_mask=valid_mask)
        assert not np.isnan(X).any(), f"fold {fold}: NaN in features after valid-trial filter"

        test_mask = subject_ids == fold
        train_mask = ~test_mask
        val_subjects = []
        if source:
            pool = sorted(set(range(N_SUBS)) - {fold})
            val_subjects = pick_validation_subjects(pool, fold)
            if stored_val[str(fold)] != val_subjects:
                raise RuntimeError(
                    f"fold {fold}: validation subjects {val_subjects} differ from those "
                    f"excluded from the normalisation statistics {stored_val[str(fold)]}")
            val_mask = np.isin(subject_ids, val_subjects)
            train_mask = train_mask & ~val_mask
            assert not (val_mask & test_mask).any()

        X_train = torch.tensor(X[train_mask])
        y_train = torch.tensor(y[train_mask])
        X_test = torch.tensor(X[test_mask])
        y_test = torch.tensor(y[test_mask])

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, persistent_workers=True)
        if source:
            val_ds = TensorDataset(torch.tensor(X[val_mask]), torch.tensor(y[val_mask]))
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, persistent_workers=True)
        else:
            val_loader = test_loader

        model = build_model()
        trainer = ClassifierTrainer(
            model=model,
            num_classes=2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            metrics=['accuracy'],
            accelerator=accelerator,
            devices=1,
        )

        callbacks = [pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, mode='min')]
        best = BestValLossWeights() if source else None
        if best is not None:
            callbacks.append(best)

        trainer.fit(
            train_loader, val_loader,
            max_epochs=args.epochs,
            default_root_dir=os.path.join(args.output_dir, 'lightning', run_name,
                                          f'fold_{fold}'),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=(fold == 0),
        )
        if best is not None:
            best.restore(model)

        score = trainer.test(test_loader, enable_progress_bar=False,
                             enable_model_summary=False)[0]
        acc = float(score['test_accuracy'])

        y_true, y_pred = predict(model, test_loader)
        bal_acc = balanced_accuracy(y_true, y_pred)
        maj = majority_baseline(y_true)

        preds_file = os.path.join(preds_dir, f'fold{fold:02d}_sub{fold:02d}.npz')
        np.savez(preds_file, y_true=y_true, y_pred=y_pred, test_sub=np.array([fold]))

        fold_results.append({
            'fold': fold,
            'test_subject': fold,
            'validation_subjects': val_subjects,
            'best_val_loss_epoch': best.best_epoch if best is not None else None,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'majority_baseline': maj,
            'n_windows': int(len(y_true)),
        })
        print(f"  Fold {fold:2d}: acc={acc:.4f}, bal_acc={bal_acc:.4f}, maj={maj:.4f}")

    mean_acc, std_acc = mean_sd([r['accuracy'] for r in fold_results])
    mean_bal, std_bal = mean_sd([r['balanced_accuracy'] for r in fold_results])
    mean_maj, _ = mean_sd([r['majority_baseline'] for r in fold_results])
    print(f"\nLOSO DGCNN ({len(fold_results)}/{N_SUBS} folds)")
    print(f"  Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Balanced accuracy: {mean_bal:.4f} +/- {std_bal:.4f}")
    print(f"  Majority baseline: {mean_maj:.4f}")

    results = {
        'dataset': 'DEAP',
        'model': 'dgcnn',
        'eval_mode': 'loso',
        'target': args.target,
        'selection': args.selection,
        'features': f'de5_{args.window_sec}s_runningnorm_lds_zscore',
        'shuffle_labels': args.shuffle_labels,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_balanced_accuracy': mean_bal,
        'std_balanced_accuracy': std_bal,
        'std_ddof': 0,
        'mean_majority_baseline': mean_maj,
        'n_folds_evaluated': len(fold_results),
        'fold_results': fold_results,
        'predictions_dir': preds_dir,
        'args': vars(args),
    }
    result_path = os.path.join(args.output_dir, f'{run_name}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {result_path}")
    return results


def run_sd(args):
    de_raw = np.load(os.path.join(args.features_dir, 'de_raw.npy'))  # playback order
    labels, _ = load_deap_labels(args.ratings, target=args.target)

    valid_mask = load_valid_trials(args.features_dir, args.ratings, order='canonical')
    n_valid = int(valid_mask.sum())
    print(f"Valid trials (canonical): {n_valid}/{N_SUBS * N_TRIALS}")
    if n_valid != EXPECTED_VALID_COUNT:
        print(f"  WARNING: expected {EXPECTED_VALID_COUNT} valid trials for the full dataset.")

    # de_raw.npy is in playback order; labels are in canonical order.
    playback_order = load_deap_playback_to_canonical(args.ratings)
    de_raw = reorder_playback_to_canonical(de_raw, playback_order)

    # Per-subject z-score over the subject's valid windows (NaN-aware).
    de_zscore = np.full_like(de_raw, np.nan)
    for sub in range(N_SUBS):
        mean = np.nanmean(de_raw[sub], axis=0)
        std = np.nanstd(de_raw[sub], axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        de_zscore[sub] = (de_raw[sub] - mean) / std

    source = args.selection == 'source'
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    n_splits = 10
    all_subject_results = []

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = (f"deap_sd_{args.split}_{args.target}"
                f"{'_shuffled' if args.shuffle_labels else ''}_{args.selection}_{run_stamp}")
    preds_dir = os.path.join(args.output_dir, 'predictions', run_name)
    os.makedirs(preds_dir, exist_ok=True)

    for sub in range(N_SUBS):
        sub_data = de_zscore[sub]
        sub_labels = labels[sub].copy()
        if args.shuffle_labels:
            np.random.seed(99 + sub)
            np.random.shuffle(sub_labels)

        y_sub = np.repeat(sub_labels, N_TIMESTEPS_PER_TRIAL)
        X_sub = sub_data.reshape(-1, N_BANDS, N_CHANNELS).transpose(0, 2, 1).astype(np.float32)

        valid_canonical = np.where(valid_mask[sub])[0]
        if len(valid_canonical) < n_splits:
            print(f"  Sub {sub:02d}: only {len(valid_canonical)} valid trials; skipping")
            continue
        np.random.seed(42)
        trial_indices = valid_canonical.copy()
        np.random.shuffle(trial_indices)
        trial_folds = np.array_split(trial_indices, n_splits)

        sub_accs = []
        sub_fold_results = []

        for fold in range(n_splits):
            if args.split == 'trial':
                test_trials = trial_folds[fold]
                train_trials_all = np.concatenate(
                    [trial_folds[i] for i in range(n_splits) if i != fold]
                )

                if source:
                    # One ninth of the training trials (about 10 % of all).
                    rng_val = np.random.RandomState(42 + fold)
                    n_val = max(1, len(train_trials_all) // (n_splits - 1))
                    perm = rng_val.permutation(len(train_trials_all))
                    val_trials = train_trials_all[perm[:n_val]]
                    train_trials = train_trials_all[perm[n_val:]]
                else:
                    train_trials = train_trials_all
                    val_trials = test_trials

                train_idx = trial_windows(train_trials)
                val_idx = trial_windows(val_trials)
                test_idx = trial_windows(test_trials)
            else:
                # Within-trial split: fold k tests the k-th of 10 consecutive
                # blocks of every trial (the last block absorbs the remainder).
                train_idx_list, val_idx_list, test_idx_list = [], [], []
                for t in valid_canonical:
                    t_start = t * N_TIMESTEPS_PER_TRIAL
                    chunk_indices = np.arange(t_start, t_start + N_TIMESTEPS_PER_TRIAL)
                    n_chunks = len(chunk_indices)
                    fold_size = n_chunks // n_splits

                    def block(k):
                        b_start = k * fold_size
                        b_end = b_start + fold_size if k < n_splits - 1 else n_chunks
                        return b_start, b_end

                    test_start, test_end = block(fold)
                    test_idx_list.append(chunk_indices[test_start:test_end])
                    keep = np.ones(n_chunks, dtype=bool)
                    keep[test_start:test_end] = False
                    if source:
                        val_start, val_end = block((fold + 1) % n_splits)
                        val_idx_list.append(chunk_indices[val_start:val_end])
                        keep[val_start:val_end] = False
                    train_idx_list.append(chunk_indices[keep])
                train_idx = np.concatenate(train_idx_list)
                test_idx = np.concatenate(test_idx_list)
                val_idx = np.concatenate(val_idx_list) if source else test_idx

            assert not np.isnan(X_sub[train_idx]).any()
            assert not np.isnan(X_sub[val_idx]).any()
            assert not np.isnan(X_sub[test_idx]).any()
            if source:
                assert not np.intersect1d(val_idx, test_idx).size
                assert not np.intersect1d(val_idx, train_idx).size
            if fold == 0:
                print(f"  Sub {sub:02d} split: train={len(train_idx)}, "
                      f"val={len(val_idx)}, test={len(test_idx)}")

            X_train = torch.tensor(X_sub[train_idx])
            y_train = torch.tensor(y_sub[train_idx])
            X_val = torch.tensor(X_sub[val_idx])
            y_val = torch.tensor(y_sub[val_idx])
            X_test = torch.tensor(X_sub[test_idx])
            y_test = torch.tensor(y_sub[test_idx])

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            test_ds = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=2, persistent_workers=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, persistent_workers=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=2, persistent_workers=True)

            model = build_model()
            trainer = ClassifierTrainer(
                model=model,
                num_classes=2,
                lr=args.lr,
                weight_decay=args.weight_decay,
                metrics=['accuracy'],
                accelerator=accelerator,
                devices=1,
            )

            callbacks = [pl.callbacks.EarlyStopping(
                monitor='val_loss', patience=args.patience, mode='min')]
            best = BestValLossWeights() if source else None
            if best is not None:
                callbacks.append(best)

            trainer.fit(
                train_loader, val_loader,
                max_epochs=args.epochs,
                default_root_dir=os.path.join(args.output_dir, 'lightning', run_name,
                                              f'sub_{sub:02d}', f'fold_{fold}'),
                callbacks=callbacks,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            if best is not None:
                best.restore(model)

            score = trainer.test(test_loader, enable_progress_bar=False,
                                 enable_model_summary=False)[0]
            plain_acc = float(score['test_accuracy'])
            sub_accs.append(plain_acc)

            y_true, y_pred = predict(model, test_loader)
            bal_acc = balanced_accuracy(y_true, y_pred)
            maj = majority_baseline(y_true)

            preds_file = os.path.join(preds_dir, f'sub{sub:02d}_fold{fold:02d}.npz')
            np.savez(preds_file, y_true=y_true, y_pred=y_pred,
                     sub=np.array([sub]), fold=np.array([fold]))

            sub_fold_results.append({
                'fold': fold,
                'best_val_loss_epoch': best.best_epoch if best is not None else None,
                'accuracy': plain_acc,
                'balanced_accuracy': bal_acc,
                'majority_baseline': maj,
                'n_windows': int(len(y_true)),
            })

        sub_mean_plain = float(np.mean(sub_accs))
        sub_mean_bal, _ = mean_sd([r['balanced_accuracy'] for r in sub_fold_results])
        sub_mean_maj, _ = mean_sd([r['majority_baseline'] for r in sub_fold_results])
        all_subject_results.append({
            'subject': sub,
            'mean_accuracy': sub_mean_plain,
            'mean_balanced_accuracy': sub_mean_bal,
            'mean_majority_baseline': sub_mean_maj,
            'fold_results': sub_fold_results,
        })
        print(f"  Sub {sub:02d}: acc={sub_mean_plain:.4f}, bal={sub_mean_bal:.4f}, "
              f"maj={sub_mean_maj:.4f}")

    # Mean and SD across subjects of each subject's 10-fold mean.
    mean_acc, std_acc = mean_sd([r['mean_accuracy'] for r in all_subject_results])
    mean_bal, std_bal = mean_sd([r['mean_balanced_accuracy'] for r in all_subject_results])
    mean_maj, _ = mean_sd([r['mean_majority_baseline'] for r in all_subject_results])
    print(f"\nSD DGCNN ({len(all_subject_results)}/{N_SUBS} subjects, split={args.split})")
    print(f"  Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Balanced accuracy: {mean_bal:.4f} +/- {std_bal:.4f}")
    print(f"  Majority baseline: {mean_maj:.4f}")

    results = {
        'dataset': 'DEAP',
        'model': 'dgcnn',
        'eval_mode': 'subject_dependent',
        'target': args.target,
        'split': args.split,
        'selection': args.selection,
        'features': f'de5_{args.window_sec}s_zscore',
        'shuffle_labels': args.shuffle_labels,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_balanced_accuracy': mean_bal,
        'std_balanced_accuracy': std_bal,
        'std_ddof': 0,
        'mean_majority_baseline': mean_maj,
        'n_subjects_evaluated': len(all_subject_results),
        'subject_results': all_subject_results,
        'predictions_dir': preds_dir,
        'args': vars(args),
    }
    result_path = os.path.join(args.output_dir, f'{run_name}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {result_path}")
    return results


def set_window(window_sec):
    """Set the number of DE windows per 60-s trial (1 s -> 60, 4 s -> 15)."""
    global N_TIMESTEPS_PER_TRIAL, N_TIMESTEPS_TOTAL
    N_TIMESTEPS_PER_TRIAL = 60 // window_sec
    N_TIMESTEPS_TOTAL = N_TRIALS * N_TIMESTEPS_PER_TRIAL


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DGCNN on DEAP DE features')
    parser.add_argument('--features-dir', required=True,
                        help='Output directory of deap_features.py.')
    parser.add_argument('--ratings', required=True,
                        help="DEAP's participant_ratings.csv.")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--eval-mode', default='loso', choices=['loso', 'sd'])
    parser.add_argument('--split', default='trial', choices=['trial', 'temporal'],
                        help='Subject-dependent split (ignored for loso).')
    parser.add_argument('--selection', default='source', choices=['source', 'test'])
    parser.add_argument('--target', default='valence', choices=['valence', 'arousal'])
    parser.add_argument('--window-sec', type=int, default=1, choices=[1, 4],
                        help='DE window length the features were computed with.')
    parser.add_argument('--shuffle-labels', action='store_true',
                        help='Permute trial labels within each subject (permutation control).')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    pl.seed_everything(42, workers=True)
    args = parse_args(argv)
    set_window(args.window_sec)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_mode == 'loso':
        return run_loso(args)
    return run_sd(args)


if __name__ == '__main__':
    main()
