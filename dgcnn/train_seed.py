"""Train DGCNN on SEED, leave-one-subject-out, 3 classes (negative/neutral/positive).

Input: the released SEED extracted-feature folder with 1-s windows, holding
``<subject>_<date>.mat`` (15 subjects x 3 sessions; keys ``de_LDS1`` ..
``de_LDS15``, each of shape (62, T, 5)) and ``label.mat``. The three sessions
of a subject are pooled; features are z-scored per subject over that
subject's own windows.

Checkpoint selection (--selection):
  source (default)  15 % of the 14 training subjects (seed 42 + fold) form the
                    validation set and are excluded from training; early
                    stopping on the validation loss and the weights with the
                    lowest validation loss are evaluated on the test subject.
  test              early stopping monitors the test subject's loss and the
                    weights at the stopping epoch are evaluated (the rule
                    behind the reported SEED number).

Usage:
    python train_seed.py --data-dir /path/to/SEED/ExtractedFeatures_1s \
        --output-dir runs/seed --selection test
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheeg.trainers import ClassifierTrainer

from common import (BestValLossWeights, balanced_accuracy, mean_sd,
                    pick_validation_subjects, predict)
from dgcnn import DGCNN

N_SUBS = 15
N_SESSIONS = 3
N_TRIALS_PER_SESSION = 15
N_TRIALS_POOLED = N_SESSIONS * N_TRIALS_PER_SESSION
N_CHANNELS = 62
N_BANDS = 5
N_CLASSES = 3


def load_seed_labels(data_dir):
    """Per-trial stimulus labels, (15,) int64 in {0, 1, 2} = {neg, neu, pos}."""
    lf = sio.loadmat(os.path.join(data_dir, "label.mat"))
    raw = lf['label'].flatten()  # values in {1, 0, -1}
    assert sorted(np.unique(raw).tolist()) == [-1, 0, 1], \
        f"Unexpected label values: {np.unique(raw)}"
    labels = (raw + 1).astype(np.int64)
    return labels


def load_seed_subject(data_dir, subject_id):
    """All three sessions of one subject (subject_id in 1..15).

    Returns:
        X: (n_windows, 62, 5) float32 de_LDS features, one row per 1-s window.
        y: (n_windows,) int64 labels.
        trial_ids: (n_windows,) global trial index in [0, 45).
    """
    labels = load_seed_labels(data_dir)
    pattern = os.path.join(data_dir, f"{subject_id}_*.mat")
    files = sorted(glob.glob(pattern))
    assert len(files) == N_SESSIONS, \
        f"Expected {N_SESSIONS} sessions for subject {subject_id}, got {len(files)}: {files}"

    X_list, y_list, trial_id_list = [], [], []
    global_trial = 0
    for session_idx, f in enumerate(files):
        d = sio.loadmat(f)
        for k in range(1, N_TRIALS_PER_SESSION + 1):
            key = f'de_LDS{k}'
            if key not in d:
                raise RuntimeError(f"Missing {key} in {f}")
            feat = d[key].astype(np.float32)  # (62, T, 5)
            assert feat.shape[0] == N_CHANNELS and feat.shape[2] == N_BANDS, \
                f"Unexpected shape for {key} in {f}: {feat.shape}"
            T = feat.shape[1]
            feat = np.transpose(feat, (1, 0, 2))  # (T, 62, 5)
            X_list.append(feat)
            y_list.append(np.full(T, labels[k - 1], dtype=np.int64))
            trial_id_list.append(np.full(T, global_trial, dtype=np.int64))
            global_trial += 1

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    trial_ids = np.concatenate(trial_id_list)
    return X, y, trial_ids


def run_loso(args):
    source = args.selection == 'source'

    all_data = {}
    for sub in range(1, N_SUBS + 1):
        X, y, trial_ids = load_seed_subject(args.data_dir, sub)
        # Per-subject, per-feature z-score over the subject's own windows.
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = ((X - mu) / sd).astype(np.float32)
        all_data[sub] = (X, y, trial_ids)
        print(f"  sub {sub:02d}: X={X.shape}, y distribution={np.bincount(y).tolist()}")

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'seed_loso_{args.selection}_{run_stamp}'
    preds_dir = os.path.join(args.output_dir, 'predictions', run_name)
    os.makedirs(preds_dir, exist_ok=True)

    fold_results = []

    for fold, test_sub in enumerate(range(1, N_SUBS + 1)):
        print(f"\nFold {fold + 1}/{N_SUBS}: held-out subject {test_sub}")

        val_subjects = []
        if source:
            pool = [s for s in range(1, N_SUBS + 1) if s != test_sub]
            val_subjects = pick_validation_subjects(pool, fold)
            assert test_sub not in val_subjects

        X_test, y_test, _ = all_data[test_sub]
        X_train_parts, y_train_parts = [], []
        X_val_parts, y_val_parts = [], []
        for sub in range(1, N_SUBS + 1):
            if sub == test_sub:
                continue
            X_s, y_s, _ = all_data[sub]
            if sub in val_subjects:
                X_val_parts.append(X_s)
                y_val_parts.append(y_s)
            else:
                X_train_parts.append(X_s)
                y_train_parts.append(y_s)
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts)

        print(f"  Train: {X_train.shape}, y dist: {np.bincount(y_train).tolist()}")
        print(f"  Test:  {X_test.shape}, y dist: {np.bincount(y_test).tolist()}")

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, persistent_workers=True)
        if source:
            X_val = np.concatenate(X_val_parts, axis=0)
            y_val = np.concatenate(y_val_parts)
            val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, persistent_workers=True)
            print(f"  Val:   {X_val.shape} (subjects {val_subjects})")
        else:
            val_loader = test_loader

        model = DGCNN(
            in_channels=N_BANDS,
            num_electrodes=N_CHANNELS,
            hid_channels=64,
            num_layers=2,
            num_classes=N_CLASSES,
            dropout=0.5,
        )

        trainer = ClassifierTrainer(
            model=model,
            num_classes=N_CLASSES,
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
        plain_acc = float((y_true == y_pred).mean())
        bal_acc = balanced_accuracy(y_true, y_pred)

        # Chance-level reference for balanced accuracy: 1/3 + 2 standard
        # errors, with the 45 trials of the test subject as the unit.
        se = math.sqrt((1.0 / 3.0) * (2.0 / 3.0) / N_TRIALS_POOLED)
        threshold = (1.0 / 3.0) + 2.0 * se
        passes = bool(bal_acc > threshold) if not math.isnan(bal_acc) else False

        preds_file = os.path.join(preds_dir, f'fold{fold:02d}_sub{test_sub:02d}.npz')
        np.savez(preds_file, y_true=y_true, y_pred=y_pred, test_sub=np.array([test_sub]))

        fold_results.append({
            'fold': fold,
            'test_subject': test_sub,
            'validation_subjects': val_subjects,
            'best_val_loss_epoch': best.best_epoch if best is not None else None,
            'accuracy': acc,
            'plain_accuracy_from_predictions': plain_acc,
            'balanced_accuracy': bal_acc,
            'chance_threshold_balanced_accuracy': threshold,
            'above_chance_threshold': passes,
            'n_windows': int(len(y_true)),
        })
        print(f"  Fold {fold:2d} (sub {test_sub:02d}): acc={acc:.4f}, "
              f"bal_acc={bal_acc:.4f}, above threshold={passes}")

    mean_acc, std_acc = mean_sd([r['accuracy'] for r in fold_results])
    mean_bal, std_bal = mean_sd([r['balanced_accuracy'] for r in fold_results])
    n_pass = sum(1 for r in fold_results if r['above_chance_threshold'])
    print(f"\nSEED DGCNN LOSO, 3 classes")
    print(f"  Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Balanced accuracy: {mean_bal:.4f} +/- {std_bal:.4f}")
    print(f"  Subjects above chance threshold: {n_pass}/{N_SUBS}")

    results = {
        'dataset': 'SEED',
        'model': 'dgcnn',
        'eval_mode': 'loso',
        'selection': args.selection,
        'features': 'de_LDS (1-s windows) + per-subject z-score',
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_balanced_accuracy': mean_bal,
        'std_balanced_accuracy': std_bal,
        'std_ddof': 0,
        'n_subjects_above_chance_threshold': n_pass,
        'n_folds_evaluated': len(fold_results),
        'fold_results': fold_results,
        'predictions_dir': preds_dir,
        'args': vars(args),
    }
    out = os.path.join(args.output_dir, f'{run_name}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DGCNN on SEED, leave-one-subject-out')
    parser.add_argument('--data-dir', required=True,
                        help='SEED extracted-feature folder (1-s windows) with '
                             '<subject>_<date>.mat and label.mat.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--selection', default='source', choices=['source', 'test'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pl.seed_everything(42, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    return run_loso(args)


if __name__ == '__main__':
    main()
