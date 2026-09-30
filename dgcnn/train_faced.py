"""Train DGCNN on FACED binary valence, 10-fold cross-subject.

Input: the per-subject DE features written by faced_features.py
(``subNNN.npy``, shape (28, 30, 32, 5)). The 24 non-neutral clips are used
(clips 1-12 anger/disgust/fear/sadness, clips 17-28 amusement/inspiration/
joy/tenderness; the four neutral clips 13-16 are dropped).

Labels (--label-source):
  stim         clip category: clips 1-12 negative (0), clips 17-28 positive (1).
  self_report  the subject's own valence rating from After_remarks.mat
               (0-7 scale), positive when rating >= --sam-threshold (3.0).

Folds: 123 subjects in 10 consecutive blocks (folds 0-8: 12 subjects each,
fold 9: the last 15). Features are z-scored per subject over that subject's
own windows.

Checkpoint selection (--selection):
  source (default)  15 % of the fold's training subjects (seed 42 + fold)
                    form the validation set and are excluded from training;
                    early stopping on the validation loss (patience 20, at most
                    200 epochs) and the weights with the lowest validation loss
                    are evaluated.
  fixed_epochs      50 epochs, no validation set, final-epoch weights (the
                    rule behind the reported FACED numbers).
  test              early stopping on the test subjects' loss (patience 20, at
                    most 200 epochs); the weights at the stopping epoch are
                    evaluated (the same rule as train_deap.py/train_seed.py
                    --selection test).

Usage:
    python train_faced.py --data-dir /path/to/faced_de --output-dir runs/faced \
        --label-source stim --selection fixed_epochs
"""

import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from common import mean_sd, pick_validation_subjects
from dgcnn import DGCNN

N_SUBS = 123
N_CHANNELS = 32
N_BANDS = 5
N_TRIALS_FULL = 28
N_WINDOWS_PER_TRIAL = 30

NEG_TRIALS = list(range(0, 12))    # anger, disgust, fear, sadness
POS_TRIALS = list(range(16, 28))   # amusement, inspiration, joy, tenderness
KEEP_TRIALS = NEG_TRIALS + POS_TRIALS


def load_faced_de_subject(de_root, sub_id):
    """(n_trials, 30, 32, 5) DE features of one subject."""
    path = os.path.join(de_root, f'sub{sub_id:03d}.npy')
    de = np.load(path)
    if de.shape[1:] != (N_WINDOWS_PER_TRIAL, N_CHANNELS, N_BANDS):
        raise ValueError(
            f'sub {sub_id}: expected (?, {N_WINDOWS_PER_TRIAL}, {N_CHANNELS}, '
            f'{N_BANDS}), got {de.shape}')
    return de


def load_self_report_ratings(ratings_dir, n_subjects=N_SUBS):
    """Self-reported ratings from FACED's ``Data/subNNN/After_remarks.mat``.

    Returns:
        dict {(subject_index, clip_index): {'valence': float, 'arousal': float}}
        with 0-based subject and clip indices (clip_index = vid - 1). Valence is
        column 9 and arousal column 8 of ``score`` (0-7 scale).
    """
    import scipy.io as sio

    ratings = {}
    for sub_idx in range(0, n_subjects):
        mat_path = os.path.join(ratings_dir, f'sub{sub_idx:03d}', 'After_remarks.mat')
        if not os.path.exists(mat_path):
            print(f"  WARNING: missing {mat_path}, skipping subject {sub_idx}")
            continue

        mat = sio.loadmat(mat_path)
        data = mat['After_remark']  # (28, 1) structured array

        for i in range(data.shape[0]):
            row = data[i, 0]
            vid = int(row['vid'].item())
            trial_id = vid - 1
            valence = float(row['score'][0, 9])
            arousal = float(row['score'][0, 8])
            ratings[(sub_idx, trial_id)] = {'valence': valence, 'arousal': arousal}
    return ratings


def build_binary_arrays(de_root, n_subs, label_source='stim', ratings=None,
                        sam_threshold=3.0):
    """Per-window arrays for the 24-clip binary valence task.

    Returns:
        X: (n_windows, 32, 5) float32; y: (n_windows,) int64;
        subj_idx: (n_windows,) subject index; trial_idx: (n_windows,) 0..23.
    """
    X_list, y_list, sub_list, trial_list = [], [], [], []
    for sub_id in range(n_subs):
        de = load_faced_de_subject(de_root, sub_id)  # (28, 30, 32, 5)
        de = de[KEEP_TRIALS]                          # (24, 30, 32, 5)
        if label_source == 'stim':
            trial_labels = np.concatenate([
                np.zeros(len(NEG_TRIALS), dtype=np.int64),
                np.ones(len(POS_TRIALS), dtype=np.int64),
            ])
        else:
            trial_labels = np.zeros(24, dtype=np.int64)
            for new_trial_idx, orig_vid in enumerate(KEEP_TRIALS):
                sam_val = ratings[(sub_id, orig_vid)]['valence']
                trial_labels[new_trial_idx] = int(sam_val >= sam_threshold)
        n_trials_subj = 24

        y_subj = np.repeat(trial_labels, N_WINDOWS_PER_TRIAL)
        X_subj = de.reshape(-1, N_CHANNELS, N_BANDS)
        trial_id_subj = np.repeat(np.arange(n_trials_subj, dtype=np.int64),
                                  N_WINDOWS_PER_TRIAL)
        sub_id_subj = np.full(X_subj.shape[0], sub_id, dtype=np.int64)

        X_list.append(X_subj)
        y_list.append(y_subj)
        sub_list.append(sub_id_subj)
        trial_list.append(trial_id_subj)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    subj_idx = np.concatenate(sub_list)
    trial_idx = np.concatenate(trial_list)
    return X, y, subj_idx, trial_idx


def make_lnso_folds(n_subs, n_folds):
    """Consecutive subject blocks; the last fold absorbs the remainder."""
    fold_size = n_subs // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_subs
        folds.append(list(range(start, end)))
    return folds


def run_fold(args, fold, X_all, y_all, subj_all, device):
    # Every fold starts from the same random state, as when each fold is run
    # in its own process.
    pl.seed_everything(args.base_seed, workers=True)
    num_classes = 2

    folds = make_lnso_folds(args.n_subs, args.n_folds)
    test_subjects = folds[fold]
    test_mask = np.isin(subj_all, test_subjects)

    val_subjects = []
    val_mask = None
    if args.selection == 'source':
        train_pool_subs = [s for s in range(args.n_subs) if s not in test_subjects]
        val_subjects = pick_validation_subjects(train_pool_subs, fold,
                                                frac=args.val_frac, seed=args.val_seed)
        val_mask = np.isin(subj_all, val_subjects)
        assert not (val_mask & test_mask).any()

    if val_mask is not None:
        train_mask = ~(test_mask | val_mask)
    else:
        train_mask = ~test_mask

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    X_val = X_all[val_mask] if val_mask is not None else None
    y_val = y_all[val_mask] if val_mask is not None else None

    print(f'\nFold {fold}: test subjects {test_subjects}')
    if val_subjects:
        print(f'  validation subjects ({len(val_subjects)}): {val_subjects}')
    print(f'  sizes: train={len(y_train)} val={0 if y_val is None else len(y_val)} '
          f'test={len(y_test)}')

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, persistent_workers=True)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, persistent_workers=True)
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=args.batch_size, shuffle=False,
            num_workers=2, persistent_workers=True)

    model = DGCNN(
        in_channels=N_BANDS,
        num_electrodes=N_CHANNELS,
        hid_channels=args.hid_channels,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def eval_loader(loader):
        """(mean batch loss, accuracy) on a loader."""
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss_sum += criterion(logits, yb).item()
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return loss_sum / max(len(loader), 1), correct / max(total, 1)

    monitor_loader = val_loader if args.selection == 'source' else test_loader
    best_loss = float('inf')
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0
    epoch = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / max(len(train_loader), 1)

        if args.selection in ('source', 'test'):
            mon_loss, mon_acc = eval_loader(monitor_loader)
            if mon_loss < best_loss:
                best_loss = mon_loss
                best_epoch = epoch
                if args.selection == 'source':
                    best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            print(f'[epoch {epoch + 1:3d}/{args.epochs}] train_loss={train_loss:.4f} '
                  f'monitored_loss={mon_loss:.4f} best={best_loss:.4f} (epoch {best_epoch + 1})')
            if patience_counter >= args.patience:
                print(f'  early stop at epoch {epoch + 1}')
                break
        else:
            # Test accuracy is printed for monitoring only; it does not
            # influence training. The evaluation is kept because it advances
            # the random state (DataLoader iterator seed) exactly as in the
            # runs that produced the reported numbers.
            _, test_acc_now = eval_loader(test_loader)
            print(f'[epoch {epoch + 1:3d}/{args.epochs}] train_loss={train_loss:.4f} '
                  f'test_acc={test_acc_now:.4f}')

    if args.selection == 'source' and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            y_pred_list.append(logits.argmax(dim=1).cpu().numpy())
            y_true_list.append(yb.numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    plain_acc = float((y_true == y_pred).mean())
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = float('nan')
    try:
        f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    except Exception:
        f1_macro = float('nan')

    print(f'  fold {fold}: accuracy={plain_acc:.4f} balanced={bal_acc:.4f} f1={f1_macro:.4f}')
    return {
        'fold': fold,
        'test_subjects': [int(s) for s in test_subjects],
        'validation_subjects': [int(s) for s in val_subjects],
        'n_epochs_run': int(epoch + 1),
        'best_monitored_epoch': int(best_epoch) if best_epoch >= 0 else None,
        'best_monitored_loss': float(best_loss) if best_epoch >= 0 else None,
        'accuracy': plain_acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'n_train': int(len(y_train)),
        'n_val': 0 if y_val is None else int(len(y_val)),
        'n_test': int(len(y_test)),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='DGCNN on FACED binary valence, 10-fold cross-subject')
    p.add_argument('--data-dir', required=True, help='Output directory of faced_features.py.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--label-source', default='stim', choices=['stim', 'self_report'])
    p.add_argument('--ratings-dir', default=None,
                   help="FACED 'Data' directory with subNNN/After_remarks.mat "
                        "(required for --label-source self_report).")
    p.add_argument('--sam-threshold', type=float, default=3.0,
                   help='Valence rating at or above which a trial is positive.')
    p.add_argument('--selection', default='source',
                   choices=['source', 'fixed_epochs', 'test'])
    p.add_argument('--folds', type=int, nargs='*', default=None,
                   help='Fold indices to run (default: all).')
    p.add_argument('--n-folds', type=int, default=10)
    p.add_argument('--n-subs', type=int, default=N_SUBS)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=None,
                   help='Default: 50 for fixed_epochs, 200 (maximum) otherwise.')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hid-channels', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args(argv)
    if args.epochs is None:
        args.epochs = 50 if args.selection == 'fixed_epochs' else 200
    if args.label_source == 'self_report' and args.ratings_dir is None:
        p.error('--ratings-dir is required with --label-source self_report')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    ratings = None
    if args.label_source == 'self_report':
        ratings = load_self_report_ratings(args.ratings_dir, args.n_subs)
    X_all, y_all, subj_all, _ = build_binary_arrays(
        args.data_dir, args.n_subs, label_source=args.label_source,
        ratings=ratings, sam_threshold=args.sam_threshold)
    print(f'X: {X_all.shape}; label counts: {np.bincount(y_all).tolist()}')

    # Per-subject z-score over each subject's own windows.
    for sub_id in range(args.n_subs):
        mask = subj_all == sub_id
        xs = X_all[mask]
        mu = xs.mean(axis=0, keepdims=True)
        sd = xs.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_all[mask] = ((xs - mu) / sd).astype(np.float32)

    folds = args.folds if args.folds else list(range(args.n_folds))
    fold_results = [run_fold(args, fold, X_all, y_all, subj_all, device) for fold in folds]

    # Sample SD (ddof=1) across folds, as for the reported FACED numbers.
    mean_acc, std_acc = mean_sd([r['accuracy'] for r in fold_results], ddof=1)
    mean_bal, std_bal = mean_sd([r['balanced_accuracy'] for r in fold_results], ddof=1)
    print(f'\nFACED binary valence ({args.label_source}), selection={args.selection}')
    print(f'  Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}')
    print(f'  Balanced accuracy: {mean_bal:.4f} +/- {std_bal:.4f}')

    run_name = (f'faced_binary_{args.label_source}_{args.selection}_'
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results = {
        'dataset': 'FACED',
        'model': 'dgcnn',
        'task': 'binary valence (24 clips)',
        'label_source': args.label_source,
        'sam_threshold': args.sam_threshold if args.label_source == 'self_report' else None,
        'selection': args.selection,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_balanced_accuracy': mean_bal,
        'std_balanced_accuracy': std_bal,
        'std_ddof': 1,
        'n_folds_evaluated': len(fold_results),
        'fold_results': fold_results,
        'args': vars(args),
    }
    out = os.path.join(args.output_dir, f'{run_name}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved: {out}')
    return results


if __name__ == '__main__':
    main()
