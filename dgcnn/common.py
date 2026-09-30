"""Helpers shared by the DGCNN training scripts.

Checkpoint selection (``--selection``):
  source        a validation set drawn only from the training data is used for
                early stopping and for choosing the reported checkpoint (the
                weights with the lowest validation loss). The test subject or
                fold is not looked at before the final evaluation.
  test          early stopping monitors the loss on the held-out test data and
                the weights at the stopping epoch are evaluated. This is the
                rule behind the DEAP and SEED numbers reported in the paper.
  fixed_epochs  (FACED only) a fixed number of epochs, final-epoch weights.
"""

import math

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score

VALIDATION_FRACTION = 0.15
VALIDATION_SEED = 42


def pick_validation_subjects(pool, fold, frac=VALIDATION_FRACTION,
                             seed=VALIDATION_SEED):
    """Draw a fraction of the training subjects as the validation subjects.

    ``pool`` lists the training subjects of one fold (the test subjects
    already removed). The draw depends only on the pool, the fold index and
    the seed, so the feature pipeline and the training scripts obtain the same
    subjects independently.
    """
    n_val = max(1, int(round(frac * len(pool))))
    rng = np.random.RandomState(seed + fold)
    shuffled = np.array(pool)
    rng.shuffle(shuffled)
    return sorted(int(s) for s in shuffled[:n_val].tolist())


class BestValLossWeights(pl.Callback):
    """Keep an in-memory copy of the weights with the lowest validation loss."""

    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_state = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get('val_loss')
        if current is None:
            return
        current = float(current)
        if current < self.best_loss:
            self.best_loss = current
            self.best_epoch = int(trainer.current_epoch)
            self.best_state = {k: v.detach().clone()
                               for k, v in pl_module.model.state_dict().items()}

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def predict(model, loader):
    """Return (y_true, y_pred) for a loader, evaluating the model in eval mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true_list.append(yb.numpy())
            y_pred_list.append(pred)
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def balanced_accuracy(y_true, y_pred):
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return float('nan')


def majority_baseline(y_true):
    """Share of the most frequent class among the evaluated labels."""
    if len(y_true) > 0:
        _, counts = np.unique(y_true, return_counts=True)
        return float(counts.max() / counts.sum())
    return float('nan')


def mean_sd(values, ddof=0):
    """Mean and standard deviation over finite values (NaN if none)."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float('nan'), float('nan')
    sd = float(np.std(vals, ddof=ddof)) if len(vals) > ddof else 0.0
    return float(np.mean(vals)), sd
