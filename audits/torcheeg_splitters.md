# Within-trial splitting in TorchEEG 1.1.2

TorchEEG is a widely used EEG deep-learning library. Several of its data splitters assign the
windows of one trial to both the training and the test set, so that near-duplicate windows of
the same trial appear on both sides of the split. The line numbers below refer to the TorchEEG
1.1.2 source distribution on PyPI (`pip download torcheeg==1.1.2 --no-deps`), directory
`torcheeg/model_selection/`.

| Splitter | What it splits | Evidence |
|---|---|---|
| `KFoldGroupbyTrial` | k-fold over the windows **within each trial** | `k_fold_groupby_trial.py:86-90`: for each `trial_id`, `self.k_fold.split(trial_info)` splits that trial's rows |
| `KFoldPerSubjectGroupbyTrial` | the same, per subject | `k_fold_per_subject_groupby_trial.py:111-116`: for each `trial_id`, `self.k_fold.split(trial_info)` |
| `KFoldPerSubjectCrossTrial` | k-fold over **whole trials** (no within-trial leakage) | `k_fold_per_subject_cross_trial.py:108-112`: `self.k_fold.split(trial_ids)` |
| `train_test_split_groupby_trial` | hold-out split **within each trial** | its docstring, `split_groupby_trial.py:25`: "the first 80% of samples of each trial are used for training, and the last 20% of samples are used for testing" |

With a within-trial splitter, every test fold contains windows of trials that also appear in
training. In a subject-dependent evaluation this raises accuracy by exploiting the temporal
autocorrelation of adjacent windows rather than stimulus-related activity; the paper measures
the effect on DEAP with DGCNN (+36.68 pp on valence and +33.43 pp on arousal, trial-level
versus within-trial split). Assigning whole trials to folds, as `KFoldPerSubjectCrossTrial`
does, avoids it. Leave-one-subject-out evaluation is not affected, because all of a subject's
data stay in one fold.
