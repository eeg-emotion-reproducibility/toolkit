# Reproducibility and Comparability of EEG Emotion Recognition

Code and data accompanying the paper *Reproducibility and Comparability of EEG Emotion
Recognition: Controlled Measurements and a Code-Level Audit* (under review).

The paper follows a typical EEG emotion-recognition pipeline stage by stage and asks, at each
stage, whether a published accuracy can be regenerated from its paper and released code, and
whether accuracies obtained under different setups can be compared. This repository holds the
analyses that can be rerun from the public datasets, the protocol survey, the reporting
checklist, and the DGCNN code used as the supervised control.

## Contents

| Folder / file | Contents | Paper |
|---|---|---|
| `labels/` | DEAP and FACED label statistics; inter-subject agreement (Krippendorff's alpha) | Section III-B; Tables 5-7 |
| `preprocessing/` | DE features from DAEST's preprocessed FACED signal; correlation between the two FACED pipelines; DEAP spectral content below 4 Hz | Section III-A |
| `dgcnn/` | DGCNN training on DEAP, SEED and FACED | Sections III-B, III-C, IV-D; Tables 8, 9, 11 |
| `survey/` | Protocol survey: 45 paper-by-dataset records | Section II-C; Table 3 |
| `evaluation_protocols/` | Within-trial splitting in TorchEEG 1.1.2 | Section III-C |
| `CHECKLIST.md` | Minimum reporting checklist | Section V |

## Data

The datasets are not redistributed here; obtain them from their providers.

| Dataset | Source | Used by |
|---|---|---|
| DEAP | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html (metadata files, `data_preprocessed_python`, and the raw BDF recordings used by `dgcnn/`) | `labels/`, `preprocessing/deap_delta_band.py`, `dgcnn/` |
| SEED | https://bcmi.sjtu.edu.cn/home/seed/ | `dgcnn/` |
| FACED | https://doi.org/10.7303/syn50614194 (ratings in `Data/subNNN/After_remarks.mat`; processed data and DE features) | `labels/`, `preprocessing/`, `dgcnn/` |
| DREAMER | https://zenodo.org/records/546113 (`DREAMER.mat`) | `labels/krippendorff_alpha.py` |

## Environment

Python 3.10. The label and preprocessing scripts need numpy, scipy, pandas, xlrd (DEAP's `.xls`
metadata), krippendorff, mne and matplotlib; the DGCNN code additionally needs PyTorch,
PyTorch Lightning, TorchEEG and scikit-learn (versions in `dgcnn/README_DGCNN.md`). The label analyses below were
checked with numpy 2.2, scipy 1.15, pandas 2.3, xlrd 2.0 and krippendorff 0.8.

## Label analyses

```
python labels/deap_labels.py --metadata-dir DEAP/metadata
python labels/faced_labels.py --faced-dir FACED/Data
python labels/krippendorff_alpha.py --faced-dir FACED/Data --deap-metadata-dir DEAP/metadata --dreamer-mat DREAMER.mat
```

`deap_labels.py` prints the majority-class baseline of binary valence under thresholds 4.0-6.0
(Table 5: 72.2, 63.1, 56.6, 54.1 and 57.3 %), the share of valence ratings within +/-1.0 of the
threshold 5.0 (385/1,280 = 30.1 %), and the disagreement between stimulus-assigned and
self-reported labels at threshold 5.0 (valence 286/1,280 = 22.3 %, arousal 462/1,280 = 36.1 %).
`faced_labels.py` prints the share of valence ratings within +/-1.0 of the midpoint 3.5
(1,280/3,444 = 37.2 %), the label cross-tabulation of Table 7 (disagreement 585/2,952 = 19.8 %
on valence and 1,272/2,952 = 43.1 % on arousal), and the mutual information between the two
binary labels (0.290 bits on valence against a self-report entropy of 0.987 bits; 0.006 against
0.967 bits on arousal). `krippendorff_alpha.py` prints Table 6.

## Preprocessing analyses

```
python preprocessing/daest_de_features.py --src DAEST/preprocessed --dst features/daest_de
python preprocessing/cross_pipeline_correlation.py --chen-de-dir FACED/EEG_Features/DE --daest-de-dir features/daest_de
python preprocessing/deap_delta_band.py --data-dir DEAP/data_preprocessed_python --figure deap_psd.pdf
```

The correlation between the DE features of the two pipelines, over 123 subjects, is
r = 0.55 +/- 0.17, and 0.22, 0.34, 0.35, 0.54 and 0.62 in the delta, theta, alpha, beta and
gamma bands.

## DGCNN

`dgcnn/` trains DGCNN on DEAP (leave-one-subject-out, and subject-dependent 10-fold with a
trial-level or within-trial split), SEED (leave-one-subject-out) and FACED (10-fold,
stimulus-assigned or self-reported labels). `dgcnn/README_DGCNN.md` gives the data layout, the
package versions and the commands. By default each training script selects its checkpoint on
validation data drawn from the training subjects; `--selection test` (DEAP, SEED) and
`--selection fixed_epochs` (FACED) give the settings behind the numbers in the paper.

## Protocol survey

`survey/protocol_survey.csv` holds the 45 records; `survey/README.md` defines the columns.

## TorchEEG splitters

`evaluation_protocols/torcheeg_splitters.md` lists the TorchEEG 1.1.2 splitters that place windows of one trial
in both the training and the test set, with the source lines.

## License

The code is released under the MIT License (`LICENSE`). The survey and the documentation are
released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA.md`).
