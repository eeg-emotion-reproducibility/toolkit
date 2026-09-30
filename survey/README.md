# Protocol survey

`protocol_survey.csv` holds 45 paper-by-dataset records from 31 publications that report
cross-subject EEG emotion recognition on DEAP (13 records), FACED (14), SEED (8), SEED-IV (7) and
DREAMER (3), published 2018-2025. A publication that reports several dataset-task pairs contributes
one record per pair, and each record was checked against the publication's PDF. The sample is
targeted rather than exhaustive: for each dataset it covers established cross-subject baselines and
recent methods reporting the highest accuracies, including the dataset papers' own baselines. The
paper summarises the survey in Section II-C and Table 3.

Each record describes the evaluation in the publication that reports the accuracy. Where that
publication is not the method's own paper, the `Paper` field says so ("result reported by ..."), and
`Year` and `Venue` refer to the reporting publication.

| Column | Content |
|---|---|
| `Paper` | Method and authors |
| `Year`, `Venue` | Year and venue of the publication that reports the accuracy |
| `Dataset`, `Task` | Dataset and classification task |
| `N_subjects`, `N_channels` | Subjects and EEG channels used |
| `Preprocessing` | Preprocessing as described in the publication |
| `Feature_type`, `Freq_bands` | Input features and frequency bands |
| `Segment_length_s` | Window length (and overlap, where stated) |
| `CV_scheme` | Cross-validation scheme as described in the publication |
| `Threshold` | Binarization threshold of the self-reported ratings; `stimulus` or `N/A` for stimulus-assigned labels |
| `Majority_baseline_pct` | Majority-class baseline (%); "derived" where computed from the dataset's label counts rather than reported |
| `Majority_baseline_reported` | Whether the publication reports the majority-class baseline |
| `Reported_acc_pct`, `Std` | Reported accuracy (%) and its spread as reported (V / A = valence / arousal) |
| `Code_available` | Whether code is publicly available |

`n/r`: not reported in the publication. `N/A`: not applicable.
