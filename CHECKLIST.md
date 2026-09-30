# Minimum reporting checklist for EEG emotion recognition results

Each stage of the analysis pipeline can change a reported accuracy, and each can differ between
two studies that use the same dataset. The checklist has two parts: what a paper must state, or
release in runnable form, for its number to be regenerated, and which additional results make
the number comparable with numbers reported elsewhere.

## A. For the number to be regenerated

1. **Preprocessing.** Whether the dataset's official preprocessed release was used or the raw
   recordings were re-preprocessed; if re-preprocessed, every step (filtering, re-referencing,
   artifact handling, normalization) with its parameters. The features and frequency bands with
   their limits in Hz, and the channels used.
2. **Label construction.** The label type (assigned to the stimulus, or self-reported by the
   subject); for self-reports, the rating used, the binarization threshold and whether the
   threshold value itself counts as high (`>=`) or low (`>`); the trials excluded.
3. **Data split.** Whether whole trials or windows of trials are assigned to folds, and how
   subjects are assigned to folds (the partition scheme, the number of folds, and the seed or
   the subject list of each fold).
4. **Checkpoint rule and its validation source.** How the reported checkpoint was chosen
   (fixed number of epochs, early stopping, or best of several evaluations) and on which data
   (training subjects held out for validation, or the test subjects).
5. **Every artifact behind the number.** Code, configuration files, and any preprocessing or
   format converter the pipeline needs, in a form that runs as released.

## B. For the number to be compared with other numbers

6. **The majority-class baseline** of the test labels under the stated threshold.
7. **The result under the other label type**, where the dataset provides both stimulus-assigned
   and self-reported labels.
8. **The source-validated result** beside any result whose checkpoint was selected on test data.
