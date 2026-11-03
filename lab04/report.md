# Lab 4: data pipeline and quality

## Goal

Build a reproducible pipeline for the UCI Adult data that ingests, validates,
cleans and transforms it without leaking test information, checks label
quality, and ends with a hashed processed dataset, a config file and a data
card. The whole pipeline runs with one command: `python lab04/src/pipeline.py`.

## Problem definition

**Task.** Predict from census data whether a person earns more than $50,000 a
year. This is binary classification.

**Inputs.** 14 features of one person: 6 numeric (age, fnlwgt, education-num,
capital-gain, capital-loss, hours-per-week) and 8 categorical (workclass,
education, marital-status, occupation, relationship, race, sex,
native-country).

**Output.** income: `>50K` (positive class) or `<=50K`.

**Success metrics.** F1 on the `>50K` class and accuracy, both measured on a
held-out test set. F1 matters more here: only 23.9 % of people earn over $50K,
so a model that always answers `<=50K` already gets 76.1 % accuracy while
being useless.

**Constraints.**

- **No leakage.** The data is split before any statistic is computed. Every
  learned value (medians, modes, percentiles, scaling, categories) comes from
  the training set only, and the test set is never changed.
- **Reproducible.** Fixed seed 42, all settings in `config.yaml`, pinned
  package versions, a SHA256 of the processed data, and one command that
  rebuilds everything.

**Scope.** This is a course exercise. The data contains sensitive attributes
and 1994 census patterns (see `data_card.md`), so the model is not meant for
real decisions about people.

## Method

The steps live in `src/` and read their settings from `config.yaml`:

1. **Ingestion** (`pipeline.py`) loads the table from OpenML into `data/` and
   logs its shape and a SHA256 of the raw content. Nothing is changed. Later
   steps work on a copy where the label column `class` is renamed to `income`.
2. **Validation** (`validation.py`) checks the schema from the manual:
   - column set and types;
   - ranges (age 17-90, hours 1-99, capital-gain 0-99999, capital-loss 0-4356);
   - missing values under the `?` policy;
   - categories from the UCI data dictionary;
   - exact duplicates;
   - data-dictionary consistency: education against education-num, and
     Husband/Wife against sex and marital status.

   It writes `results/validation_report.md` and `.json`, and never drops a row.
3. **Split, then clean** (`pipeline.py`, `cleaning.py`):
   - a stratified 70/30 split by income;
   - exact duplicates are dropped from the training set only.

   The preprocessor is one scikit-learn pipeline, fitted on the training rows:
   - `?` becomes a missing value;
   - the derived features are added;
   - numeric columns: median imputation, clipping to the training 1st and 99th
     percentiles, standard scaling;
   - categorical columns: mode imputation and one-hot encoding (unknown test
     categories are ignored).

   It is saved with joblib as `results/preprocessor.joblib`.
4. **Derived features**: `capital_net = capital-gain - capital-loss` and
   `hours_per_age = hours-per-week / (age - 16)`. Both use only the same row,
   so they cannot leak. LogisticRegression (max_iter 1000) is trained twice,
   without and with them.
5. **Label quality** (`label_quality.py`):
   - finds groups of rows with identical features but different labels;
   - draws 50 random rows (seed 42) and re-checks them against the data
     dictionary (`results/label_audit_sample.csv`).
6. **Hash and record**: the processed train and test sets are written as CSV
   to `data/processed/` and hashed with SHA256.
   `results/run_record.yaml` stores the hash together with the full config,
   the row counts, the leakage checks, the scores and the package versions.

## Results

### Ingestion

48,842 rows and 15 columns, as expected. Raw SHA256:
`8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.

### Validation report

50 checks were run on the raw data; all are listed in
`results/validation_report.md`. Every type, range and category check passed.
The checks that found something:

| Check | Result |
|---|---|
| Missing values (`?` policy) | workclass 2,799, occupation 2,809, native-country 857 (allowed in categorical features); none in numeric columns or the label |
| Exact duplicates | 52 rows repeat an earlier row |
| Husband but Female | 1 row (id 23390) |
| Wife but Male | 3 rows (ids 5661, 16856, 43422) |
| education vs education-num | 0 mismatches |

The literal `?` count is 0. OpenML already turns the `?` markers of the
original files into empty values, so the pipeline treats both `?` and empty
values as missing. The 4 Husband/Wife rows are kept: they are reported, not
silently removed.

### Cleaning and leakage checks

| Item | Value |
|---|---|
| Train / test after split | 34,189 / 14,653 rows |
| Exact duplicates dropped from train | 24 (34,189 → 34,165) |
| Test rows changed | 0 (checked in code) |
| Train and test sharing a row id | 0 |
| Test rows identical to a train row | 24 |

The fitted statistics (`results/fitted_statistics.json`) equal the values
computed on the training rows and differ from the full dataset. For example:

- fnlwgt median: 178,272 on train, 178,144.5 on all data;
- fnlwgt 1st percentile: 27,035.2 on train, 27,166 on all data.

This shows the test rows were not used for fitting. The 24 test rows that copy
a training row are real repeated people in Adult. They were kept because the
manual says to leave the test set untouched, but they make the test score
slightly optimistic.

### Derived features

| Features | Columns after encoding | Accuracy | F1 (>50K) |
|---|---|---|---|
| Without derived features | 105 | 0.8540 | 0.6601 |
| With capital_net and hours_per_age | 107 | 0.8578 | 0.6696 |

Both features help a little: accuracy goes up by 0.38 points and F1 by 0.95
points.

### Label quality

- **Duplicate-conflict audit:** 5 groups of rows have the same 14 features but
  different income labels, 10 rows in total (`results/duplicate_conflicts.csv`).
  For example, two rows describe a 34-year-old woman with a Masters degree,
  working 50 hours a week as an Exec-managerial in the private sector: one is
  labelled `>50K`, the other `<=50K`. At least one label in each pair is wrong,
  or the 14 features are not enough to decide the label.
- **50 random rows** re-checked against the data dictionary: 0 impossible
  values. Every value is in the allowed categories and ranges,
  education-num matches education, and Husband/Wife match sex and marital
  status.

### Hash and reproducibility

Processed dataset (34,165 train rows and 14,653 test rows, 107 features plus
the label):

`9f7e9df55c31d69890114abdb6078fc295853647206a90b444d910dc159e7cf4`

I deleted `results/` and `data/processed/` and ran the one command again.
The hash came out identical. A fresh Python process also loaded
`preprocessor.joblib` and produced exactly the same processed test set.

## Three conclusions

1. **Splitting first is what makes the pipeline leakage-free.** Every statistic
   is learned after the split, and the saved values match the training set,
   not the full data. One kind of leakage cannot be removed by the pipeline:
   24 test rows are copies of training rows in the raw data itself.
2. **Validation should report problems, not hide them.** The schema and range
   checks passed, but the data still has 52 duplicates, 4 Husband/Wife rows
   that contradict the sex column, and 5 pairs of identical people with
   opposite labels. Dropping them silently would have hidden real label noise.
3. **Reproducibility needs more than a fixed seed.** The config file, pinned
   versions, a saved preprocessor and a SHA256 of the output let anyone check
   that they rebuilt exactly the same dataset. The hash matched after a clean
   rerun.
