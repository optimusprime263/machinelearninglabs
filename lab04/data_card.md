# Data card: UCI Adult (Census Income)

## Source

- **Dataset:** Adult, also called Census Income. Loaded with
  `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590).
- **Original source:** UCI Machine Learning Repository, donated on 30 April 1996
  by Barry Becker and Ronny Kohavi.
- **Citation:** Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5XW20
- **How it was collected:** Barry Becker extracted it from the 1994 US Census
  database. Only records matching these conditions were kept: age over 16,
  gross income over 100, final weight over 1, and hours worked over 0.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the
UCI page. OpenML lists the dataset as public. Use is allowed with attribution
(see the citation above).

## Size and structure

- 48,842 rows and 15 columns: 14 features and the label.
- **Numeric features (6):** age, fnlwgt, education-num, capital-gain,
  capital-loss, hours-per-week.
- **Categorical features (8):** workclass, education, marital-status,
  occupation, relationship, race, sex, native-country.
- **Label:** income, `<=50K` (76.1 %) or `>50K` (23.9 %). OpenML names this
  column `class`; the pipeline renames it to `income` in a copy.
- **Missing values:** workclass 2,799, occupation 2,809, native-country 857.
  In total, 3,620 rows (7.4 %) have at least one missing value. The original
  files mark them with `?`; OpenML already loads them as empty values.
- **Duplicates:** 52 rows repeat an earlier row exactly, and 5 groups of
  identical people have different income labels (10 rows).
- **Raw size:** about 0.7 MB in the OpenML cache. The processed CSVs are about
  27 MB and are not committed; their SHA256 is in `results/run_record.yaml`.

## Known biases

- **Time and place:** the data describes the US in 1994. The $50K threshold
  is in 1994 dollars, so it means a different living standard today, and the
  patterns do not carry over to other countries.
- **Who is in the data:** 66.8 % of the rows are men and 85.5 % are White;
  89.7 % were born in the United States. Smaller groups have few examples,
  so errors on them are less reliable.
- **Unequal label rates:** 30.4 % of men but 10.9 % of women earn over $50K.
  The rate is 25.4 % for White and 12.1 % for Black people. A model trained on
  this data learns these historical gaps and can repeat them in its
  predictions.
- **Proxy features:** relationship (Husband, Wife) and marital-status carry
  information about sex even if the sex column is removed.
- **Sampling weight as a feature:** fnlwgt is a census weight (how many people
  a row stands for), not a property of the person. It is kept because the lab
  uses all 14 features, but it should not be read as a personal attribute.
- **Label noise:** the 5 conflicting duplicate groups show that the same
  features can come with different labels.

## Privacy concerns

- The data has no names or addresses, but several columns together (age,
  sex, race, native-country, occupation, marital status) are quasi-identifiers.
  For people in small groups they could help re-identify someone.
- It contains sensitive personal information: race, sex, marital status,
  income and country of birth.
- The data is public and the people in it are anonymous. Still, a model
  trained on it should not be used for real decisions about people, such as
  credit, hiring or benefits. It is suitable only for teaching and research.

## Processing in this lab

- **Validation** runs on the raw data and reports violations without dropping
  rows.
- **Split:** a stratified 70/30 split, with seed 42, is done before any
  cleaning.
- **Fitted on the training rows only:**
  - median and mode imputation;
  - clipping at the 1st and 99th percentiles;
  - scaling and one-hot encoding.
- **Duplicates:** exact duplicates are removed from the training set only;
  the test set is never changed.
- **Derived features:**
  - `capital_net = capital-gain - capital-loss`;
  - `hours_per_age = hours-per-week / (age - 16)`.
