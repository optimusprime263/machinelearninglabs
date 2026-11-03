# Validation report

Raw dataset: 48842 rows. Checks only report; no row was dropped or changed at this step.

| Check | Column | Violations | Detail |
|---|---|---|---|
| columns | all | 0 | 15 columns; missing none; extra none |
| type | age | 0 | expected int, found int64 |
| type | workclass | 0 | expected category, found category |
| type | fnlwgt | 0 | expected int, found int64 |
| type | education | 0 | expected category, found category |
| type | education-num | 0 | expected int, found int64 |
| type | marital-status | 0 | expected category, found category |
| type | occupation | 0 | expected category, found category |
| type | relationship | 0 | expected category, found category |
| type | race | 0 | expected category, found category |
| type | sex | 0 | expected category, found category |
| type | capital-gain | 0 | expected int, found int64 |
| type | capital-loss | 0 | expected int, found int64 |
| type | hours-per-week | 0 | expected int, found int64 |
| type | native-country | 0 | expected category, found category |
| type | income | 0 | expected category, found category |
| missing | age | 0 | 0 empty + 0 '?' |
| missing | workclass | 0 | 2799 empty + 0 '?' |
| missing | fnlwgt | 0 | 0 empty + 0 '?' |
| missing | education | 0 | 0 empty + 0 '?' |
| missing | education-num | 0 | 0 empty + 0 '?' |
| missing | marital-status | 0 | 0 empty + 0 '?' |
| missing | occupation | 0 | 2809 empty + 0 '?' |
| missing | relationship | 0 | 0 empty + 0 '?' |
| missing | race | 0 | 0 empty + 0 '?' |
| missing | sex | 0 | 0 empty + 0 '?' |
| missing | capital-gain | 0 | 0 empty + 0 '?' |
| missing | capital-loss | 0 | 0 empty + 0 '?' |
| missing | hours-per-week | 0 | 0 empty + 0 '?' |
| missing | native-country | 0 | 857 empty + 0 '?' |
| missing | income | 0 | 0 empty + 0 '?' |
| range | age | 0 | allowed 17-90, found 17-90; 0 below, 0 above |
| range | hours-per-week | 0 | allowed 1-99, found 1-99; 0 below, 0 above |
| range | capital-gain | 0 | allowed 0-99999, found 0-99999; 0 below, 0 above |
| range | capital-loss | 0 | allowed 0-4356, found 0-4356; 0 below, 0 above |
| category | workclass | 0 | 8 allowed values |
| category | education | 0 | 16 allowed values |
| category | marital-status | 0 | 7 allowed values |
| category | occupation | 0 | 14 allowed values |
| category | relationship | 0 | 6 allowed values |
| category | race | 0 | 5 allowed values |
| category | sex | 0 | 2 allowed values |
| category | native-country | 0 | 41 allowed values |
| category | income | 0 | 2 allowed values |
| duplicates | all | 52 | 52 rows repeat an earlier row exactly (reported here; removed later from the training set only) |
| dictionary | education-num does not match education | 0 | 0 rows |
| dictionary | Husband but Female | 1 | 1 rows: row ids [23390] |
| dictionary | Wife but Male | 3 | 3 rows: row ids [5661, 16856, 43422] |
| dictionary | Husband/Wife but not married | 0 | 0 rows |
| label | income | 0 | <=50K: 37155, >50K: 11687 |

Violations outside duplicates: 4. Exact duplicate rows: 52.
