# Project 1: House Price Prediction (End-to-End ML Pipeline)

A complete regression pipeline: data exploration, preprocessing, model comparison with cross-validation, and evaluation.

## Dataset
California Housing dataset (built into scikit-learn), about 20,000 districts.
Target: `MedHouseVal` (median house value, in units of $100,000).

## Method
- Exploratory analysis: target distribution and correlation matrix.
- Preprocessing with scikit-learn `Pipeline`: median imputation and standard scaling for numeric features, one-hot encoding for categorical ones.
- Models compared: Linear Regression, Ridge, Random Forest, Gradient Boosting.
- Evaluation: 5-fold cross-validation on the training set, then a held-out 20% test set (R2, MAE, RMSE).
- Permutation importance to see which features matter most.

## Results

| Model | CV R2 | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|---|
| Random Forest | 0.8056 | 0.8060 | 0.3268 | 0.5042 |
| Gradient Boosting | 0.7881 | 0.7756 | 0.3717 | 0.5422 |
| Ridge | 0.6115 | 0.5758 | 0.5332 | 0.7456 |
| Linear Regression | 0.6115 | 0.5758 | 0.5332 | 0.7456 |

Random Forest performed best (test R2 = 0.806). The tree-based models clearly beat the linear ones, which suggests the relationship between features and price is non-linear. Cross-validation and test scores are close, so there is no sign of serious overfitting.

Plots are saved in `outputs_p1/`.

## Limitations
- Hyperparameters were not tuned; default settings were used.
- Only one train/test split was used for the final test score.
- Next steps: hyperparameter search, feature engineering, and trying the same pipeline on a Bangladesh housing dataset.

## How to run
```
pip install -r requirements.txt
python p1_tabular_ml.py
```
To use your own data: `python p1_tabular_ml.py --csv data.csv --target price`
