"""
Project 1: End-to-end tabular ML (regression)

Demo (works instantly, no download needed):
    python p1_tabular_ml.py

Your own dataset (e.g. Dhaka house prices from Kaggle):
    python p1_tabular_ml.py --csv dhaka_houses.csv --target price

Outputs go to ./outputs_p1/ : EDA plots, results table, best model, feature importance.
"""
import argparse
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = "outputs_p1"
SEED = 42


def load_data(csv_path, target):
    if csv_path:
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            raise SystemExit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
        return df, target
    from sklearn.datasets import fetch_california_housing

    df = fetch_california_housing(as_frame=True).frame
    return df, "MedHouseVal"


def run_eda(df, target):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nDescribe:\n", df.describe().T)

    plt.figure(figsize=(6, 4))
    df[target].hist(bins=50)
    plt.title(f"Distribution of {target}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/target_distribution.png", dpi=150)
    plt.close()

    num = df.select_dtypes(include=np.number)
    corr = num.corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT}/correlation.png", dpi=150)
    plt.close()


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default="price")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df, target = load_data(args.csv, args.target)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    run_eda(df, target)

    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(
            {
                "model": name,
                "cv_r2": round(cv_r2, 4),
                "test_r2": round(r2_score(y_test, pred), 4),
                "test_mae": round(mean_absolute_error(y_test, pred), 4),
                "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            }
        )
        fitted[name] = pipe
        print(rows[-1])

    results = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    results.to_csv(f"{OUT}/results.csv", index=False)
    print("\n", results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    joblib.dump(best, f"{OUT}/best_model.joblib")

    # Actual vs predicted
    pred = best.predict(X_test)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, pred, s=5, alpha=0.4)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{best_name}: actual vs predicted")
    plt.tight_layout()
    plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=150)
    plt.close()

    # Permutation importance on raw columns
    imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1)
    imp_df = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    plt.figure(figsize=(6, 4))
    imp_df.plot(kind="barh")
    plt.title("Permutation importance")
    plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name}. Everything saved in ./{OUT}/")


if __name__ == "__main__":
    main()
