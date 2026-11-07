"""
Python Project C: Streamlit CSV explorer + quick baseline model.
Upload any CSV, inspect it, plot it, and train a baseline model in one click.

    streamlit run p6_csv_explorer_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CSV Explorer", layout="wide")
st.title("CSV Explorer + Baseline Model")

file = st.file_uploader("Upload a CSV file", type="csv")
if file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df = pd.read_csv(file)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Plots", "Correlation", "Baseline model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Column types and missing values")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()})
    st.dataframe(info, use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

with tab2:
    if num_cols:
        col = st.selectbox("Numeric column (histogram)", num_cols)
        bins = st.slider("Bins", 5, 100, 30)
        counts, edges = np.histogram(df[col].dropna(), bins=bins)
        st.bar_chart(pd.DataFrame({"count": counts}, index=np.round(edges[:-1], 2)))
    if cat_cols:
        col = st.selectbox("Categorical column (top values)", cat_cols)
        st.bar_chart(df[col].value_counts().head(15))
    if len(num_cols) >= 2:
        x = st.selectbox("X", num_cols, index=0)
        y = st.selectbox("Y", num_cols, index=1)
        st.scatter_chart(df[[x, y]].dropna())

with tab3:
    if len(num_cols) >= 2:
        st.dataframe(df[num_cols].corr().round(2).style.background_gradient(cmap="coolwarm"), use_container_width=True)
    else:
        st.write("Need at least two numeric columns.")

with tab4:
    target = st.selectbox("Target column", df.columns)
    if st.button("Train baseline Random Forest"):
        data = df.dropna(subset=[target]).copy()
        X = pd.get_dummies(data.drop(columns=[target]), drop_first=True).fillna(0)
        y = data[target]
        if X.shape[1] == 0:
            st.error("No usable feature columns.")
            st.stop()
        is_classification = (y.dtype == object) or (y.nunique() <= 10)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, random_state=42) if is_classification else RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if is_classification:
            st.success(f"Classification accuracy: {accuracy_score(yte, pred):.3f}")
        else:
            st.success(f"Regression R2: {r2_score(yte, pred):.3f}")
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        st.subheader("Top feature importances")
        st.bar_chart(imp)
