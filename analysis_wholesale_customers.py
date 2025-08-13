"""
Wholesale Customers — Clustering & EDA
--------------------------------------

This script:
1) Ensures the dataset is present (downloads from UCI on first run).
2) Performs EDA & preprocessing.
3) Runs K-Means with K selection (Elbow & Silhouette).
4) Reduces to 2D with PCA and visualizes clusters.
5) Exports artifacts & a concise Markdown report.

Run:
    python src/analysis_wholesale_customers.py
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import ensure_dataset, load_data, describe_numeric, FEATURE_COLS, LOCAL_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_TEMPLATE = os.path.join(PROJECT_ROOT, "report_template.md")

os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    # 1) Data -----------------------------------------------------------------
    ensure_dataset(LOCAL_PATH)
    df = load_data(LOCAL_PATH)

    # Some versions include 'Channel' and 'Region'; keep them for profiling but exclude from scaling
    meta_cols = [c for c in ["Channel","Region"] if c in df.columns]
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    if not feature_cols:
        raise RuntimeError("No expected feature columns found.")

    # 2) EDA ------------------------------------------------------------------
    eda_summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "missing_by_col": df.isna().sum().to_dict(),
        "describe": describe_numeric(df[feature_cols]),
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(OUT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    # Correlation
    corr = df[feature_cols].corr()
    corr.to_csv(os.path.join(OUT_DIR, "corr_matrix.csv"), index=True)

    # 3) Preprocessing ---------------------------------------------------------
    # Log1p transform to reduce skew, then RobustScaler (resistant to outliers)
    X = df[feature_cols].copy()
    X_log = np.log1p(X)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_log)
    pd.DataFrame(X_scaled, columns=[f"{c}_scaled" for c in feature_cols]).to_csv(
        os.path.join(OUT_DIR, "scaled_features.csv"), index=False
    )

    # PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
    pca_df.to_csv(os.path.join(OUT_DIR, "pca_2d.csv"), index=False)

    # 4) Model selection: Elbow & Silhouette ---------------------------------
    k_values = list(range(2, 11))
    inertias = []
    silhouettes = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Elbow plot
    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("k (clusters)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Plot — KMeans")
    save_fig(os.path.join(OUT_DIR, "elbow_plot.png"))

    # Silhouette plot
    plt.figure()
    plt.plot(k_values, silhouettes, marker="o")
    plt.xlabel("k (clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores vs k")
    save_fig(os.path.join(OUT_DIR, "silhouette_plot.png"))

    # Choose k with max silhouette (break ties with elbow min inertia trend)
    best_k = int(k_values[int(np.argmax(silhouettes))])

    # 5) Final model -----------------------------------------------------------
    kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_clusters = df.copy()
    df_clusters["cluster"] = labels
    df_clusters.to_csv(os.path.join(OUT_DIR, "clustered_customers.csv"), index=False)

    # Cluster profiles (means on original scale)
    profiles = df_clusters.groupby("cluster")[feature_cols].mean().sort_index()
    profiles["size"] = df_clusters.groupby("cluster").size().sort_index()
    profiles.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"))

    # 6) Visualization in PCA space ------------------------------------------
    plt.figure()
    for cl in sorted(np.unique(labels)):
        idx = labels == cl
        plt.scatter(pca_df.loc[idx, "PC1"], pca_df.loc[idx, "PC2"], label=f"Cluster {cl}", alpha=0.8)
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA (2D) — KMeans, k={best_k}")
    save_fig(os.path.join(OUT_DIR, "pca_clusters.png"))

    # 7) Short report ----------------------------------------------------------
    explained = pca.explained_variance_ratio_.round(4)
    elbow_hint = "See elbow_plot.png and silhouette_plot.png in outputs/."
    report = open(REPORT_TEMPLATE, "r", encoding="utf-8").read()
    report = report.format(
        datetime=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        features=", ".join(feature_cols),
        best_k=best_k,
        pc1=explained[0] if len(explained)>0 else 0,
        pc2=explained[1] if len(explained)>1 else 0,
        elbow_hint=elbow_hint
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Done. Best k = {best_k}. See outputs/ for results.")

if __name__ == "__main__":
    main()