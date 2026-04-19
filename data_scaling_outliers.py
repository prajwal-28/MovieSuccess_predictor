"""
Movie Success Predictor - Outlier Detection & Feature Scaling Module
=====================================================================
Reads the preprocessed dataset, removes outliers via the IQR method,
applies StandardScaler to numerical features, and saves the final
clean dataset ready for model training.

Pipeline:
  1. Load processed_movies.csv
  2. Detect & remove outliers (IQR) on selected numeric columns
  3. Apply StandardScaler to numeric features
  4. Separate features (X) and target (y)
  5. Save final_movies.csv

Constraints: pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
#  Column groups (single source of truth)
# ──────────────────────────────────────────────

# Columns to check for outliers
OUTLIER_COLS = ["budget", "revenue", "rating", "votes", "runtime", "roi"]

# Columns to scale (numeric features, excluding one-hot dummies & target)
SCALE_COLS   = ["budget", "revenue", "rating", "votes", "runtime", "year", "roi"]

TARGET_COL   = "label"


# ──────────────────────────────────────────────
#  1. Load Dataset
# ──────────────────────────────────────────────

def load_dataset(filepath: str = "processed_movies.csv") -> pd.DataFrame:
    """Load the preprocessed movie dataset from CSV."""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded '{filepath}'  ->  {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"[ERROR] '{filepath}' not found. Run data_preprocessing.py first.")
        raise


# ──────────────────────────────────────────────
#  2. Outlier Detection & Removal (IQR Method)
# ──────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Compute a boolean mask of outliers for a single column using the
    Interquartile Range (IQR) method.

    Outlier bounds:
        Lower = Q1 - 1.5 * IQR
        Upper = Q3 + 1.5 * IQR

    Parameters
    ----------
    series : pd.Series  — numeric column

    Returns
    -------
    pd.Series (bool) — True where value IS an outlier
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)


def remove_outliers(df: pd.DataFrame,
                    cols: list = None) -> pd.DataFrame:
    """
    Detect and remove rows containing outliers across all specified columns.

    Prints:
      - Row count before and after removal
      - Number of outlier rows flagged per column

    Parameters
    ----------
    df   : pd.DataFrame  — input data
    cols : list          — numeric columns to inspect (default: OUTLIER_COLS)

    Returns
    -------
    pd.DataFrame  with outlier rows removed
    """
    if cols is None:
        cols = OUTLIER_COLS

    # Keep only columns that exist in the dataframe
    cols = [c for c in cols if c in df.columns]

    rows_before = len(df)

    print("\n" + "=" * 60)
    print(" OUTLIER DETECTION  (IQR Method)")
    print("=" * 60)
    print(f"\n  Rows before removal : {rows_before}")
    print(f"  Columns inspected   : {cols}\n")

    # Build a combined outlier mask (union of all columns)
    combined_mask = pd.Series(False, index=df.index)

    col_results = []
    for col in cols:
        col_mask   = detect_outliers_iqr(df[col])
        n_outliers = col_mask.sum()
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = round(Q1 - 1.5 * IQR, 2)
        upper = round(Q3 + 1.5 * IQR, 2)

        col_results.append({
            "Column":    col,
            "Q1":        round(Q1, 2),
            "Q3":        round(Q3, 2),
            "IQR":       round(IQR, 2),
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outliers":  int(n_outliers),
        })

        combined_mask |= col_mask

    # Print per-column summary table
    summary_df = pd.DataFrame(col_results)
    print(summary_df.to_string(index=False))

    # Remove outlier rows
    df_clean = df[~combined_mask].reset_index(drop=True)
    rows_after   = len(df_clean)
    rows_removed = rows_before - rows_after

    print(f"\n  Rows after removal  : {rows_after}")
    print(f"  Total rows removed  : {rows_removed}")

    if rows_removed == 0:
        print("\n  [INFO] No outlier rows were removed (dataset is clean).")

    return df_clean


# ──────────────────────────────────────────────
#  3. StandardScaler on Numeric Features
# ──────────────────────────────────────────────

def scale_features(df: pd.DataFrame,
                   scale_cols: list = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to the specified numeric columns.

    StandardScaler transforms each column to:
        z = (x - mean) / std  →  mean ≈ 0, std ≈ 1

    One-hot dummy columns and the target are left untouched.

    Parameters
    ----------
    df         : pd.DataFrame  — data after outlier removal
    scale_cols : list          — columns to scale (default: SCALE_COLS)

    Returns
    -------
    df_scaled  : pd.DataFrame       — full dataframe with scaled columns
    scaler     : StandardScaler     — fitted scaler (reuse in inference)
    """
    if scale_cols is None:
        scale_cols = SCALE_COLS

    # Only scale columns that actually exist
    scale_cols = [c for c in scale_cols if c in df.columns]

    df_scaled = df.copy()
    scaler    = StandardScaler()

    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    print("\n" + "=" * 60)
    print(" FEATURE SCALING  (StandardScaler)")
    print("=" * 60)
    print(f"\n  Scaled columns: {scale_cols}")

    # Verification: post-scaling stats for scaled columns
    stats = df_scaled[scale_cols].agg(["mean", "std"]).round(4)
    print("\n  Post-scaling verification (should be mean~0, std~1):")
    print(stats.to_string())

    return df_scaled, scaler


# ──────────────────────────────────────────────
#  4. Separate Features and Target
# ──────────────────────────────────────────────

def split_features_target(df: pd.DataFrame,
                           target_col: str = TARGET_COL):
    """
    Split the DataFrame into feature matrix X and target series y.

    Parameters
    ----------
    df         : pd.DataFrame  — fully processed data
    target_col : str           — name of the target column

    Returns
    -------
    X : pd.DataFrame  — features
    y : pd.Series     — integer-encoded target
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print("\n" + "=" * 60)
    print(" FEATURE / TARGET SPLIT")
    print("=" * 60)
    print(f"\n  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"\n  Feature columns ({len(X.columns)}):")

    # Print in groups of 4 for readability
    cols = X.columns.tolist()
    for i in range(0, len(cols), 4):
        print("   ", "  |  ".join(f"{c}" for c in cols[i:i+4]))

    print(f"\n  Target distribution:")
    label_map = {0: "Average", 1: "Flop", 2: "Hit"}
    dist = y.value_counts().sort_index()
    for code, count in dist.items():
        print(f"    {code} ({label_map.get(code, code)})  ->  {count} rows")

    return X, y


# ──────────────────────────────────────────────
#  5. Save Final Dataset
# ──────────────────────────────────────────────

def save_final_dataset(X: pd.DataFrame,
                       y: pd.Series,
                       output_path: str = "final_movies.csv") -> None:
    """
    Merge features and target then save to CSV.

    Parameters
    ----------
    X           : pd.DataFrame  — scaled feature matrix
    y           : pd.Series     — target labels
    output_path : str           — destination file path
    """
    final_df = pd.concat([X, y], axis=1)
    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" SAVED  ->  '{output_path}'")
    print("=" * 60)
    print(f"  Shape   : {final_df.shape[0]} rows x {final_df.shape[1]} cols")
    print(f"  Columns : {final_df.columns.tolist()}")


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

def main() -> None:
    """Run the complete outlier-removal and scaling pipeline."""

    print("Movie Success Predictor -- Outlier Detection & Scaling")
    print("=" * 60)

    # Step 1 — Load
    df = load_dataset("processed_movies.csv")

    # Step 2 — Remove outliers via IQR
    df = remove_outliers(df, cols=OUTLIER_COLS)

    # Step 3 — Scale numeric features
    df_scaled, scaler = scale_features(df, scale_cols=SCALE_COLS)

    # Step 4 — Separate features and target
    X, y = split_features_target(df_scaled)

    # Step 5 — Save
    save_final_dataset(X, y, "final_movies.csv")

    print("\n" + "=" * 60)
    print(" Scaling complete. Ready for model training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
