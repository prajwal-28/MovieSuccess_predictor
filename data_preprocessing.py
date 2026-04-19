"""
Movie Success Predictor - Data Preprocessing Module
=====================================================
Handles all data cleaning and feature engineering steps
before model training.

Pipeline:
  1. Load raw dataset
  2. Handle missing values (median imputation)
  3. Create ROI-based target column
  4. Drop unnecessary columns
  5. Encode categorical features (Label + One-Hot)
  6. Separate features (X) and target (y)
  7. Save processed dataset

Constraints: pandas, numpy, sklearn (encoding only)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
#  1. Load Dataset
# ──────────────────────────────────────────────

def load_dataset(filepath: str = "movies.csv") -> pd.DataFrame:
    """Load the raw movie dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded '{filepath}'  ->  {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File '{filepath}' not found.")
        raise


# ──────────────────────────────────────────────
#  2. Handle Missing Values
# ──────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using column medians for numeric columns.
    Prints a before/after report.

    Parameters
    ----------
    df : pd.DataFrame  (modified in-place copy)

    Returns
    -------
    pd.DataFrame  with no missing numeric values
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print(" MISSING VALUES — BEFORE IMPUTATION")
    print("=" * 60)

    before = df.isnull().sum()
    before_report = before[before > 0]

    if before_report.empty:
        print("  No missing values found.\n")
        return df

    print(before_report.to_string())

    # Identify numeric columns with missing data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed_cols = []

    for col in numeric_cols:
        missing_n = df[col].isnull().sum()
        if missing_n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_cols.append((col, missing_n, round(median_val, 2)))

    # Report after
    print("\n" + "=" * 60)
    print(" MISSING VALUES — AFTER IMPUTATION")
    print("=" * 60)
    after = df.isnull().sum().sum()

    for col, n, med in imputed_cols:
        print(f"  [FILLED] '{col}'  ->  {n} value(s) replaced with median = {med}")

    print(f"\n  Total remaining missing values: {after}")
    return df


# ──────────────────────────────────────────────
#  3. Create ROI-Based Target Column
# ──────────────────────────────────────────────

def create_roi_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ROI and derive the 'label' target column.

    ROI = revenue / budget
      > 2.0  ->  Hit
      1.0–2.0 ->  Average
      < 1.0  ->  Flop

    Note: The original 'status' column (if present) is dropped
    and replaced by the ROI-derived 'label' for consistency.
    """
    df = df.copy()

    # Guard against division by zero
    df["roi"] = np.where(
        df["budget"] > 0,
        df["revenue"] / df["budget"],
        np.nan
    )

    def classify_roi(roi):
        if pd.isna(roi):
            return "Unknown"
        elif roi > 2.0:
            return "Hit"
        elif roi >= 1.0:
            return "Average"
        else:
            return "Flop"

    df["label"] = df["roi"].apply(classify_roi)

    print("\n" + "=" * 60)
    print(" ROI-BASED TARGET COLUMN  ('label')")
    print("=" * 60)
    print(df[["title", "budget", "revenue", "roi", "label"]]
          .round({"roi": 3})
          .to_string(index=False))

    print("\n  Class distribution:")
    print(df["label"].value_counts().to_string())

    # Drop original status column — replaced by ROI-derived label
    if "status" in df.columns:
        df.drop(columns=["status"], inplace=True)
        print("\n  [DROPPED] 'status' column (replaced by ROI-derived 'label')")

    return df


# ──────────────────────────────────────────────
#  4. Drop Unnecessary Columns
# ──────────────────────────────────────────────

def drop_unnecessary_columns(df: pd.DataFrame,
                              cols_to_drop: list = None) -> pd.DataFrame:
    """
    Drop columns that are not useful for model training.

    Default drop list: ['title', 'director']
      - 'title'    : unique identifier, not a predictive feature
      - 'director' : too many unique values for simple encoding;
                     can be revisited later with target encoding

    Parameters
    ----------
    cols_to_drop : list, optional
        Override the default list of columns to drop.
    """
    df = df.copy()

    if cols_to_drop is None:
        cols_to_drop = ["title", "director"]

    # Only drop columns that actually exist
    existing = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)

    print("\n" + "=" * 60)
    print(" DROPPED COLUMNS")
    print("=" * 60)
    print(f"  Removed: {existing}")
    print(f"  Remaining columns: {df.columns.tolist()}")

    return df


# ──────────────────────────────────────────────
#  5. Encode Categorical Columns
# ──────────────────────────────────────────────

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:

    - 'genre'  : One-Hot Encoding (low cardinality, interpretable)
    - Any other remaining object columns: Label Encoding

    The target column 'label' is NOT encoded here; it is handled
    separately in step 6.

    Returns
    -------
    pd.DataFrame  with all features numeric
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print(" CATEGORICAL ENCODING")
    print("=" * 60)

    # --- One-Hot Encoding for 'genre' ---
    if "genre" in df.columns:
        genre_dummies = pd.get_dummies(df["genre"], prefix="genre", dtype=int)
        df = pd.concat([df.drop(columns=["genre"]), genre_dummies], axis=1)
        print(f"  [ONE-HOT] 'genre'  ->  {genre_dummies.columns.tolist()}")

    # --- Label Encoding for any remaining object columns (except target) ---
    le = LabelEncoder()
    remaining_cat = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != "label"
    ]

    for col in remaining_cat:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [LABEL]   '{col}'  ->  integer encoded")

    print(f"\n  Final column count: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────
#  6. Separate Features and Target
# ──────────────────────────────────────────────

def split_features_target(df: pd.DataFrame,
                           target_col: str = "label"):
    """
    Separate the DataFrame into feature matrix X and target series y.

    Also label-encodes the target for downstream ML compatibility
    and prints the label mapping.

    Returns
    -------
    X : pd.DataFrame  — feature columns
    y : pd.Series     — encoded integer target
    label_map : dict  — integer -> class name mapping
    """
    print("\n" + "=" * 60)
    print(" FEATURE / TARGET SPLIT")
    print("=" * 60)

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encode target labels to integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name="label")

    label_map = dict(enumerate(le.classes_))
    print(f"  Target encoding map: {label_map}")
    print(f"\n  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"\n  Feature columns:\n  {X.columns.tolist()}")

    return X, y, label_map


# ──────────────────────────────────────────────
#  7. Save Processed Dataset
# ──────────────────────────────────────────────

def save_processed_dataset(X: pd.DataFrame,
                            y: pd.Series,
                            output_path: str = "processed_movies.csv") -> None:
    """
    Concatenate features and target, then save as a CSV file.

    Parameters
    ----------
    X           : feature DataFrame
    y           : encoded target Series
    output_path : destination file path
    """
    processed = pd.concat([X, y], axis=1)
    processed.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" SAVED  ->  '{output_path}'")
    print("=" * 60)
    print(f"  Shape  : {processed.shape[0]} rows x {processed.shape[1]} cols")
    print(f"  Columns: {processed.columns.tolist()}")


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

def main() -> None:
    """Run the complete preprocessing pipeline."""

    print("Movie Success Predictor -- Data Preprocessing")
    print("=" * 60)

    # Step 1 — Load
    df = load_dataset("movies.csv")

    # Step 2 — Handle missing values
    df = handle_missing_values(df)

    # Step 3 — Create ROI-based target column
    df = create_roi_target(df)

    # Step 4 — Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Step 5 — Encode categorical columns
    df = encode_categorical_columns(df)

    # Step 6 — Separate features and target
    X, y, label_map = split_features_target(df)

    # Step 7 — Save
    save_processed_dataset(X, y, "processed_movies.csv")

    print("\n" + "=" * 60)
    print(" Preprocessing complete. Ready for model training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
