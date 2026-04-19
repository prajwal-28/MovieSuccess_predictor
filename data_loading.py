"""
Movie Success Predictor - Data Loading & Exploration Module
============================================================
This module handles loading the movie dataset and performing
initial exploratory data analysis (EDA) using pandas and numpy.

Classification Target: Hit / Average / Flop
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
#  1. Data Loading
# ──────────────────────────────────────────────

def load_dataset(filepath: str = "movies.csv") -> pd.DataFrame:
    """
    Load the movie dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (default: "movies.csv").

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully from '{filepath}'")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found. Please check the path.")
        raise
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File '{filepath}' is empty.")
        raise
    except Exception as e:
        print(f"❌ Unexpected error while loading dataset: {e}")
        raise


# ──────────────────────────────────────────────
#  2. Basic Information
# ──────────────────────────────────────────────

def display_basic_info(df: pd.DataFrame) -> None:
    """
    Display the shape, column names, and data types of the DataFrame.
    """
    print("\n" + "=" * 60)
    print(" BASIC DATASET INFORMATION")
    print("=" * 60)

    # Dataset shape
    rows, cols = df.shape
    print(f"\n📐 Dataset Shape: {rows} rows × {cols} columns")

    # Column names
    print(f"\n📋 Column Names ({cols} total):")
    for i, col in enumerate(df.columns, start=1):
        print(f"   {i:>3}. {col}")

    # Data types
    print(f"\n🔤 Data Types:")
    print(df.dtypes.to_string())


# ──────────────────────────────────────────────
#  3. Preview Rows
# ──────────────────────────────────────────────

def show_first_rows(df: pd.DataFrame, n: int = 5) -> None:
    """
    Display the first `n` rows of the DataFrame.

    Parameters
    ----------
    n : int
        Number of rows to display (default: 5).
    """
    print("\n" + "=" * 60)
    print(f" FIRST {n} ROWS")
    print("=" * 60)
    print(df.head(n).to_string(index=False))


# ──────────────────────────────────────────────
#  4. Descriptive Statistics
# ──────────────────────────────────────────────

def show_descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Display descriptive statistics for both numeric and
    categorical columns.
    """
    print("\n" + "=" * 60)
    print(" DESCRIPTIVE STATISTICS  (Numeric Columns)")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("  ⚠️  No numeric columns found.")
    else:
        print(numeric_df.describe().round(2).to_string())

    # Categorical summary
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        print("\n" + "=" * 60)
        print(" DESCRIPTIVE STATISTICS  (Categorical Columns)")
        print("=" * 60)
        print(cat_df.describe().to_string())


# ──────────────────────────────────────────────
#  5. Missing Values
# ──────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and display missing values per column.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Column, Missing Count, Missing %.
    """
    print("\n" + "=" * 60)
    print(" MISSING VALUES REPORT")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.values,
    })

    # Sort by missing count descending for readability
    missing_report = missing_report.sort_values(
        by="Missing Count", ascending=False
    ).reset_index(drop=True)

    if missing_report["Missing Count"].sum() == 0:
        print("\n  🎉 No missing values found in any column!")
    else:
        print(missing_report.to_string(index=False))
        total = missing_report["Missing Count"].sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"\n  📊 Total missing cells: {total} / {total_cells} "
              f"({(total / total_cells * 100):.2f}%)")

    return missing_report


# ──────────────────────────────────────────────
#  Main Driver
# ──────────────────────────────────────────────

def main() -> None:
    """Run the full data-loading and exploration pipeline."""

    print("🎬 Movie Success Predictor — Data Loading & Exploration")
    print("=" * 60)

    # Step 1: Load
    df = load_dataset("movies.csv")

    # Step 2: Basic info
    display_basic_info(df)

    # Step 3: Preview
    show_first_rows(df)

    # Step 4: Statistics
    show_descriptive_statistics(df)

    # Step 5: Missing values
    check_missing_values(df)

    print("\n" + "=" * 60)
    print(" ✅ Data exploration complete. Ready for preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
