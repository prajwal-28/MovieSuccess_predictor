"""
Movie Success Predictor - Large Dataset Preprocessing
=====================================================
Processes the TMDB 5000 movies dataset.

Pipeline:
1. Load dataset & select relevant columns
2. Handle missing data & zero budget/revenue
3. Extract year and primary genre
4. Compute ROI and Success Label
5. Outlier removal (IQR) on ROI, budget, revenue
6. One-Hot Encode genres
7. Scale numeric features
8. Save finalized dataset
"""

import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler

def get_first_genre(genre_str):
    """Extract the first genre from the JSON-like string."""
    try:
        # Convert string representation of list to actual list
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list) and len(genres) > 0:
            return genres[0]['name']
    except (ValueError, SyntaxError):
        pass
    return "Unknown"

def main():
    print("🎬 Large Dataset Preprocessing Pipeline")
    print("======================================")

    # 1. Load dataset
    filepath = "tmdb_5000_movies.csv"
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded '{filepath}' -> {df.shape[0]} rows x {df.shape[1]} cols")
    except FileNotFoundError:
        print(f"[ERROR] '{filepath}' not found. Please ensure the dataset is in the directory.")
        return

    # 2. Select and rename columns
    columns_to_keep = [
        "budget", "revenue", "runtime", "vote_average", 
        "vote_count", "release_date", "genres"
    ]
    
    # Check if all required columns exist
    missing_cols = [c for c in columns_to_keep if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns in dataset: {missing_cols}")
        return

    df = df[columns_to_keep].copy()
    df.rename(columns={"vote_average": "rating", "vote_count": "votes"}, inplace=True)

    # 3. Handle missing values
    # Drop rows where budget or revenue is 0
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    print(f"  -> Rows after dropping zero budget/revenue: {len(df)}")

    # Fill other numeric missing values with median
    numeric_cols = ["runtime", "rating", "votes"]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. Process genres column (extract only first genre)
    df['genre'] = df['genres'].apply(get_first_genre)
    df = df[df['genre'] != "Unknown"] # Drop rows where genre couldn't be extracted

    # 5. Extract year from release_date
    df = df.dropna(subset=['release_date']) # drop rows without release date
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df.dropna(subset=['year']) # Drop if date format was invalid

    # 6. Create ROI
    df['roi'] = df['revenue'] / df['budget']

    # 7. Create target variable based on ROI
    def get_label(roi):
        if roi > 2:
            return "Hit"
        elif roi >= 1:
            return "Average"
        else:
            return "Flop"

    df['label'] = df['roi'].apply(get_label)

    # 8. Remove unnecessary columns
    df.drop(columns=['genres', 'release_date'], inplace=True)
    
    # Encode Target Variable (0: Average, 1: Flop, 2: Hit) to match model training logic
    label_mapping = {"Average": 0, "Flop": 1, "Hit": 2}
    df['label_encoded'] = df['label'].map(label_mapping)

    print(f"\n[INFO] Data after initial cleaning: {len(df)} rows")
    print("Class Distribution:")
    print(df['label'].value_counts())

    # 9. Outlier handling using IQR (only on ROI, budget, revenue)
    print("\n[INFO] Starting Outlier Detection (IQR)")
    outlier_cols = ['roi', 'budget', 'revenue']
    combined_mask = pd.Series(False, index=df.index)
    
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Mark rows that are outliers
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        combined_mask |= col_outlier_mask
        print(f"  - {col}: Found {col_outlier_mask.sum()} outliers")

    df_clean = df[~combined_mask].reset_index(drop=True)
    print(f"\n[INFO] Rows remaining after removing outliers: {len(df_clean)}")

    # 10. Encode genre using One-Hot Encoding
    genre_dummies = pd.get_dummies(df_clean['genre'], prefix='genre', dtype=int)
    df_clean = pd.concat([df_clean, genre_dummies], axis=1)
    df_clean.drop(columns=['genre'], inplace=True)

    # Drop original text label, keep encoded one
    df_clean.drop(columns=['label'], inplace=True)
    df_clean.rename(columns={'label_encoded': 'label'}, inplace=True)

    # 11. Apply StandardScaler on numeric features
    scale_cols = ["budget", "revenue", "rating", "votes", "runtime", "year", "roi"]
    scaler = StandardScaler()
    df_clean[scale_cols] = scaler.fit_transform(df_clean[scale_cols])

    # 12. Ensure dataset has at least 1000 rows
    if len(df_clean) < 1000:
        print(f"\n[WARNING] Dataset has only {len(df_clean)} rows, which is less than the required 1000.")
    else:
        print(f"\n[OK] Final dataset row count is good: {len(df_clean)} rows")

    # 13. Save final dataset
    output_filename = "final_movies_large.csv"
    
    # Reorder columns to place label at the end
    features = [c for c in df_clean.columns if c != 'label']
    df_clean = df_clean[features + ['label']]
    
    df_clean.to_csv(output_filename, index=False)
    
    print("\n" + "=" * 60)
    print(f" SAVED  ->  '{output_filename}'")
    print("=" * 60)
    print(f"  Shape   : {df_clean.shape[0]} rows x {df_clean.shape[1]} cols")
    print(f"  Target  : [0: Average, 1: Flop, 2: Hit]")

if __name__ == "__main__":
    main()
