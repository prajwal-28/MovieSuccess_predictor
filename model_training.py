"""
Movie Success Predictor - Model Training Module
================================================
Loads the final scaled dataset, trains three classifiers with
GridSearchCV hyperparameter tuning, evaluates all models, prints
a comparison table + confusion matrix for the best model, and
saves the best model as best_model.pkl.

Models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier

Constraints: sklearn, joblib (small dataset aware)
"""

import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")   # suppress convergence / UndefinedMetric


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

TARGET_COL  = "label"
LABEL_NAMES = {0: "Average", 1: "Flop", 2: "Hit"}
RANDOM_SEED = 42

# With a larger dataset, 5-fold CV is appropriate
CV_FOLDS = 5


# ──────────────────────────────────────────────
#  1. Load Dataset
# ──────────────────────────────────────────────

def load_dataset(filepath: str = "final_movies_large.csv") -> pd.DataFrame:
    """Load the final scaled movie dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded '{filepath}'  ->  {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print(f"[ERROR] '{filepath}' not found. Run data_scaling_outliers.py first.")
        raise


# ──────────────────────────────────────────────
#  2. Train-Test Split
# ──────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               test_size: float = 0.20,
               random_state: int = RANDOM_SEED):
    """
    Separate features / target and perform a stratified train-test split.

    stratify=y ensures all three classes appear in both splits
    even on a small dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Prevent Data Leakage: drop fields used to calculate the label
    X = df.drop(columns=[TARGET_COL, "budget", "revenue", "roi"], errors="ignore")
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print(" TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"\n  Total samples : {len(df)}")
    print(f"  Train size    : {len(X_train)} rows ({(1-test_size)*100:.0f}%)")
    print(f"  Test size     : {len(X_test)}  rows ({test_size*100:.0f}%)")
    print(f"\n  Train label distribution:")
    for code, name in LABEL_NAMES.items():
        print(f"    {code} ({name})  ->  {(y_train == code).sum()} samples")
    print(f"\n  Test label distribution:")
    for code, name in LABEL_NAMES.items():
        print(f"    {code} ({name})  ->  {(y_test == code).sum()} samples")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
#  3. GridSearchCV Helper
# ──────────────────────────────────────────────

def tune_model(estimator,
               param_grid: dict,
               X_train,
               y_train,
               cv_folds: int = CV_FOLDS,
               label: str = "") -> object:
    """
    Run GridSearchCV with StratifiedKFold on the given estimator.

    Parameters
    ----------
    estimator  : sklearn estimator
    param_grid : dict of hyperparameter options
    label      : display name for printing

    Returns
    -------
    best estimator (already fitted on X_train)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",      # macro F1 — fair across unbalanced classes
        n_jobs=-1,
        refit=True,              # re-fit best params on full training set
    )
    grid.fit(X_train, y_train)

    print(f"\n  [{label}]")
    print(f"    Best Params : {grid.best_params_}")
    print(f"    CV F1-macro : {grid.best_score_:.4f}")

    return grid.best_estimator_


# ──────────────────────────────────────────────
#  4. Train All Three Models
# ──────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """
    Train Logistic Regression, Decision Tree, and Random Forest
    with GridSearchCV.

    Returns
    -------
    dict  { model_name: fitted_best_estimator }
    """
    print("\n" + "=" * 60)
    print(" MODEL TRAINING  (GridSearchCV)")
    print("=" * 60)

    # ── Logistic Regression ──────────────────
    lr = tune_model(
        estimator=LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        ),
        param_grid={
            "C": [0.1, 1, 10],
        },
        X_train=X_train,
        y_train=y_train,
        label="Logistic Regression",
    )

    # ── Decision Tree ────────────────────────
    dt = tune_model(
        estimator=DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        param_grid={
            "max_depth": [3, 5, 10],
        },
        X_train=X_train,
        y_train=y_train,
        label="Decision Tree",
    )

    # ── Random Forest ────────────────────────
    rf = tune_model(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
        },
        X_train=X_train,
        y_train=y_train,
        label="Random Forest",
    )

    return {
        "Logistic Regression": lr,
        "Decision Tree":       dt,
        "Random Forest":       rf,
    }


# ──────────────────────────────────────────────
#  5. Evaluate a Single Model
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute Accuracy, Precision, Recall, and F1-score (macro avg).

    Returns
    -------
    dict of metric values for the comparison table
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
    }


# ──────────────────────────────────────────────
#  6. Comparison Table
# ──────────────────────────────────────────────

def compare_models(models: dict, X_test, y_test) -> tuple[pd.DataFrame, str]:
    """
    Evaluate all models and print a side-by-side comparison table.

    Returns
    -------
    results_df : pd.DataFrame  — metrics for all models
    best_name  : str           — name of the best model (by F1-Score)
    """
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION — COMPARISON TABLE")
    print("=" * 60)

    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        rows.append(metrics)

    results_df = pd.DataFrame(rows).set_index("Model")

    print("\n" + results_df.to_string())

    # Best model = highest macro F1-Score
    best_name = results_df["F1-Score"].idxmax()
    print(f"\n  [BEST MODEL]  '{best_name}'  "
          f"->  F1-Score = {results_df.loc[best_name, 'F1-Score']:.4f}")

    return results_df, best_name


# ──────────────────────────────────────────────
#  7. Confusion Matrix for Best Model
# ──────────────────────────────────────────────

def show_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    """
    Print a formatted confusion matrix and full classification report
    for the best model.
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print(f" CONFUSION MATRIX  ({model_name})")
    print("=" * 60)

    # Header row
    class_labels = [f"Pred:{LABEL_NAMES[i]}" for i in [0, 1, 2]]
    header       = f"{'Actual':<18}" + "  ".join(f"{c:>14}" for c in class_labels)
    print("\n  " + header)
    print("  " + "-" * (len(header) + 2))

    for i, row in enumerate(cm):
        actual_label = f"Act:{LABEL_NAMES[i]}"
        row_str = "  ".join(f"{v:>14}" for v in row)
        print(f"  {actual_label:<18}{row_str}")

    # Classification report
    print(f"\n{'─'*60}")
    print(f" CLASSIFICATION REPORT  ({model_name})")
    print(f"{'─'*60}")
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0,
    ))


# ──────────────────────────────────────────────
#  8. Save Best Model
# ──────────────────────────────────────────────

def save_best_model(model,
                    model_name: str,
                    output_path: str = "best_model.pkl") -> None:
    """
    Persist the best model to disk using joblib.

    Parameters
    ----------
    model       : fitted sklearn estimator
    model_name  : display name for logging
    output_path : destination file path
    """
    joblib.dump(model, output_path)

    print("\n" + "=" * 60)
    print(f" MODEL SAVED  ->  '{output_path}'")
    print("=" * 60)
    print(f"  Model type : {type(model).__name__}")
    print(f"  Model name : {model_name}")
    print(f"  Load with  : joblib.load('{output_path}')")


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

def main() -> None:
    """Run the complete model training pipeline."""

    print("Movie Success Predictor -- Model Training")
    print("=" * 60)
    print(f"  Random seed  : {RANDOM_SEED}")
    print(f"  CV folds     : {CV_FOLDS}")
    print(f"  CV scoring   : macro F1")

    # Step 1 — Load
    df = load_dataset("final_movies_large.csv")

    # Step 2 — Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 3 & 4 — Train with GridSearchCV
    models = train_models(X_train, y_train)

    # Step 5 & 6 — Evaluate and compare
    results_df, best_name = compare_models(models, X_test, y_test)

    # Step 7 — Confusion matrix for best model
    show_confusion_matrix(models[best_name], X_test, y_test, best_name)

    # Step 8 — Save best model
    save_best_model(models[best_name], best_name, "best_model.pkl")

    print("\n" + "=" * 60)
    print(" Training complete. Run predict.py to make predictions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
