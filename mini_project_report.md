# Movie Success Predictor using Machine Learning
**Mini Project Report**

## Problem Statement
* Predicts whether a movie will be a Hit, Average, or Flop.
* Evaluates using movie metadata and financial data.
* Solves the uncertainty of box office performance.

## Objectives
* Perform extensive data analysis on box office trends.
* Implement structured data preprocessing flows.
* Engineer custom features for classification.
* Train multiple machine learning models.
* Compare model metrics to select the standalone best performer.
* Build an interactive, user-facing web application.

## Dataset Description
* **Total Records:** ~25 custom movie sample cases.
* **Features Used:** Budget, Revenue, Rating, Votes, Runtime, Release Year, Genre.
* **Target Variable:** Status (Hit / Average / Flop) derived mathematically from ROI.

## Data Preprocessing
* Handled missing numerical values systematically using median imputation.
* Encoded categorical attributes (Genre) using One-Hot Encoding schemas.
* Removed non-predictive baseline fields (Director, Title).

## EDA (Exploratory Data Analysis)
* Evaluated budget vs. revenue correlations effectively.
* Discovered actionable genre distribution impacts.
* Compiled summary statistics highlighting standard deviations and averages.

## Outlier Handling
* Implemented the IQR (Interquartile Range) methodology.
* Automatically detected and dropped extreme values to prevent artificial data skew.

## Feature Engineering
* Calculated Return on Investment (ROI = Revenue / Budget).
* Mapped definitive thresholds: ROI > 2 (Hit), ROI > 1 (Average), else (Flop).

## Data Normalization
* Standardized features mapping mean to ~0 and standard deviation to ~1.
* Applied `StandardScaler` to ensure scale-independent algorithms evaluated effectively.

## Train/Test Split
* Split data exactly into 80% Training and 20% Testing sets.
* Enforced `stratify` checks to guarantee class equilibrium on small datasets.

## Model Training
* Logistic Regression (Baseline linearity test).
* Decision Tree (Non-linear rule-based evaluator).
* Random Forest (Ensemble meta-estimator testing robustness).

## Hyperparameter Tuning
* Automated testing via `GridSearchCV`.
* Tested depths, leaves, iterations, and estimator counts simultaneously.

## Model Evaluation
* Scored models using Accuracy, Precision, and Recall metrics.
* **Selected Model:** Random Forest Classifier. selected based on highest cross-validated macro-F1 score mappings during training.

## System Build
* Developed an independent RESTful backend utilizing a Flask API.
* Designed a responsive, glassmorphism-styled frontend UI via HTML/CSS/JavaScript.

## Deployment Strategy
* **Backend:** Cloud-hosted live processing endpoint hosted on Render.
* **Frontend:** Static, visually-rich UI deployed globally via Vercel.

## Conclusion
* Project successfully demonstrates a complete end-to-end Machine Learning pipeline.
* Transitions complex, unstructured metrics into confident class predictions.
* Proves capable of effectively scaling to larger scale catalogs natively.
