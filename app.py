"""
Movie Success Predictor - Flask Backend API
===========================================
Serves a trained ML model via a REST endpoint.

Endpoint:
    POST /predict
    Content-Type: application/json

Example request body:
    {
        "budget":  160000000,
        "revenue": 836800000,
        "rating":  8.8,
        "votes":   2200000,
        "runtime": 148,
        "year":    2010,
        "roi":     5.23,
        "genre":   "Sci-Fi"
    }

Response:
    { "prediction": "Hit" }

Run:
    python app.py
"""

import os
import joblib
from flask_cors import CORS
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ──────────────────────────────────────────────
#  Constants — must match training pipeline
# ──────────────────────────────────────────────

MODEL_PATH = "best_model.pkl"

# Exact column order used during model training (final_movies.csv)
FEATURE_COLUMNS = [
    "budget", "revenue", "rating", "votes",
    "runtime", "year", "roi",
    "genre_Action", "genre_Animation", "genre_Crime",
    "genre_Drama", "genre_Musical", "genre_Romance",
    "genre_Sci-Fi", "genre_Thriller", "genre_Western",
]

# Numeric fields that come directly from the request body
NUMERIC_FIELDS = ["budget", "revenue", "rating", "votes", "runtime", "year", "roi"]

# Valid genre values accepted in the API
VALID_GENRES = [
    "Action", "Animation", "Crime",
    "Drama", "Musical", "Romance",
    "Sci-Fi", "Thriller", "Western",
]

# Label map: integer output → class name
LABEL_MAP = {0: "Average", 1: "Flop", 2: "Hit"}

# Scaling parameters derived from training data (final_movies.csv)
# These reproduce StandardScaler without re-fitting at request time,
# keeping the API stateless and fast.
# Exact values from StandardScaler fit on post-outlier-removal training data
SCALE_MEANS = {
    "budget":  1.448421e+08,
    "revenue": 3.871526e+08,
    "rating":  7.315789e+00,
    "votes":   9.625263e+05,
    "runtime": 1.357895e+02,
    "year":    2.010526e+03,
    "roi":     3.529841e+00,
}
SCALE_STDS = {
    "budget":  8.322466e+07,
    "revenue": 2.765200e+08,
    "rating":  1.502371e+00,
    "votes":   8.628671e+05,
    "runtime": 2.040273e+01,
    "year":    8.267682e+00,
    "roi":     3.537916e+00,
}


# ──────────────────────────────────────────────
#  App Initialisation
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from the frontend

def load_model(path: str = MODEL_PATH):
    """Load the pickled sklearn model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. "
            "Run model_training.py first."
        )
    model = joblib.load(path)
    print(f"[OK] Model loaded: {type(model).__name__}  <-  '{path}'")
    return model

# Load once at startup — avoids repeated disk I/O on every request
model = load_model()


# ──────────────────────────────────────────────
#  Input Validation
# ──────────────────────────────────────────────

def validate_input(data: dict) -> tuple[dict | None, str | None]:
    """
    Validate and coerce incoming JSON payload.

    Checks:
      - All required fields are present
      - Numeric fields can be cast to float
      - Genre is one of the known categories
      - No negative budget / revenue

    Returns
    -------
    (clean_data, None)  on success
    (None, error_msg)   on failure
    """
    required_fields = NUMERIC_FIELDS + ["genre"]

    # ── 1. Presence check ──────────────────────
    missing = [f for f in required_fields if f not in data]
    if missing:
        return None, f"Missing required fields: {missing}"

    clean = {}

    # ── 2. Numeric type & range checks ─────────
    for field in NUMERIC_FIELDS:
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            return None, f"Field '{field}' must be a number, got: {data[field]!r}"

        if field in ("budget", "revenue", "votes", "runtime", "year") and value < 0:
            return None, f"Field '{field}' cannot be negative."

        if field == "rating" and not (0.0 <= value <= 10.0):
            return None, f"'rating' must be between 0 and 10, got: {value}"

        if field == "year" and not (1900 <= int(value) <= 2100):
            return None, f"'year' must be between 1900 and 2100, got: {int(value)}"

        clean[field] = value

    # ── 3. Genre check ─────────────────────────
    genre = str(data["genre"]).strip()
    if genre not in VALID_GENRES:
        return None, (
            f"Invalid genre '{genre}'. "
            f"Must be one of: {VALID_GENRES}"
        )
    clean["genre"] = genre

    return clean, None


# ──────────────────────────────────────────────
#  Preprocessing
# ──────────────────────────────────────────────

def encode_genre(genre: str) -> dict:
    """
    Convert the genre string to one-hot encoded columns.

    Returns a dict of all genre_* columns set to 0 or 1.
    """
    one_hot = {f"genre_{g}": 0 for g in VALID_GENRES}
    one_hot[f"genre_{genre}"] = 1
    return one_hot


def scale_numeric(clean: dict) -> dict:
    """
    Apply StandardScaler using pre-computed training statistics.

    z = (x - mean) / std

    Only the 7 continuous numeric features are scaled;
    one-hot genre columns are left as 0/1 integers.
    """
    scaled = {}
    for field in NUMERIC_FIELDS:
        mean = SCALE_MEANS[field]
        std  = SCALE_STDS[field]
        scaled[field] = (clean[field] - mean) / std
    return scaled


def build_feature_vector(clean: dict) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with columns in the exact
    order used during model training.

    Steps:
      1. Scale numeric features
      2. One-hot encode genre
      3. Combine into FEATURE_COLUMNS order

    Returns
    -------
    pd.DataFrame  — shape (1, 16)
    """
    scaled   = scale_numeric(clean)
    one_hot  = encode_genre(clean["genre"])

    # Merge scaled numerics + one-hot features
    row = {**scaled, **one_hot}

    # Build DataFrame with controlled column order
    feature_df = pd.DataFrame([row])[FEATURE_COLUMNS]
    return feature_df


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Health-check endpoint — confirm the API is running."""
    return jsonify({
        "status":   "ok",
        "service":  "Movie Success Predictor API",
        "endpoint": "POST /predict",
        "required_fields": NUMERIC_FIELDS + ["genre"],
        "valid_genres":    VALID_GENRES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Content-Type: application/json

    Accepts a JSON body with movie features and returns the
    predicted success label: Hit / Average / Flop.

    Success response  (200):
        { "prediction": "Hit", "confidence": "model_class" }

    Error response  (400 / 422):
        { "error": "<description>" }
    """
    # ── Parse JSON body ─────────────────────────
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body is not valid JSON."}), 400

    # ── Validate ────────────────────────────────
    clean, err = validate_input(data)
    if err:
        return jsonify({"error": err}), 422

    # ── Preprocess ──────────────────────────────
    try:
        feature_df = build_feature_vector(clean)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    # ── Predict ─────────────────────────────────
    try:
        pred_int   = int(model.predict(feature_df)[0])
        pred_label = LABEL_MAP.get(pred_int, "Unknown")

        # Probability scores (not all estimators support this)
        response = {"prediction": pred_label}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_df)[0]
            response["probabilities"] = {
                LABEL_MAP[i]: round(float(p), 4)
                for i, p in enumerate(proba)
                if i in LABEL_MAP
            }

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(response), 200


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # PORT is injected automatically by Render (and other cloud platforms).
    # Falls back to 5000 for local development.
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("\n  Movie Success Predictor — Flask API")
    print("  ====================================")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  GET  http://localhost:{port}/         (health check)\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
