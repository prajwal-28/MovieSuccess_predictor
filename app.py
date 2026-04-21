"""
Movie Success Predictor - Flask Backend API
===========================================
Serves a trained Random Forest ML model via a REST endpoint.
Note: Predicts without data leakage (no budget or revenue required).

Endpoint:
    POST /predict
    Content-Type: application/json

Example request body:
    {
        "rating":  8.8,
        "votes":   2200000,
        "runtime": 148,
        "year":    2010,
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

# Exact numeric fields trained on
NUMERIC_FIELDS = ["rating", "votes", "runtime", "year"]

# Exact valid genres available in the TMDB large dataset
VALID_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", 
    "Documentary", "Drama", "Family", "Fantasy", "Foreign", 
    "History", "Horror", "Music", "Mystery", "Romance", 
    "Science Fiction", "Thriller", "War", "Western"
]

# Exact feature column order seen by the model during training
FEATURE_COLUMNS = [
    "runtime", "rating", "votes", "year", 
    "genre_Action", "genre_Adventure", "genre_Animation", 
    "genre_Comedy", "genre_Crime", "genre_Documentary", 
    "genre_Drama", "genre_Family", "genre_Fantasy", 
    "genre_Foreign", "genre_History", "genre_Horror", 
    "genre_Music", "genre_Mystery", "genre_Romance", 
    "genre_Science Fiction", "genre_Thriller", "genre_War", 
    "genre_Western"
]

# Label map: integer output → class name
LABEL_MAP = {0: "Average", 1: "Flop", 2: "Hit"}

# Scaling parameters derived from new large training data
# These reproduce the StandardScaler
SCALE_MEANS = {
    'rating': 6.22938, 
    'votes': 646.32763, 
    'runtime': 109.60243, 
    'year': 2002.04256
}
SCALE_STDS = {
    'rating': 0.86751, 
    'votes': 816.92100, 
    'runtime': 19.89296, 
    'year': 12.29971
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
        except (ValueError, TypeError):
            return None, f"Field '{field}' must be a number, got: {data[field]!r}"

        if field in ("votes", "runtime") and value < 0:
            return None, f"Field '{field}' cannot be negative."

        if field == "rating" and not (0.0 <= value <= 10.0):
            return None, f"'rating' must be between 0 and 10, got: {value}"

        if field == "year" and not (1900 <= int(value) <= 2100):
            return None, f"'year' must be between 1900 and 2100, got: {int(value)}"

        clean[field] = value

    # ── 3. Genre check ─────────────────────────
    genre = str(data["genre"]).strip()
    if genre not in VALID_GENRES:
        # Fallback to similar available genres or strict block.
        # Strict block for integrity
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
    """
    one_hot = {f"genre_{g}": 0 for g in VALID_GENRES}
    one_hot[f"genre_{genre}"] = 1
    return one_hot


def scale_numeric(clean: dict) -> dict:
    """
    Apply StandardScaler using pre-computed training statistics.
    z = (x - mean) / std
    """
    scaled = {}
    for field in NUMERIC_FIELDS:
        mean = SCALE_MEANS[field]
        std  = SCALE_STDS[field]
        scaled[field] = (clean[field] - mean) / std
    return scaled


def build_feature_vector(clean: dict) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame.
    """
    scaled   = scale_numeric(clean)
    one_hot  = encode_genre(clean["genre"])

    row = {**scaled, **one_hot}

    # Build DataFrame with exact feature columns
    feature_df = pd.DataFrame([row])[FEATURE_COLUMNS]
    
    # Fill any missing encoded genres explicitly with zero
    for col in FEATURE_COLUMNS:
        if col not in feature_df:
            feature_df[col] = 0.0
            
    return feature_df[FEATURE_COLUMNS]


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
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body is not valid JSON."}), 400

    clean, err = validate_input(data)
    if err:
        return jsonify({"error": err}), 422

    try:
        feature_df = build_feature_vector(clean)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    try:
        pred_int   = int(model.predict(feature_df)[0])
        pred_label = LABEL_MAP.get(pred_int, "Unknown")
        
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
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("\n  Movie Success Predictor — Flask API (No Leakage Edition)")
    print("  ========================================================")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  GET  http://localhost:{port}/         (health check)\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
