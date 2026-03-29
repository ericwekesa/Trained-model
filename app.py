import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

# =========================================================
# CONFIG
# =========================================================
# Artifacts live in the same directory as this file (project root).
# Resolving relative to __file__ ensures the app works regardless
# of the working directory it is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow all origins in development; tighten in production by setting the
# ALLOWED_ORIGINS environment variable to a comma-separated list of origins.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"
CORS(app, resources={r"/*": {"origins": _origins}})

# ── Module-level state ────────────────────────────────────────────────────────
rf_model = None
command_vectorizer = None
response_vectorizer = None
label_encoder = None
config = None
model_ready = False
load_error = None

# Ephemeral counter – resets on each process restart
prediction_count = 0


# =========================================================
# ARTIFACT HELPERS
# =========================================================
def _artifact(filename: str) -> str:
    """Return the absolute path to an artifact file."""
    return os.path.join(BASE_DIR, filename)


def load_artifacts():
    """Load all ML artefacts from the project root at startup."""
    global rf_model, command_vectorizer, response_vectorizer
    global label_encoder, config, model_ready, load_error

    try:
        rf_model = joblib.load(_artifact("rf_model.pkl"))
        command_vectorizer = joblib.load(_artifact("command_vectorizer.pkl"))

        # response_vectorizer – optional; training artefact may not exist yet.
        resp_path = _artifact("response_vectorizer.pkl")
        if os.path.exists(resp_path):
            response_vectorizer = joblib.load(resp_path)
        else:
            response_vectorizer = None
            print("WARNING: response_vectorizer.pkl not found – "
                  "response text will be zero-padded to match expected feature width.")

        # label_encoder – optional.
        le_path = _artifact("label_encoder.pkl")
        if os.path.exists(le_path):
            label_encoder = joblib.load(le_path)
        else:
            label_encoder = None
            print("WARNING: label_encoder.pkl not found – "
                  "raw integer class indices will be returned as labels.")

        config_path = _artifact("config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        model_ready = True
        load_error = None
        print("All artefacts loaded successfully.")

    except Exception as e:
        model_ready = False
        load_error = str(e)
        print(f"Error loading artefacts: {e}")


load_artifacts()


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_input(command: str, response: str):
    """
    Transform input text into the exact same feature space used during training.

    Feature pipeline (MUST match training order):
      1. command  TF-IDF vectorisation
      2. response TF-IDF vectorisation  (zero-vector fallback when absent)
      3. command length  (scalar, float32)
      4. response length (scalar, float32)

    Uses scipy sparse hstack – never calls .toarray() during inference.
    """
    # 1. Command TF-IDF
    cmd_features = command_vectorizer.transform([command])

    # 2. Response TF-IDF (fall back to zeros when vectoriser is absent)
    if response_vectorizer is not None:
        resp_features = response_vectorizer.transform([response])
    else:
        # Width = total expected – cmd width – 2 metadata columns
        n_expected = rf_model.n_features_in_
        resp_width = max(0, n_expected - cmd_features.shape[1] - 2)
        resp_features = csr_matrix((1, resp_width), dtype=np.float32)

    # 3 & 4. Metadata scalars
    meta_features = csr_matrix(
        np.array([[len(command), len(response)]], dtype=np.float32)
    )

    combined = hstack([cmd_features, resp_features, meta_features], format="csr")
    return combined


# =========================================================
# RISK LEVEL ASSIGNMENT
# =========================================================
_MALICIOUS_LABELS = {"malicious", "attack", "1", "true", "malware",
                     "reconnaissance", "exploitation", "lateral_movement"}


def _risk_level(label: str, confidence: float) -> str:
    if label.lower() in _MALICIOUS_LABELS:
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        return "low"
    return "low"


# =========================================================
# PREDICTION
# =========================================================
def predict_activity(command: str, response: str) -> dict:
    global prediction_count

    if not model_ready:
        raise RuntimeError(f"Model not ready: {load_error}")

    features = preprocess_input(command, response)

    expected = rf_model.n_features_in_
    actual = features.shape[1]
    if actual != expected:
        raise ValueError(
            f"Feature mismatch: generated {actual} features, "
            f"model expects {expected}. Verify preprocessing logic."
        )

    # Single predict_proba call – faster than predict() + predict_proba()
    probs = rf_model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    if label_encoder is not None:
        predicted_label = str(label_encoder.inverse_transform([pred_idx])[0])
    else:
        # Map index to class directly from the estimator
        predicted_label = str(rf_model.classes_[pred_idx])

    prediction_count += 1

    return {
        "predicted_label": predicted_label,
        "confidence_score": round(confidence, 4),
        "risk_level": _risk_level(predicted_label, confidence),
    }


# =========================================================
# ROUTES
# =========================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Return API and model readiness status."""
    payload = {
        "status": "ok" if model_ready else "error",
        "message": "API is running" if model_ready else (load_error or "Model not loaded"),
        "model_loaded": model_ready,
        "prediction_count": prediction_count,
        "artifacts": {
            "rf_model": rf_model is not None,
            "command_vectorizer": command_vectorizer is not None,
            "response_vectorizer": response_vectorizer is not None,
            "label_encoder": label_encoder is not None,
        },
    }
    return jsonify(payload), 200 if model_ready else 500


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict honeypot command behaviour.

    Expected JSON body:
      {
        "command":    "<shell command>",     (required)
        "response":   "<command output>",    (optional, defaults to "")
        "session_id": "<session uuid>"       (optional, echoed back)
      }

    Returns:
      {
        "predicted_label": "malicious" | "benign" | ...,
        "confidence_score": 0.94,
        "risk_level": "high" | "medium" | "low",
        "session_id": "<echoed>"
      }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Malformed JSON body"}), 400

    command = str(data.get("command", "")).strip()
    response = str(data.get("response", "")).strip()
    session_id = data.get("session_id", "")

    if not command:
        return jsonify({"error": "'command' is required and must not be empty"}), 400

    try:
        result = predict_activity(command, response)
        result["session_id"] = session_id
        return jsonify(result), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503      # Service Unavailable
    except ValueError as e:
        return jsonify({"error": str(e)}), 400      # Bad Request
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# =========================================================
# ENTRY POINT  (dev only – production uses gunicorn via Procfile)
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
