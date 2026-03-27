import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from scipy.sparse import hstack, csr_matrix

# =========================================================
# CONFIG
# =========================================================
ARTIFACTS_DIR = "artifacts" # This will be a subfolder in the deployment environment

app = Flask(__name__)

rf_model = None
command_vectorizer = None
response_vectorizer = None
label_encoder = None
config = None
model_ready = False
load_error = None


# =========================================================
# LOAD ARTIFACTS ONCE
# =========================================================
def load_artifacts():
    global rf_model, command_vectorizer, response_vectorizer
    global label_encoder, config, model_ready, load_error

    try:
        rf_model = joblib.load(os.path.join(ARTIFACTS_DIR, "rf_model.pkl"))
        command_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "command_vectorizer.pkl"))
        response_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "response_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

        config_path = os.path.join(ARTIFACTS_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {} # Default empty config if not found

        model_ready = True
        load_error = None
        print("All artifacts loaded successfully.")

    except Exception as e:
        model_ready = False
        load_error = str(e)
        print(f"Error loading artifacts: {e}")


load_artifacts()


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_input(command: str, response: str):
    """
    Transform input text into the same feature space used during training.
    """
    # 1. Command TF-IDF Vectorization
    cmd_features = command_vectorizer.transform([command])

    # 2. Response TF-IDF Vectorization
    resp_features = response_vectorizer.transform([response])

    # 3. Command Length & 4. Response Length (Metadata features)
    # Using csr_matrix for consistency with other sparse features
    meta_features = csr_matrix(
        np.array([[len(command), len(response)]], dtype=np.float32)
    )

    # Combine all features horizontally
    combined_features = hstack(
        [cmd_features, resp_features, meta_features],
        format="csr"
    )

    return combined_features


# =========================================================
# PREDICTION
# =========================================================
def predict_malicious_activity(command: str, response: str):
    if not model_ready:
        raise RuntimeError(f"Model not ready: {load_error}")

    features = preprocess_input(command, response)

    expected_features = rf_model.n_features_in_
    actual_features = features.shape[1]

    if actual_features != expected_features:
        raise ValueError(
            f"Feature mismatch: generated {actual_features} features, "
            f"but model expects {expected_features}. Please check preprocessing logic."
        )

    # Faster than calling predict() and predict_proba() separately
    probs = rf_model.predict_proba(features)[0]
    pred_num = int(np.argmax(probs)) # Get the index of the max probability
    confidence = float(probs[pred_num]) # Get the confidence for that class
    predicted_label = label_encoder.inverse_transform([pred_num])[0] # Decode the label

    # Assign risk level based on predicted label and confidence
    if predicted_label == "malicious":
        if confidence >= 0.8:
            risk_level = "high"
        elif confidence >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
    else:
        risk_level = "low" # Benign activities are always low risk

    return {
        "predicted_label": predicted_label,
        "confidence_score": round(confidence, 4),
        "risk_level": risk_level
    }


# =========================================================
# ROUTES
# =========================================================
@app.route("/health", methods=["GET"])
def health_check():
    if model_ready:
        return jsonify({
            "status": "ok",
            "message": "API is running",
            "model_loaded": True
        }), 200

    return jsonify({
        "status": "error",
        "message": load_error or "Model not loaded",
        "model_loaded": False
    }), 500


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    command = data.get("command")
    response = data.get("response")

    if not command or not response:
        return jsonify({"error": "'command' and 'response' are required in JSON body"}), 400

    try:
        prediction_result = predict_malicious_activity(command, response)
        return jsonify(prediction_result), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503  # Service Unavailable (model not ready)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400  # Bad Request (feature mismatch)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500 # Internal Server Error

if __name__ == '__main__':
    # This block is for local development or debugging only.
    # Gunicorn will handle running the app in production.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))