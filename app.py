# Honeypot ML Inference & Alert API – v2.0.0
"""
Real-time honeypot threat detection backend.

Architecture decision – Server-Sent Events (SSE) over WebSockets:
  • SSE is one-directional (server → client), which is exactly what alert
    streaming requires — the frontend never needs to send data over the
    stream channel.
  • SSE reconnects automatically via the browser EventSource API.
  • SSE works through HTTP/1.1 without protocol upgrade, making it
    compatible with every reverse proxy and cloud platform (including
    Render's free tier which kills idle WebSocket connections).
  • No extra library needed – plain Flask + threading.

Endpoints
─────────
  GET  /            API info
  GET  /health      Liveness + model status
  POST /predict     One-shot prediction (no alert emitted)
  POST /event       Ingest a honeypot event → predict → emit alert
  GET  /alerts/stream   SSE stream (text/event-stream)
  GET  /alerts/recent   Last N alerts (JSON)
  POST /alerts/test     Inject a synthetic alert for testing
"""

import os
import json
import uuid
import logging
import time
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("honeypot-api")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_ALERTS_HISTORY = int(os.environ.get("MAX_ALERTS_HISTORY", 200))

app = Flask(__name__)

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"
CORS(app, resources={r"/*": {"origins": _origins}})

# ─────────────────────────────────────────────────────────────────────────────
# ML Artifact state
# ─────────────────────────────────────────────────────────────────────────────
rf_model = None
command_vectorizer = None
response_vectorizer = None
label_encoder = None
config: dict = {}
model_ready = False
load_error: Optional[str] = None
prediction_count = 0


def _artifact(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


def load_artifacts():
    global rf_model, command_vectorizer, response_vectorizer
    global label_encoder, config, model_ready, load_error

    try:
        rf_model = joblib.load(_artifact("rf_model.pkl"))
        command_vectorizer = joblib.load(_artifact("command_vectorizer.pkl"))

        resp_path = _artifact("response_vectorizer.pkl")
        response_vectorizer = joblib.load(resp_path) if os.path.exists(resp_path) else None
        if not response_vectorizer:
            log.warning("response_vectorizer.pkl absent – using zero-padding fallback")

        le_path = _artifact("label_encoder.pkl")
        label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None
        if not label_encoder:
            log.warning("label_encoder.pkl absent – using raw class indices")

        cfg_path = _artifact("config.json")
        config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

        model_ready = True
        load_error = None
        log.info("All ML artefacts loaded successfully.")

    except Exception as exc:
        model_ready = False
        load_error = str(exc)
        log.error("Artefact load failed: %s", exc)


load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# SSE Alert Bus
# ─────────────────────────────────────────────────────────────────────────────
# Each SSE subscriber gets its own queue so alerts are fanned out to every
# connected browser tab independently.
_subscriber_lock = threading.Lock()
_subscribers: list[queue.Queue] = []

# In-memory ring buffer of recent alerts (latest first)
_alert_history: deque = deque(maxlen=MAX_ALERTS_HISTORY)
_alert_history_lock = threading.Lock()


def _register_subscriber() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=100)
    with _subscriber_lock:
        _subscribers.append(q)
    log.info("SSE subscriber connected  (total=%d)", len(_subscribers))
    return q


def _unregister_subscriber(q: queue.Queue):
    with _subscriber_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass
    log.info("SSE subscriber disconnected (total=%d)", len(_subscribers))


def _broadcast(alert: dict):
    """Push alert to all connected SSE subscribers and the history ring."""
    payload = json.dumps(alert)
    with _subscriber_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)
    with _alert_history_lock:
        _alert_history.appendleft(alert)
    log.info("Alert broadcast: %s  risk=%s  label=%s",
             alert.get("alert_id"), alert.get("risk_level"), alert.get("predicted_label"))

# ─────────────────────────────────────────────────────────────────────────────
# ML Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(command: str, response: str):
    cmd_features = command_vectorizer.transform([command])

    if response_vectorizer is not None:
        resp_features = response_vectorizer.transform([response])
    else:
        n_expected = rf_model.n_features_in_
        resp_width = max(0, n_expected - cmd_features.shape[1] - 2)
        resp_features = csr_matrix((1, resp_width), dtype=np.float32)

    meta_features = csr_matrix(
        np.array([[len(command), len(response)]], dtype=np.float32)
    )
    return hstack([cmd_features, resp_features, meta_features], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# Attack Type Classification (rule-assisted)
# ─────────────────────────────────────────────────────────────────────────────
_ATTACK_RULES = [
    # (pattern_substrings, attack_type, boost_to_critical)
    (["aws/credentials", "aws_access_key", "aws_secret", ".ssh/id_rsa",
      "id_dsa", "id_ecdsa", ".netrc", "shadow", "passwd"],
     "credential_access", True),
    (["wget", "curl", "python -c", "bash -i", "/dev/tcp", "nc -e",
      "ncat", "mkfifo", "socat"],
     "malware_download", True),
    (["chmod +s", "sudo", "su root", "setuid", "/etc/sudoers",
      "visudo", "pkexec"],
     "privilege_escalation", True),
    (["crontab", "/etc/rc", "systemctl enable", "init.d", ".bashrc",
      ".profile", "~/.bash_profile", "authorized_keys"],
     "persistence", False),
    (["nmap", "masscan", "ping", "traceroute", "arp -a", "netstat",
      "ss -", "ifconfig", "ip addr", "cat /etc/hosts",
      "uname -a", "uname -r", "whoami", "id ", "cat /etc/os-release"],
     "reconnaissance", False),
    (["ssh ", "scp ", "rsync ", "psexec", "wmic", "net use",
      "mount ", "rdesktop"],
     "lateral_movement", True),
    (["python", "perl", "ruby", "bash ", "sh -c", "exec(",
      "system(", "eval(", "base64"],
     "execution", False),
]

def _classify_attack_type(command: str, response: str) -> str:
    combined = (command + " " + response).lower()
    for patterns, attack_type, _ in _ATTACK_RULES:
        if any(p in combined for p in patterns):
            return attack_type
    return "other"


def _should_escalate_to_critical(command: str, response: str) -> bool:
    combined = (command + " " + response).lower()
    for patterns, _, escalate in _ATTACK_RULES:
        if escalate and any(p in combined for p in patterns):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Risk level
# ─────────────────────────────────────────────────────────────────────────────
_MALICIOUS_LABELS = {"malicious", "attack", "1", "true", "malware",
                     "reconnaissance", "exploitation", "lateral_movement"}

def _risk_level(label: str, confidence: float,
                command: str = "", response: str = "") -> str:
    if label.lower() in _MALICIOUS_LABELS:
        if _should_escalate_to_critical(command, response):
            return "critical"
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.5:
            return "medium"
        return "low"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(command: str, response: str) -> dict:
    global prediction_count
    if not model_ready:
        raise RuntimeError(f"Model not ready: {load_error}")

    features = preprocess_input(command, response)
    if features.shape[1] != rf_model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: got {features.shape[1]}, "
            f"expected {rf_model.n_features_in_}"
        )

    probs = rf_model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    if label_encoder is not None:
        predicted_label = str(label_encoder.inverse_transform([pred_idx])[0])
    else:
        predicted_label = str(rf_model.classes_[pred_idx])

    prediction_count += 1
    return {
        "predicted_label": predicted_label,
        "confidence_score": round(confidence, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alert builder
# ─────────────────────────────────────────────────────────────────────────────
_alert_counter = 0
_alert_counter_lock = threading.Lock()

def _next_alert_id() -> str:
    global _alert_counter
    with _alert_counter_lock:
        _alert_counter += 1
        return f"alert-{_alert_counter:06d}"


def build_alert(event: dict, prediction: dict) -> dict:
    command  = event.get("command", "")
    response = event.get("response", "")
    label    = prediction["predicted_label"]
    conf     = prediction["confidence_score"]
    risk     = _risk_level(label, conf, command, response)
    atype    = _classify_attack_type(command, response)

    return {
        "alert_id":         _next_alert_id(),
        "timestamp":        event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "predicted_label":  label,
        "confidence_score": conf,
        "risk_level":       risk,
        "attack_type":      atype,
        # location
        "session_id":       event.get("session_id", ""),
        "source_ip":        event.get("source_ip", "unknown"),
        "host":             event.get("host", "unknown"),
        "container_name":   event.get("container_name", ""),
        "application_name": event.get("application_name", ""),
        "endpoint":         event.get("endpoint", ""),
        "process":          event.get("process", ""),
        # raw payload
        "command":          command,
        "response_snippet": response[:300] if response else "",
        # meta
        "acknowledged":     False,
        "notes":            "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Honeypot ML Inference & Alert API",
        "version": "2.0.0",
        "status": "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "endpoints": {
            "health":         "GET  /health",
            "predict":        "POST /predict",
            "event":          "POST /event",
            "stream":         "GET  /alerts/stream",
            "recent_alerts":  "GET  /alerts/recent",
            "test_alert":     "POST /alerts/test",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health():
    payload = {
        "status":           "ok" if model_ready else "error",
        "message":          "API is running" if model_ready else (load_error or "Model not loaded"),
        "model_loaded":     model_ready,
        "prediction_count": prediction_count,
        "alert_count":      len(_alert_history),
        "sse_subscribers":  len(_subscribers),
        "artifacts": {
            "rf_model":            rf_model is not None,
            "command_vectorizer":  command_vectorizer is not None,
            "response_vectorizer": response_vectorizer is not None,
            "label_encoder":       label_encoder is not None,
        },
    }
    return jsonify(payload), 200 if model_ready else 500


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """One-shot prediction only – does NOT emit a real-time alert."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json(silent=True) or {}

    command    = str(data.get("command", "")).strip()
    response   = str(data.get("response", "")).strip()
    session_id = data.get("session_id", "")

    if not command:
        return jsonify({"error": "'command' is required"}), 400

    try:
        pred = run_prediction(command, response)
        pred["session_id"] = session_id
        return jsonify(pred), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("Predict error")
        return jsonify({"error": str(exc)}), 500


@app.route("/event", methods=["POST"])
def ingest_event():
    """
    Ingest a full honeypot event, run ML prediction, build a structured
    alert, persist it to history, and broadcast to all SSE subscribers.

    Expected JSON body (all fields except 'command' are optional):
    {
      "command":          "cat ~/.aws/credentials",
      "response":         "[default]\\naws_access_key_id=AKIA...",
      "session_id":       "sess-001",
      "source_ip":        "192.168.1.5",
      "host":             "honeypot-node-1",
      "container_name":   "web-app-container",
      "application_name": "node-service",
      "endpoint":         "/api/upload",
      "process":          "python3",
      "timestamp":        "2026-03-30T20:10:00Z"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    event = request.get_json(silent=True) or {}

    command  = str(event.get("command", "")).strip()
    response = str(event.get("response", "")).strip()

    if not command:
        return jsonify({"error": "'command' is required"}), 400

    try:
        prediction = run_prediction(command, response)
        alert      = build_alert(event, prediction)
        _broadcast(alert)
        return jsonify({
            "status":     "ok",
            "alert_id":   alert["alert_id"],
            "alert":      alert,
        }), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("Event ingest error")
        return jsonify({"error": str(exc)}), 500


@app.route("/alerts/stream", methods=["GET"])
def alert_stream():
    """
    Server-Sent Events stream.

    Clients connect with:
      const es = new EventSource('/alerts/stream');
      es.onmessage = (e) => { const alert = JSON.parse(e.data); ... };

    The stream emits:
      • 'connected' event immediately on connection
      • 'alert'     event for every new alert
      • comment heartbeat every 25 s to keep the connection alive through
        proxies that kill idle connections
    """
    subscriber_q = _register_subscriber()

    def generate():
        # Immediately confirm connection
        yield "event: connected\ndata: {\"status\":\"connected\"}\n\n"

        last_heartbeat = time.time()
        try:
            while True:
                # Send heartbeat comment every 25 s
                now = time.time()
                if now - last_heartbeat >= 25:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now

                try:
                    payload = subscriber_q.get(timeout=1.0)
                    yield f"event: alert\ndata: {payload}\n\n"
                except queue.Empty:
                    pass

        except GeneratorExit:
            pass
        finally:
            _unregister_subscriber(subscriber_q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # Disable nginx buffering
            "Connection":       "keep-alive",
        },
    )


@app.route("/alerts/recent", methods=["GET"])
def recent_alerts():
    """Return the last N alerts. Default N = 50."""
    limit = min(int(request.args.get("limit", 50)), MAX_ALERTS_HISTORY)
    with _alert_history_lock:
        alerts = list(_alert_history)[:limit]
    return jsonify({"alerts": alerts, "total": len(_alert_history)}), 200


@app.route("/alerts/test", methods=["POST"])
def test_alert():
    """
    Inject a pre-built synthetic alert for UI testing.
    Accepts a partial alert dict; missing fields get safe defaults.
    """
    data = request.get_json(silent=True) or {}
    risk = data.get("risk_level", "high")
    synthetic = {
        "alert_id":         _next_alert_id(),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "predicted_label":  data.get("predicted_label", "malicious"),
        "confidence_score": data.get("confidence_score", 0.95),
        "risk_level":       risk,
        "attack_type":      data.get("attack_type", "credential_access"),
        "session_id":       data.get("session_id", "test-session-001"),
        "source_ip":        data.get("source_ip", "10.0.0.42"),
        "host":             data.get("host", "honeypot-test-node"),
        "container_name":   data.get("container_name", "test-container"),
        "application_name": data.get("application_name", "test-service"),
        "endpoint":         data.get("endpoint", "/test"),
        "process":          data.get("process", "bash"),
        "command":          data.get("command", "cat ~/.aws/credentials"),
        "response_snippet": data.get("response_snippet",
                                     "[default]\naws_access_key_id=AKIATEST123\n"),
        "acknowledged":     False,
        "notes":            "Synthetic test alert",
    }
    _broadcast(synthetic)
    return jsonify({"status": "ok", "alert": synthetic}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # threaded=True is required for SSE to work in dev mode
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
