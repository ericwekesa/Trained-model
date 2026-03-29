from __future__ import annotations

# Honeypot ML Inference API  – v3.0.0
# Changes from v2:
#  - /predict now stores every result in an in-memory ring buffer
#  - /predict now returns correct risk_level (fixes raw 0/1 label bug)
#  - Added GET  /results          – all stored results, newest first
#  - Added GET  /results/latest   – single most-recent result
#  - Added GET  /results/summary  – counts, risk breakdown, latest attack
#  - Added DELETE /results        – clear all results (demo reset)
#  - SSE stream event renamed to "result" (was "alert") for consistency

import os
import json
import logging
import time
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("honeypot-api")

# ── App & CORS ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", 500))

app = Flask(__name__)
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"
CORS(app, resources={r"/*": {"origins": _origins}})

# ── ML state ──────────────────────────────────────────────────────────────────
rf_model            = None
command_vectorizer  = None
response_vectorizer = None
label_encoder       = None
config: dict        = {}
model_ready         = False
load_error: Optional[str] = None
prediction_count    = 0


def _artifact(name: str) -> str:
    return os.path.join(BASE_DIR, name)


def load_artifacts():
    global rf_model, command_vectorizer, response_vectorizer
    global label_encoder, config, model_ready, load_error
    try:
        rf_model           = joblib.load(_artifact("rf_model.pkl"))
        command_vectorizer = joblib.load(_artifact("command_vectorizer.pkl"))

        rp = _artifact("response_vectorizer.pkl")
        response_vectorizer = joblib.load(rp) if os.path.exists(rp) else None

        lp = _artifact("label_encoder.pkl")
        label_encoder = joblib.load(lp) if os.path.exists(lp) else None

        cp = _artifact("config.json")
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            config = json.loads(raw) if raw else {}
        else:
            config = {}

        model_ready = True
        load_error  = None
        log.info("Artefacts loaded. label_encoder=%s", label_encoder is not None)
    except Exception as exc:
        model_ready = False
        load_error  = str(exc)
        log.error("Artefact load failed: %s", exc)


load_artifacts()

# ── SSE bus ───────────────────────────────────────────────────────────────────
_sub_lock: threading.Lock       = threading.Lock()
_subscribers: List[queue.Queue] = []
_sse_history: deque             = deque(maxlen=200)
_sse_history_lock               = threading.Lock()


def _register_sub() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=100)
    with _sub_lock:
        _subscribers.append(q)
    log.info("SSE subscriber connected (total=%d)", len(_subscribers))
    return q


def _unregister_sub(q: queue.Queue):
    with _sub_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass
    log.info("SSE subscriber disconnected (total=%d)", len(_subscribers))


def _broadcast(payload: dict):
    data = json.dumps(payload)
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for d in dead:
            _subscribers.remove(d)
    with _sse_history_lock:
        _sse_history.appendleft(payload)


# ── Result store ──────────────────────────────────────────────────────────────
_results: deque    = deque(maxlen=MAX_RESULTS)
_results_lock      = threading.Lock()
_result_counter    = 0
_result_counter_lk = threading.Lock()


def _next_id() -> int:
    global _result_counter
    with _result_counter_lk:
        _result_counter += 1
        return _result_counter


def _store(record: dict):
    with _results_lock:
        _results.appendleft(record)


def _all_results() -> list:
    with _results_lock:
        return list(_results)


def _clear():
    global _result_counter
    with _results_lock:
        _results.clear()
    with _result_counter_lk:
        _result_counter = 0
    with _sse_history_lock:
        _sse_history.clear()
    log.info("All results cleared.")


# ── ML helpers ────────────────────────────────────────────────────────────────
def _preprocess(command: str, response: str):
    cmd_f = command_vectorizer.transform([command])
    if response_vectorizer is not None:
        resp_f = response_vectorizer.transform([response])
    else:
        w      = max(0, rf_model.n_features_in_ - cmd_f.shape[1] - 2)
        resp_f = csr_matrix((1, w), dtype=np.float32)
    meta_f = csr_matrix(np.array([[len(command), len(response)]], dtype=np.float32))
    return hstack([cmd_f, resp_f, meta_f], format="csr")


# When label_encoder is absent the model returns raw class integers.
# Trained with 0 = benign, 1 = malicious.
_INT_LABEL = {"0": "benign", "1": "malicious"}

_MALICIOUS_SET = {
    "malicious", "attack", "1", "malware",
    "reconnaissance", "exploitation", "lateral_movement",
}

_ATTACK_RULES: list = [
    (["aws/credentials", "aws_access_key", "aws_secret", ".ssh/id_rsa",
      "shadow", "passwd", ".netrc"],                       "credential_access",    True),
    (["wget http", "curl http", "python -c", "bash -i",
      "/dev/tcp", "nc -e", "mkfifo", "socat"],             "malware_download",     True),
    (["chmod +x", "./payload", "./exploit", "./shell"],     "execution",            True),
    (["chmod +s", "sudo", "su root", "setuid",
      "/etc/sudoers", "pkexec"],                           "privilege_escalation", True),
    (["crontab", "systemctl enable", "authorized_keys",
      ".bashrc", ".profile"],                              "persistence",          False),
    (["whoami", "uname", "ls -l", "id ", "cat /etc",
      "ifconfig", "ip addr", "netstat", "ps aux", "ls -la"], "reconnaissance",     False),
    (["ssh ", "scp ", "rsync ", "net use"],                 "lateral_movement",     True),
    (["python", "perl", "ruby", "bash ", "sh -c",
      "exec(", "eval(", "base64"],                         "execution",            False),
]


def _attack_type(cmd: str, resp: str) -> str:
    combined = (cmd + " " + resp).lower()
    for patterns, atype, _ in _ATTACK_RULES:
        if any(p in combined for p in patterns):
            return atype
    return "other"


def _is_critical(cmd: str, resp: str) -> bool:
    combined = (cmd + " " + resp).lower()
    return any(
        escalate and any(p in combined for p in patterns)
        for patterns, _, escalate in _ATTACK_RULES
    )


def _risk(label: str, conf: float, cmd: str = "", resp: str = "") -> str:
    if label.lower() in _MALICIOUS_SET:
        if _is_critical(cmd, resp):
            return "critical"
        if conf >= 0.85:
            return "high"
        if conf >= 0.50:
            return "medium"
        return "low"
    return "low"


def _run_prediction(command: str, response: str) -> dict:
    global prediction_count
    if not model_ready:
        raise RuntimeError(load_error or "Model not loaded")

    features = _preprocess(command, response)
    if features.shape[1] != rf_model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: got {features.shape[1]}, "
            f"expected {rf_model.n_features_in_}"
        )

    probs = rf_model.predict_proba(features)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])

    if label_encoder is not None:
        raw = str(label_encoder.inverse_transform([idx])[0])
    else:
        raw = str(rf_model.classes_[idx])

    label = _INT_LABEL.get(raw, raw)
    prediction_count += 1
    return {"predicted_label": label, "raw_label": raw, "confidence_score": round(conf, 4)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":         "Honeypot ML Inference API",
        "version":      "3.0.0",
        "status":       "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "endpoints": {
            "health":          "GET    /health",
            "predict":         "POST   /predict",
            "results":         "GET    /results",
            "results_latest":  "GET    /results/latest",
            "results_summary": "GET    /results/summary",
            "results_clear":   "DELETE /results",
            "sse_stream":      "GET    /alerts/stream",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok" if model_ready else "error",
        "message":          "API is running" if model_ready else (load_error or "Model not loaded"),
        "model_loaded":     model_ready,
        "prediction_count": prediction_count,
        "result_count":     len(_results),
        "sse_subscribers":  len(_subscribers),
        "artifacts": {
            "rf_model":            rf_model is not None,
            "command_vectorizer":  command_vectorizer is not None,
            "response_vectorizer": response_vectorizer is not None,
            "label_encoder":       label_encoder is not None,
        },
    }), 200 if model_ready else 500


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict  – used by Colab simulation script
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data       = request.get_json(silent=True) or {}
    command    = str(data.get("command",  "")).strip()
    response   = str(data.get("response", "")).strip()
    session_id = str(data.get("session_id", "")).strip()

    if not command:
        return jsonify({"error": "'command' is required"}), 400

    try:
        pred     = _run_prediction(command, response)
        label    = pred["predicted_label"]
        conf     = pred["confidence_score"]
        risk     = _risk(label, conf, command, response)
        atype    = _attack_type(command, response)
        detected = label.lower() in _MALICIOUS_SET

        record = {
            "id":               _next_id(),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "command":          command,
            "response":         response[:300],
            "predicted_label":  label,
            "raw_label":        pred["raw_label"],
            "confidence_score": conf,
            "risk_level":       risk,
            "attack_type":      atype,
            "attack_detected":  detected,
            "session_id":       session_id,
        }

        _store(record)
        _broadcast(record)

        return jsonify({
            "predicted_label":  label,
            "confidence_score": conf,
            "risk_level":       risk,
            "attack_type":      atype,
            "attack_detected":  detected,
            "session_id":       session_id,
            "id":               record["id"],
        }), 200

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("Predict error")
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /results
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/results", methods=["GET"])
def get_results():
    limit = min(int(request.args.get("limit", 100)), MAX_RESULTS)
    rows  = _all_results()[:limit]
    return jsonify({"results": rows, "total": len(_results)}), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /results/latest
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/results/latest", methods=["GET"])
def get_latest():
    rows = _all_results()
    return jsonify({"result": rows[0] if rows else None}), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /results/summary
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/results/summary", methods=["GET"])
def get_summary():
    rows     = _all_results()
    attacks  = [r for r in rows if r.get("attack_detected")]
    risk_counts: dict = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    type_counts: dict = {}
    for r in rows:
        lvl = r.get("risk_level", "low")
        if lvl in risk_counts:
            risk_counts[lvl] += 1
        t = r.get("attack_type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    return jsonify({
        "total_events":   len(rows),
        "total_attacks":  len(attacks),
        "benign_events":  len(rows) - len(attacks),
        "risk_breakdown": risk_counts,
        "attack_types":   type_counts,
        "latest_attack":  attacks[0] if attacks else None,
        "latest_event":   rows[0]    if rows    else None,
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /results  – clear all stored results (demo/presentation reset)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/results", methods=["DELETE"])
def delete_results():
    _clear()
    return jsonify({"status": "ok", "message": "All results cleared"}), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/stream  – SSE (kept for backwards compat with SOC dashboard)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/alerts/stream", methods=["GET"])
def alert_stream():
    sub = _register_sub()

    def generate():
        yield "event: connected\ndata: {\"status\":\"connected\"}\n\n"
        last_hb = time.time()
        try:
            while True:
                now = time.time()
                if now - last_hb >= 25:
                    yield ": heartbeat\n\n"
                    last_hb = now
                try:
                    payload = sub.get(timeout=1.0)
                    # emit as both "result" and "alert" for compatibility
                    yield f"event: result\ndata: {payload}\n\n"
                    yield f"event: alert\ndata: {payload}\n\n"
                except queue.Empty:
                    pass
        except GeneratorExit:
            pass
        finally:
            _unregister_sub(sub)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/recent  – backwards compat with SOC dashboard
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/alerts/recent", methods=["GET"])
def alerts_recent():
    limit = min(int(request.args.get("limit", 50)), 200)
    with _sse_history_lock:
        items = list(_sse_history)[:limit]
    return jsonify({"alerts": items, "total": len(_sse_history)}), 200


# ── Dev entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
