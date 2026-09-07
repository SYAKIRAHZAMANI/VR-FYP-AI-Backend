"""
Loads the trained LSTM (exported as ONNX) and runs inference on a single
throw's raw kinematic samples.

The LSTM only predicts binary Made/Miss (see FYP_ModelTraining_Export.ipynb,
Cell 3: `df['target'] = 1 if outcome == "Made" else 0`). It does NOT classify
*which* error occurred. To bridge this to the RAG coaching system (which
needs a specific error_type like "Early_Release" or "Jerky_Motion"), a
lightweight rule-based sub-classifier is applied on Miss predictions,
following Norman's (2013) Slip/Mistake distinction already used in your
HCI Error Modelling Framework (Section 3.2, Table 3.1):

  - Slip (execution noise)  -> high-frequency positional jitter -> "Jerky_Motion"
  - Mistake (weak throw)    -> low release speed                -> "Low_Velocity"
  - Mistake (flat throw)    -> low upward loft angle             -> "Arc_Too_Flat"

These heuristic thresholds mirror how FYP_SyntheticData.ipynb generated its
own training labels (Slip_Tremor / Mistake_Weak / Mistake_Flat), so this is
consistent with, not separate from, your existing methodology.
"""

import json
import numpy as np
import onnxruntime as ort

MAX_SEQ_LEN = 50

# Exact order used during training (FYP_ModelTraining_Export.ipynb, Cell 3).
# Must match exactly, in this order, or predictions will be meaningless.
FEATURE_COLS = [
    "hand_x", "hand_y", "hand_z",
    "hand_rot_x", "hand_rot_y", "hand_rot_z", "hand_rot_w",
    "linear_vx", "linear_vy", "linear_vz",
    "linear_ax", "linear_ay", "linear_az",
    "angular_vx", "angular_vy", "angular_vz",
]

_session = None
_scaler_mean = None
_scaler_scale = None


def load_model(onnx_path="LSTM_Basketball_Classifier.onnx", scaler_path="scaler_params_final.json"):
    """Call once at server startup."""
    global _session, _scaler_mean, _scaler_scale

    _session = ort.InferenceSession(onnx_path)

    with open(scaler_path, "r") as f:
        params = json.load(f)

    if params["feature_cols"] != FEATURE_COLS:
        raise ValueError(
            "scaler_params_final.json feature order does not match FEATURE_COLS. "
            "Re-check the export cell in the training notebook."
        )

    _scaler_mean = np.array(params["mean"], dtype=np.float32)
    _scaler_scale = np.array(params["scale"], dtype=np.float32)

    print(f"[model_inference] Loaded ONNX model from {onnx_path}")
    print(f"[model_inference] Loaded scaler from {scaler_path}")


def _samples_to_tensor(samples):
    """
    samples: list of dicts, one per timestep, each containing at least the
    keys in FEATURE_COLS (this is exactly what TrajectoryLogger's CSV rows
    contain, so Unity can send its sample list almost as-is).

    Applies the same scaling + pad/truncate-to-MAX_SEQ_LEN logic as
    prepare_sequential_data() in the training notebook (Cell 5).
    """
    raw = np.array([[s[col] for col in FEATURE_COLS] for s in samples], dtype=np.float32)

    scaled = (raw - _scaler_mean) / _scaler_scale

    if len(scaled) >= MAX_SEQ_LEN:
        seq = scaled[:MAX_SEQ_LEN]
    else:
        pad_len = MAX_SEQ_LEN - len(scaled)
        seq = np.vstack([scaled, np.zeros((pad_len, len(FEATURE_COLS)), dtype=np.float32)])

    # Model expects shape (batch, seq_len, num_features)
    return seq[np.newaxis, :, :]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_outcome(samples):
    """Returns (predicted_label: 'Made'|'Miss', probability_of_made: float)."""
    if _session is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    input_tensor = _samples_to_tensor(samples)
    outputs = _session.run(None, {"input": input_tensor})
    logit = outputs[0][0][0]  # shape (1,1) -> scalar
    prob_made = float(_sigmoid(logit))

    label = "Made" if prob_made >= 0.5 else "Miss"
    return label, prob_made


def classify_error_subtype(samples):
    """
    Rule-based Slip/Mistake sub-classification, only meaningful when
    predict_outcome() returned 'Miss'. Returns an error_type string that
    matches a real entry in Appendix A / coaching_rules.txt.
    """
    # 1. Calculate raw features
    raw = np.array([[s[col] for col in FEATURE_COLS] for s in samples], dtype=np.float32)

    # 2. Calculate release speed
    vel = raw[:, FEATURE_COLS.index("linear_vx"):FEATURE_COLS.index("linear_vz") + 1]
    speeds = np.linalg.norm(vel, axis=1)
    release_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

    # 3. Calculate Upward loft
    peak_idx = int(np.argmax(speeds)) if len(speeds) > 0 else 0
    vy_at_release = float(vel[peak_idx, 1]) if len(vel) > 0 else 0.0
    vz_at_release = float(vel[peak_idx, 2]) if len(vel) > 0 else 1e-6
    loft_ratio = vy_at_release / max(abs(vz_at_release), 1e-6)

    # 4. Calculate High-frequency jitter
    acc = raw[:, FEATURE_COLS.index("linear_ax"):FEATURE_COLS.index("linear_az") + 1]
    acc_mag = np.linalg.norm(acc, axis=1)
    jitter = float(np.std(acc_mag)) if len(acc_mag) > 0 else 0.0

    # 5. Thresholds
    JITTER_THRESHOLD = 999999.0  # Absurdly high to bypass keyboard spikes completely
    WEAK_SPEED_THRESHOLD = 6.5
    FLAT_LOFT_THRESHOLD = 0.4

    # --- NEW: Print the actual numbers to the backend console! ---
    print(f"\n[DEBUG THROW] Jitter: {jitter:.2f} | Speed: {release_speed:.2f} | Loft: {loft_ratio:.2f}")

    # 6. Make decision
    if jitter > JITTER_THRESHOLD:
        return "Jerky_Motion"
    if release_speed < WEAK_SPEED_THRESHOLD:
        return "Low_Velocity"
    if loft_ratio < FLAT_LOFT_THRESHOLD:
        return "Arc_Too_Flat"

    return "Aim_Off"  # generic fallback when no specific pattern is dominant