from fastapi import FastAPI
from intelligent_agent import app as agent_app
import model_inference

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # Load the ONNX model + scaler once when the server starts, not per-request.
    model_inference.load_model(
        onnx_path="LSTM_Basketball_Classifier.onnx",
        scaler_path="scaler_params_final.json",
    )


@app.post("/analyze_movement")
async def analyze_movement(data: dict):
    """Legacy/manual endpoint: pass a known error_type string directly,
    bypassing the model. Useful for testing the RAG side in isolation."""
    result = agent_app.invoke({"error_type": data["error"]})
    return {"feedback": result["feedback"]}


@app.post("/predict_and_coach")
async def predict_and_coach(data: dict):
    """
    Real pipeline endpoint. Expects:
        { "samples": [ {hand_x, hand_y, ..., angular_vz}, ... ] }
    i.e. the raw per-timestep sample list for one completed throw, in the
    same shape TrajectoryLogger already produces.

    Runs: LSTM Made/Miss prediction -> (if Miss) slip/mistake heuristic ->
    RAG coaching feedback -> returns everything to Unity in one response.
    """
    samples = data["samples"]

    outcome, prob_made = model_inference.predict_outcome(samples)

    if outcome == "Made":
        return {
            "outcome": "Made",
            "probability_made": prob_made,
            "error_type": None,
            "feedback": "Nice shot! Keep that same motion.",
        }

    error_type = model_inference.classify_error_subtype(samples)
    result = agent_app.invoke({"error_type": error_type})

    return {
        "outcome": "Miss",
        "probability_made": prob_made,
        "error_type": error_type,
        "feedback": result["feedback"],
    }


@app.get("/health")
async def health():
    # Quick endpoint to confirm the server is up before wiring Unity to it.
    return {"status": "ok"}