import asyncio
from fastapi import FastAPI
from intelligent_agent import app as agent_app
import model_inference

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # 1. Load the ONNX model + scaler
    model_inference.load_model(
        onnx_path="LSTM_Basketball_Classifier.onnx",
        scaler_path="scaler_params_final.json",
    )
    
    # 2. --- NEW FIX: PRE-WARM THE RAG CACHE ---
    # We generate the tips for all 4 possible errors right now so Unity never has to wait.
    print("\n--- PRE-WARMING AI COACH CACHE (This might take 1-2 minutes on CPU) ---")
    known_errors = ["Jerky_Motion", "Low_Velocity", "Arc_Too_Flat", "Aim_Off"]
    
    for error in known_errors:
        print(f"Generating and caching tip for: {error}...")
        # We run this directly to force the Grader -> Generator pipeline to run once
        agent_app.invoke({"error_type": error})
        
    print("--- CACHE WARMED UP! SERVER IS 100% READY FOR UNITY ---\n")




@app.post("/analyze_movement")
async def analyze_movement(data: dict):
    """Legacy/manual endpoint: pass a known error_type string directly,
    bypassing the model. Useful for testing the RAG side in isolation."""
    # agent_app.invoke() is blocking (calls Ollama synchronously) - run it in
    # a worker thread so it doesn't freeze the whole server for other requests.
    result = await asyncio.to_thread(agent_app.invoke, {"error_type": data["error"]})
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

    IMPORTANT: predict_outcome() and agent_app.invoke() are both blocking
    calls (ONNX inference and synchronous Ollama HTTP calls respectively).
    Without asyncio.to_thread, FastAPI's single event loop would process
    requests strictly one-at-a-time, causing later requests to queue behind
    earlier ones and time out client-side even though the server eventually
    answers correctly - this was exactly the "Request timeout" issue seen
    when multiple throws were sent in quick succession from Unity.
    """
    samples = data["samples"]

    outcome, prob_made = await asyncio.to_thread(model_inference.predict_outcome, samples)

    if outcome == "Made":
        return {
            "outcome": "Made",
            "probability_made": prob_made,
            "error_type": None,
            "feedback": "Nice shot! Keep that same motion.",
        }

    error_type = await asyncio.to_thread(model_inference.classify_error_subtype, samples)
    result = await asyncio.to_thread(agent_app.invoke, {"error_type": error_type})

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