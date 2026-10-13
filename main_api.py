import asyncio
import json
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import edge_tts
from intelligent_agent import app as agent_app, FALLBACK_TIPS, validate_tip
import model_inference

app = FastAPI()

# 1. Setup Audio Directory
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

COACH_VOICE = "en-US-SteffanNeural"

# Files (created in the folder you run uvicorn from)
POOL_OUT = "tip_pool.json"                # written every start-up: tip + evidence, for you to read/review
POOL_REVIEWED = "tip_pool_reviewed.json"  # OPTIONAL: your approved tips. If a key is here, it is used as-is.


async def generate_voice_file(text: str, filename: str):
    """Neural voice file. Default speed (slower than before) so beginners can follow it."""
    filepath = os.path.join(AUDIO_DIR, filename)
    communicate = edge_tts.Communicate(text=text, voice=COACH_VOICE)
    await communicate.save(filepath)


# Each variation = (retrieval query, what the tip should be about).
# Queries are written in plain sentences that match the wording of coaching_rules.txt,
# and every variation covers a DIFFERENT idea so the three tips are not near-duplicates.
DIVERSE_ERROR_PROMPTS = {
    "Aim_Off": [
        ("keep the shooting elbow tucked underneath the ball, not flared out",
         "keeping the elbow under the ball"),
        ("shoulder, elbow, wrist and finger line up with the center of the basket",
         "lining the arm up with the hoop"),
        ("follow through with fingers pointing straight at the target without twisting the hand",
         "pointing the hand straight at the hoop when letting go"),
    ],
    "Arc_Too_Flat": [
        ("elevate the release point so the elbow finishes above eye level",
         "lifting the elbow up high"),
        ("reach into the high cookie jar, finish with the arm high",
         "reaching up high at the end of the throw"),
        ("visualize the ball tracing a high arc through the air above the rim",
         "sending the ball high above the hoop (focus on where the ball goes)"),
    ],
    "Low_Velocity": [
        ("extend the shooting arm fully upward at release",
         "pushing the arm fully up and straight"),
        ("use the whole body and arm, not only the hand, to generate power",
         "using the whole arm to add power"),
        ("release with smooth upward energy so the ball carries to the basket",
         "pushing the ball up and forward with more energy"),
    ],
    "Jerky_Motion": [
        ("continuous unbroken motion, do not pause the ball at the forehead",
         "not stopping halfway through the throw"),
        ("keep the shoulder, arm and hand relaxed, tension makes the motion rough",
         "relaxing the shoulder and arm"),
        ("smooth upward momentum into a high extension gives a repeatable rhythm",
         "one smooth, even motion from start to finish"),
    ],
}

# Cache to store pre-warmed feedback strings, e.g. "Aim_Off_1" -> "Keep your elbow ..."
DIVERSE_FEEDBACK_CACHE = {}

# Tracks the last used tip index so we never repeat the same tip twice in a row
last_used_variation = {k: 0 for k in DIVERSE_ERROR_PROMPTS}

GENERIC_FALLBACK = "Throw the ball higher in one smooth motion."


@app.on_event("startup")
async def startup_event():
    # 1. Load ONNX Model & Scaler
    model_inference.load_model(
        onnx_path="LSTM_Basketball_Classifier.onnx",
        scaler_path="scaler_params_final.json",
    )

    print("\n--- PRE-WARMING COACHING TIPS & VOICES ---")

    # 2. 'Made' voice
    await generate_voice_file("Nice shot! Keep that same smooth motion.", "Made.mp3")

    # 3. Optional: tips you reviewed and approved (used exactly as written)
    reviewed = {}
    if os.path.exists(POOL_REVIEWED):
        with open(POOL_REVIEWED, "r", encoding="utf-8") as f:
            reviewed = json.load(f)
        print(f"Loaded {len(reviewed)} reviewed tips from {POOL_REVIEWED}")

    report = {}

    # 4. Three tips + MP3s for every error type
    for error_type, variations in DIVERSE_ERROR_PROMPTS.items():
        for i, (query, focus) in enumerate(variations):
            cache_key = f"{error_type}_{i}"

            if cache_key in reviewed:
                tip_text = reviewed[cache_key].strip()
                report[cache_key] = {"tip": tip_text, "source": "reviewed by researcher",
                                     "query": query, "fallback_used": False}
                print(f"[{cache_key}] (reviewed): {tip_text}")
            else:
                print(f"Searching coaching manual and generating tip for [{cache_key}]...")
                try:
                    # agent_app.invoke is blocking -> run in a thread so the event loop is not frozen
                    result = await asyncio.to_thread(
                        agent_app.invoke,
                        {"error_type": cache_key, "query": query, "focus": focus},
                    )
                    tip_text = result["feedback"].strip()
                    report[cache_key] = {"tip": tip_text, "source": result.get("source", ""),
                                         "query": query, "fallback_used": result.get("used_fallback", False)}
                except Exception as e:
                    print(f"   agent failed ({e}); using hand-written fallback")
                    tip_text = FALLBACK_TIPS[error_type][i]
                    report[cache_key] = {"tip": tip_text, "source": "", "query": query, "fallback_used": True}

            DIVERSE_FEEDBACK_CACHE[cache_key] = tip_text
            await generate_voice_file(tip_text, f"{cache_key}.mp3")

    # 5. Save a readable report: review it, edit the tips, then save the edited
    #    {"Aim_Off_0": "tip", ...} as tip_pool_reviewed.json to freeze them.
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    n_fb = sum(1 for v in report.values() if v["fallback_used"])
    print(f"--- READY! {len(report)} tips ({n_fb} from fallback). Details saved to {POOL_OUT} ---\n")


@app.post("/analyze_movement")
async def analyze_movement(data: dict):
    result = await asyncio.to_thread(agent_app.invoke, {"error_type": data["error"]})
    return {"feedback": result["feedback"]}


@app.post("/predict_and_coach")
async def predict_and_coach(data: dict):
    samples = data["samples"]

    # 1. Run LSTM Inference
    outcome, prob_made = await asyncio.to_thread(model_inference.predict_outcome, samples)

    if outcome == "Made":
        return {
            "outcome": "Made",
            "probability_made": prob_made,
            "error_type": None,
            "feedback": "Nice shot! Keep that same smooth motion.",
        }

    # 2. Classify Biomechanical Error Subtype
    base_error = await asyncio.to_thread(model_inference.classify_error_subtype, samples)

    # 3. Round-Robin Rotation: pick the next variation so tips NEVER repeat back-to-back
    n_variations = len(DIVERSE_ERROR_PROMPTS.get(base_error, [0]))
    next_idx = (last_used_variation.get(base_error, 0) + 1) % n_variations
    last_used_variation[base_error] = next_idx

    keyed_error = f"{base_error}_{next_idx}"
    feedback_text = DIVERSE_FEEDBACK_CACHE.get(keyed_error, GENERIC_FALLBACK)

    return {
        "outcome": "Miss",
        "probability_made": prob_made,
        "error_type": keyed_error,  # e.g. "Aim_Off_1" -> Unity fetches Aim_Off_1.mp3
        "feedback": feedback_text,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}