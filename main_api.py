import asyncio
import os
import random
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import edge_tts
from intelligent_agent import app as agent_app
import model_inference

app = FastAPI()

# 1. Setup Audio Directory
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

COACH_VOICE = "en-US-SteffanNeural"

async def generate_voice_file(text: str, filename: str):
    """Generates a calm, natural neural voice file"""
    filepath = os.path.join(AUDIO_DIR, filename)
    
    # Removed the speed and pitch tweaks so it sounds like a real human!
    communicate = edge_tts.Communicate(
        text=text,
        voice=COACH_VOICE
    )
    await communicate.save(filepath)

# Punchy, energetic 5-to-8 word court-side cues!
# These are the search queries sent to the Vector DB to pull the correct FIBA paragraphs!
DIVERSE_ERROR_PROMPTS = {
    "Aim_Off": [
        "Lateral deviation and shooting elbow flaring outward abduction",
        "Non-dominant guide hand pushing the ball sideways side spin",
        "Unstable base, narrow stance, and poor alignment feet"
    ],
    "Arc_Too_Flat": [
        "Flight arc is too flat, need optimal entry angle 45 degrees",
        "Elevate release point shooting elbow finishes above eye level",
        "Short shots front iron insufficient leg drive jumping backward"
    ],
    "Low_Velocity": [
        "Power from lower body legs flexed knees ground reaction forces",
        "Smooth upward momentum unbroken transfer of force",
        "Proper follow-through holding release pose fully extended"
    ],
    "Jerky_Motion": [
        "Excessive muscular tension restricts wrist flexibility",
        "Hitching or pausing at the forehead disrupts momentum",
        "External focus of attention smooth fluid trajectory"
    ]
}
# Cache to store pre-warmed diverse feedback strings
DIVERSE_FEEDBACK_CACHE = {}

# Tracks the last used tip index so we never repeat the same tip twice in a row
last_used_variation = {
    "Aim_Off": 0,
    "Arc_Too_Flat": 0,
    "Low_Velocity": 0,
    "Jerky_Motion": 0
}

async def generate_voice_file(text: str, filename: str):
    """Generates an energetic neural voice file using Edge TTS"""
    filepath = os.path.join(AUDIO_DIR, filename)
    communicate = edge_tts.Communicate(
        text=text,
        voice=COACH_VOICE,
        rate="+8%",
        pitch="+2Hz"
    )
    await communicate.save(filepath)

@app.on_event("startup")
async def startup_event():
    # 1. Load ONNX Model & Scaler
    model_inference.load_model(
        onnx_path="LSTM_Basketball_Classifier.onnx",
        scaler_path="scaler_params_final.json",
    )
    
    print("\n--- PRE-WARMING DIVERSE RAG COACHING TIPS & VOICES ---")

    # 2. Pre-generate 'Made' Voice
    nice_shot_text = "Nice shot! Keep that same smooth motion."
    await generate_voice_file(nice_shot_text, "Made.mp3")

# 3. Pre-generate 3 Distinct RAG Tips & MP3s for every Error Type
    for error_type, search_queries in DIVERSE_ERROR_PROMPTS.items():
        for i, query in enumerate(search_queries):
            cache_key = f"{error_type}_{i}"
            print(f"Searching FIBA manual and generating tip for [{cache_key}]...")
            
            # 1. Query the RAG graph (Vector DB pulls FIBA text -> LLM summarizes to 8 words)
            result = agent_app.invoke({"error_type": query})
            tip_text = result["feedback"].replace('"', '').strip()
            
            # 2. Save the punchy AI tip to the cache
            DIVERSE_FEEDBACK_CACHE[cache_key] = tip_text
            
            # 3. Synthesize neural audio
            await generate_voice_file(tip_text, f"{cache_key}.mp3")

    print("--- 100% READY! PUNCHY COACH VOICE POOL GENERATED ---\n")


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
    current_idx = last_used_variation.get(base_error, 0)
    next_idx = (current_idx + 1) % len(DIVERSE_ERROR_PROMPTS.get(base_error, [0]))
    last_used_variation[base_error] = next_idx

    keyed_error = f"{base_error}_{next_idx}"
    
    # Retrieve the pre-warmed RAG tip from our pool
    feedback_text = DIVERSE_FEEDBACK_CACHE.get(
        keyed_error, 
        "Focus on maintaining a high release point and a smooth follow-through."
    )

    return {
        "outcome": "Miss",
        "probability_made": prob_made,
        "error_type": keyed_error,  # e.g., "Aim_Off_1" -> Unity fetches Aim_Off_1.mp3 automatically!
        "feedback": feedback_text,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}