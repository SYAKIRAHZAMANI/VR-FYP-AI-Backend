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

COACH_VOICE = "en-US-GuyNeural"

# Biomechanical sub-queries so RAG retrieves diverse paragraphs from coaching_rules.txt
DIVERSE_ERROR_PROMPTS = {
    "Aim_Off": [
        "Aim_Off: Focus on shooting elbow alignment under the ball",
        "Aim_Off: Focus on guide hand releasing without pushing sideways",
        "Aim_Off: Focus on squaring shoulders and pointing toes to the basket"
    ],
    "Arc_Too_Flat": [
        "Arc_Too_Flat: Focus on pushing arm upward to ceiling for a 45 degree angle",
        "Arc_Too_Flat: Focus on elbow finishing above eye level on release",
        "Arc_Too_Flat: Focus on lofting the ball into the cylinder rather than driving forward"
    ],
    "Low_Velocity": [
        "Low_Velocity: Focus on bending knees and exploding power from legs",
        "Low_Velocity: Focus on powerful wrist flexion to generate backspin and carry",
        "Low_Velocity: Focus on full elbow extension without short-arming the release"
    ],
    "Jerky_Motion": [
        "Jerky_Motion: Focus on fluid momentum transfer without snapping shoulder",
        "Jerky_Motion: Focus on relaxing wrist grip to eliminate release tension",
        "Jerky_Motion: Focus on rhythmic continuous motion following the BEEF principle"
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
    for error_type, prompts in DIVERSE_ERROR_PROMPTS.items():
        for i, sub_prompt in enumerate(prompts):
            cache_key = f"{error_type}_{i}"
            print(f"Generating RAG Tip & Voice for [{cache_key}]...")
            
            # Query the RAG graph (Chroma DB + Ollama)
            result = agent_app.invoke({"error_type": sub_prompt})
            tip_text = result["feedback"]
            
            # Clean up quotes if LLM added any
            tip_text = tip_text.replace('"', '').strip()
            
            DIVERSE_FEEDBACK_CACHE[cache_key] = tip_text
            
            # Synthesize matching audio file (e.g., Aim_Off_0.mp3, Aim_Off_1.mp3)
            await generate_voice_file(tip_text, f"{cache_key}.mp3")

    print("--- 100% READY! DIVERSE RAG VOICE POOL GENERATED ---\n")


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