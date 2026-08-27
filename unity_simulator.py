import requests
import time
import random

# This mimics the Unity DataRecorder script
def simulate_unity_game():
    errors = ["Early_Release", "Too_Fast", "Elbow_Drop", "Jerky_Motion"]
    print("Simulating VR Throwing...")
    
    for i in range(3):
        error = random.choice(errors)
        print(f"\n[Unity] User error detected: {error}")
        
        # Send to your AI Backend
        response = requests.post("http://127.0.0.1:8000/analyze_movement", json={"error": error})
        print(f"[AI Coach] Says: {response.json()['feedback']}")
        time.sleep(2)

if __name__ == "__main__":
    simulate_unity_game()