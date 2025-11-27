import asyncio
import websockets
import speech_recognition as sr
import base64
import httpx
import cv2
import numpy as np
import pyttsx3
import threading
import time
from PIL import Image, ImageEnhance
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
import torch
from collections import deque
import requests
import json


# ---------------- CONFIG ----------------
PI_IP = "172.18.20.13"
AUDIO_URL = f"ws://{PI_IP}:8000/ws/audio"
VIDEO_URL = f"ws://{PI_IP}:8000/ws/video"
API_BASE_URL = f"http://{PI_IP}:8000/api"
RATE = 16000
SAMPLE_WIDTH = 2
ACCUM_SECONDS = 2
BUFFER_SIZE_BYTES = RATE * SAMPLE_WIDTH * ACCUM_SECONDS

# Robot Configuration
ROBOT_NAME = "chhotu"
WAKE_WORD = "chhotu"  # lowercase for matching

# Conversation history
CONVERSATION_HISTORY = deque(maxlen=10)

# Wake word state
wake_word_detected = False
wake_word_lock = threading.Lock()

# Debug mode
DEBUG_MODE = True  # Set to False to disable verbose logging

print("\n" + "=" * 70)
print(f"🤖 {ROBOT_NAME} - VOICE-CONTROLLED ROBOT - STARTING UP")
print("=" * 70)


# ---------------- GEMINI API (TEXT ONLY) ----------------
GEMINI_API_KEY = "AIzaSyDLE9wfQuSfXxvKGtA1sLWhgePW_bKfBVU"
GEMINI_TEXT_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

print("🔄 Configuring Gemini API (text-only)...")


def get_gemini_text_response(prompt: str) -> str:
    """Call Gemini API for TEXT ONLY (no images)."""
    try:
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        url = f"{GEMINI_TEXT_API_URL}?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code != 200:
            if DEBUG_MODE:
                print(f"❌ Gemini API error {response.status_code}: {response.text}")
            return "Sorry, I'm having trouble right now."

        result = response.json()

        # Parse response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            content = candidate.get("content", {})

            if "parts" in content:
                texts = [p.get("text", "") for p in content["parts"] if "text" in p]
                text = " ".join(texts).strip()
                if text:
                    # Keep concise for TTS
                    if len(text) > 250:
                        text = text[:250].rsplit('.', 1)[0] + '.'
                    return text

        return "Sorry, I got an unexpected response."

    except requests.exceptions.Timeout:
        if DEBUG_MODE:
            print("❌ Gemini API timeout")
        return "Sorry, that took too long to process."
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ Gemini request failed: {e}")
        return "Sorry, I couldn't get an answer."


# Test Gemini API
try:
    print("🔄 Testing Gemini API connection...")
    test_response = get_gemini_text_response(f"Say 'Hello, I am {ROBOT_NAME}!' in one short sentence.")
    if "sorry" not in test_response.lower():
        print(f"✅ Gemini API working! Response: '{test_response}'")
    else:
        print(f"⚠️  Gemini API test response: {test_response}")
except Exception as e:
    print(f"⚠️  Gemini API test failed: {e}")


# ---------------- TTS ----------------
print("🔄 Initializing Text-to-Speech...")
tts_lock = threading.Lock()
tts_queue = []
tts_running = True


def tts_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)
        print("✅ TTS Ready")
        
        while tts_running:
            with tts_lock:
                if tts_queue:
                    text = tts_queue.pop(0)
                else:
                    text = None
            if text:
                print(f"🔊 Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
            else:
                time.sleep(0.1)
    except Exception as e:
        print(f"❌ TTS initialization failed: {e}")


tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


def speak(text):
    with tts_lock:
        tts_queue.append(text)


# ---------------- LIGHTWEIGHT MODELS ----------------
print("🔄 Loading intent detection model...")
recognizer = sr.Recognizer()
intent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Intent model loaded")

print("🔄 Loading YOLO object detector...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")


# ---------------- BLIP-2 (LAZY LOADING) ----------------
blip_processor = None
blip_model = None
blip_loading_lock = threading.Lock()

print("ℹ️  BLIP-2 will load on first visual question (lazy loading)")


def load_blip2_model():
    """Lazy load BLIP-2 only when visual question is asked"""
    global blip_processor, blip_model
    
    with blip_loading_lock:
        if blip_model is not None:
            return True
        
        try:
            print("\n" + "=" * 70)
            print("🔄 FIRST VISUAL QUESTION - Loading BLIP-2...")
            print("⏳ This will take 30-90 seconds (one-time only)...")
            print("=" * 70)
            
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            print("📥 Loading BLIP-2 processor...")
            blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            print("✅ Processor loaded")
            
            print("📥 Loading BLIP-2 model weights...")
            blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
            if device == "cpu":
                blip_model = blip_model.to(device)
            
            blip_model.eval()
            
            print("=" * 70)
            print("✅ BLIP-2 LOADED SUCCESSFULLY!")
            print("=" * 70 + "\n")
            return True
            
        except Exception as e:
            print(f"\n❌ BLIP-2 loading failed: {e}")
            print("⚠️  Will use YOLO fallback for visual questions\n")
            return False


# ---------------- COMMANDS ----------------
COMMAND_TEMPLATES = {
    "DANCE_FULL": {"examples": ["dance", "full dance", "start dancing"], "endpoint": "dance/full"},
    "DANCE_WAVE": {"examples": ["wave", "say hello", "wave hand"], "endpoint": "dance/wave"},
    "DANCE_NOD": {"examples": ["nod", "agree", "nod head"], "endpoint": "dance/nod"},
    "DANCE_CURIOUS": {"examples": ["look around", "curious look", "be curious"], "endpoint": "dance/curious"},
    "DANCE_EXCITED": {"examples": ["excited dance", "celebrate", "happy dance"], "endpoint": "dance/excited"},
    "STOP_DANCE": {"examples": ["stop dancing", "stop", "halt"], "endpoint": "dance/stop"},
    "VISUAL_QUESTION": {"examples": ["what is in my hand", "what do you see", "look at this", "identify this", "show me", "describe", "what color"], "endpoint": None},
    "GENERAL_CHAT": {"examples": ["hello", "hi", "how are you", "tell me a joke", "what is", "tell me about", "explain"], "endpoint": None},
    "ROBOT_QUESTION": {"examples": ["who are you", "what are you", "tell me about yourself"], "endpoint": None},
}

print("🔄 Computing command embeddings...")
command_embeddings = {}
for cmd_name, cmd_data in COMMAND_TEMPLATES.items():
    embeddings = intent_model.encode(cmd_data["examples"], convert_to_tensor=True)
    command_embeddings[cmd_name] = {"embeddings": embeddings, "endpoint": cmd_data["endpoint"]}
print("✅ Command embeddings ready")


# ---------------- ROBOT CONTEXT ----------------
ROBOT_CONTEXT = f"""
You are {ROBOT_NAME}, a friendly educational robot.
You are designed to help children learn about robotics and AI in a fun and engaging way.
You use Raspberry Pi for control, with seven servos controlling head, neck, and arms.
You can perform dances like waving, nodding, and excited celebration.
You can see with a camera and answer questions using AI.
You are cheerful, helpful, and speak in a warm, approachable tone like a friendly companion.
Keep your responses concise (2-3 sentences maximum) since you use text-to-speech.
"""


# ---------------- VIDEO STREAM ----------------
class VideoStreamHandler:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
    
    async def stream_video(self):
        self.running = True
        try:
            if DEBUG_MODE:
                print(f"📹 Connecting to video stream at {VIDEO_URL}...")
            async with websockets.connect(VIDEO_URL, max_size=None) as ws:
                print("✅ Video stream connected")
                while self.running:
                    frame_bytes = await ws.recv()
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.frame_lock:
                            self.latest_frame = frame.copy()
                    await asyncio.sleep(0.03)
        except Exception as e:
            print(f"❌ Video stream error: {e}")
    
    def get_frame(self, retries=5, delay=0.1):
        for _ in range(retries):
            with self.frame_lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            time.sleep(delay)
        return None
    
    def stop(self):
        self.running = False


video_handler = VideoStreamHandler()


# ---------------- INTENT DETECTION ----------------
def detect_intent_with_similarity(text: str, threshold: float = 0.40):
    query_embedding = intent_model.encode(text, convert_to_tensor=True)
    best_match, best_score, best_endpoint = None, threshold, None
    for cmd_name, cmd_data in command_embeddings.items():
        sims = util.cos_sim(query_embedding, cmd_data["embeddings"])[0]
        score = sims.max().item()
        if score > best_score:
            best_match, best_score, best_endpoint = cmd_name, score, cmd_data["endpoint"]
    return best_match, best_score, best_endpoint


# ---------------- ROBOT API ----------------
async def call_robot_api(endpoint: str, method: str = "POST", data: dict = None):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=data) if method == "POST" else await client.get(url)
            if r.status_code == 200:
                if DEBUG_MODE:
                    print(f"✅ Robot API: {endpoint} - Success")
                return True, r.json()
            else:
                print(f"❌ Robot API: {endpoint} - Error {r.status_code}")
                return False, r.text
    except Exception as e:
        print(f"❌ Robot API request failed: {e}")
        return False, str(e)


# ---------------- QA FUNCTIONS ----------------
def get_conversation_context():
    if not CONVERSATION_HISTORY:
        return ""
    context = "Previous conversation:\n"
    for u, b in list(CONVERSATION_HISTORY)[-5:]:
        context += f"User: {u}\n{ROBOT_NAME}: {b}\n"
    return context


def answer_robot_question(question: str):
    """Answer questions about the robot using Gemini"""
    prompt = f"{ROBOT_CONTEXT}\n\nUser asks: {question}\n\nRespond as {ROBOT_NAME} in 2-3 sentences maximum. Be friendly and enthusiastic."
    return get_gemini_text_response(prompt)


def answer_general_chat(question: str):
    """General chat using Gemini"""
    context = get_conversation_context()
    prompt = f"{ROBOT_CONTEXT}\n\n{context}\nUser: {question}\n\nRespond as {ROBOT_NAME} in 2-3 sentences maximum. Be warm, helpful, and conversational."
    return get_gemini_text_response(prompt)


def answer_visual_question_blip2_yolo(question: str):
    """Visual QA using BLIP-2 + YOLO"""
    try:
        frame = video_handler.get_frame()
        if frame is None:
            return "I can't see anything right now. Make sure my camera is working!"
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        pil_image = Image.fromarray(frame_resized)
        
        # Enhance
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)

        # Save and display image (optional in debug mode)
        if DEBUG_MODE:
            try:
                import os
                save_dir = "debug_images"
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(save_dir, f"visual_qa_{timestamp}.jpg")
                pil_image.save(image_path)
                print(f"💾 Image saved to: {image_path}")
            except Exception as e:
                print(f"⚠️  Could not save image: {e}")

        # YOLO detection
        detected_objects = []
        try:
            results = yolo_model(frame_resized, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls)]
                    confidence = float(box.conf)
                    if confidence > 0.5 and cls_name not in detected_objects:
                        detected_objects.append(cls_name)
            
            if detected_objects:
                print(f"🔍 YOLO detected: {', '.join(detected_objects[:3])}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️  YOLO detection failed: {e}")

        # Load BLIP-2
        if not load_blip2_model():
            if detected_objects:
                return f"I can see {', '.join(detected_objects[:3])}."
            return "I can't clearly identify what's in the image."

        # BLIP-2 inference
        try:
            if DEBUG_MODE:
                print(f"🖼️  Analyzing with BLIP-2...")
            
            simple_question = question.lower()
            if "what" in simple_question and "hand" in simple_question:
                simple_question = "what is this?"
            elif "what" in simple_question and ("see" in simple_question):
                simple_question = "what is in the image?"
            elif "color" in simple_question:
                simple_question = "what color is this?"
            
            if DEBUG_MODE:
                print(f"📝 Question: '{simple_question}'")
            
            inputs = blip_processor(images=pil_image, text=simple_question, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = blip_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=3,
                    repetition_penalty=1.2,
                )
            
            blip_answer = blip_processor.decode(output[0], skip_special_tokens=True).strip()
            
            # Cleanup
            if simple_question in blip_answer.lower():
                blip_answer = blip_answer.lower().replace(simple_question, "").strip()
            blip_answer = blip_answer.replace("answer:", "").strip()
            
            if DEBUG_MODE:
                print(f"📸 BLIP-2: '{blip_answer}'")
            
            if not blip_answer or len(blip_answer) < 3:
                if detected_objects:
                    return f"I can see {', '.join(detected_objects[:3])}."
                return "I can't clearly identify what's in the image."
            
            if len(blip_answer) < 15 and detected_objects:
                return f"I see {', '.join(detected_objects[:2])}. {blip_answer.capitalize()}"
            
            return blip_answer.capitalize()
            
        except Exception as e:
            print(f"❌ BLIP-2 failed: {e}")
            if detected_objects:
                return f"I can see {', '.join(detected_objects[:3])}."
            return "I'm having trouble analyzing the image."

    except Exception as e:
        print(f"❌ Visual QA Error: {e}")
        return "I'm having camera trouble."


# ---------------- WAKE WORD RESPONSES ----------------
WAKE_WORD_RESPONSES = [
    "Yes, I'm listening!",
    "Yes, how can I help you?",
    "I'm here! What would you like me to do?",
    "Yes, tell me!",
    "Ready! What's your command?",
]


def get_wake_word_response():
    """Get a random wake word acknowledgment"""
    import random
    return random.choice(WAKE_WORD_RESPONSES)


# ---------------- AUDIO PROCESSING WITH WAKE WORD + DEBUG ----------------
async def process_audio_chunk(pcm_bytes: bytes):
    global wake_word_detected
    
    try:
        audio_data = sr.AudioData(pcm_bytes, RATE, SAMPLE_WIDTH)
        text = recognizer.recognize_google(audio_data).strip()
    except sr.UnknownValueError:
        # Could not understand audio - show in debug mode
        if DEBUG_MODE:
            print("🔇 [DEBUG] Audio chunk: (unintelligible)")
        return
    except sr.RequestError as e:
        if DEBUG_MODE:
            print(f"❌ [DEBUG] Speech recognition error: {e}")
        return
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ [DEBUG] Unexpected error: {e}")
        return
    
    if not text:
        return
    
    text_lower = text.lower()
    
    # ============ DEBUG OUTPUT - ALWAYS SHOW WHAT'S HEARD ============
    print(f"🎧 [DEBUG] Heard: '{text}' (wake_word_active={wake_word_detected})")
    
    # Show wake word matching details
    if WAKE_WORD in text_lower:
        print(f"   ✅ Wake word '{WAKE_WORD}' FOUND in text!")
    else:
        print(f"   ❌ Wake word '{WAKE_WORD}' NOT found in text")
        # Show similarity for debugging
        words_in_text = text_lower.split()
        print(f"   📝 Words detected: {words_in_text}")
    # ================================================================
    
    # Check for wake word
    with wake_word_lock:
        if not wake_word_detected:
            # Listening for wake word
            if WAKE_WORD in text_lower:
                print(f"\n🎤 ✅ WAKE WORD DETECTED: '{text}'")
                print(f"👂 {ROBOT_NAME} is now listening for command...")
                
                wake_word_detected = True
                
                # Respond to wake word
                response = get_wake_word_response()
                speak(response)
                
                # Reset wake word after 10 seconds if no command given
                async def reset_wake_word():
                    await asyncio.sleep(10)
                    global wake_word_detected
                    with wake_word_lock:
                        if wake_word_detected:
                            wake_word_detected = False
                            print(f"⏰ {ROBOT_NAME} stopped listening (timeout)")
                
                asyncio.create_task(reset_wake_word())
            else:
                # Ignore everything else when not awakened
                if DEBUG_MODE:
                    print(f"   ⏭️  Ignoring (wake word not active)")
                return
        else:
            # Wake word already detected, process command
            print(f"\n🎤 ✅ COMMAND RECEIVED: '{text}'")
            
            # Reset wake word state
            wake_word_detected = False
            
            intent, confidence, endpoint = detect_intent_with_similarity(text)
            print(f"🧠 Intent: {intent} (confidence: {confidence:.2f})")

            # Handle dance commands
            if intent and (intent.startswith("DANCE") or intent == "STOP_DANCE"):
                speak("Okay, let me do that for you")
                success, _ = await call_robot_api(endpoint)
                resp = "Done!" if success else "Sorry, I couldn't do that"
                speak(resp)
                CONVERSATION_HISTORY.append((text, resp))

            # Handle visual questions
            elif intent == "VISUAL_QUESTION":
                speak("Let me take a look")
                ans = answer_visual_question_blip2_yolo(text)
                print(f"🤖 {ROBOT_NAME}: {ans}")
                speak(ans)
                CONVERSATION_HISTORY.append((text, ans))

            # Handle robot questions
            elif intent == "ROBOT_QUESTION":
                ans = answer_robot_question(text)
                print(f"🤖 {ROBOT_NAME}: {ans}")
                speak(ans)
                CONVERSATION_HISTORY.append((text, ans))

            # Handle general chat
            elif intent == "GENERAL_CHAT":
                ans = answer_general_chat(text)
                print(f"🤖 {ROBOT_NAME}: {ans}")
                speak(ans)
                CONVERSATION_HISTORY.append((text, ans))

            # Fallback
            else:
                visual_keywords = ["see", "look", "show", "image", "picture", "object", "hand", "holding", "color", "identify"]
                if any(k in text.lower() for k in visual_keywords):
                    speak("Let me check that")
                    ans = answer_visual_question_blip2_yolo(text)
                else:
                    ans = answer_general_chat(text)
                print(f"🤖 {ROBOT_NAME}: {ans}")
                speak(ans)
                CONVERSATION_HISTORY.append((text, ans))


# ---------------- AUDIO RECEIVER ----------------
async def audio_receiver():
    buffer = bytearray()
    print(f"📡 Connecting to audio at {AUDIO_URL}...")
    try:
        async with websockets.connect(AUDIO_URL, max_size=None) as ws:
            print("✅ Audio stream connected")
            print("\n" + "=" * 70)
            print(f"👂 LISTENING FOR WAKE WORD: '{WAKE_WORD.upper()}'")
            print(f"🔍 DEBUG MODE: ENABLED (showing all audio recognition)")
            print(f"💡 Say '{WAKE_WORD.upper()}' to activate, then give your command")
            print(f"💡 Example: '{WAKE_WORD}' → (wait for response) → 'wave your hand'")
            print("=" * 70 + "\n")
            
            while True:
                chunk = await ws.recv()
                chunk_bytes = base64.b64decode(chunk) if isinstance(chunk, str) else chunk
                buffer.extend(chunk_bytes)
                if len(buffer) >= BUFFER_SIZE_BYTES:
                    pcm_to_process = bytes(buffer)
                    buffer.clear()
                    asyncio.create_task(process_audio_chunk(pcm_to_process))
    except Exception as e:
        print(f"❌ Audio connection error: {e}")


# ---------------- MAIN ----------------
async def main():
    print("\n" + "=" * 70)
    print(f"🚀 STARTING {ROBOT_NAME} SYSTEM...")
    print("=" * 70)
    print(f"🤖 Robot Name: {ROBOT_NAME}")
    print(f"🔊 Wake Word: '{WAKE_WORD.upper()}'")
    print(f"🔍 Debug Mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    print("✅ Text Chat: Gemini 1.5 Flash")
    print("✅ Visual QA: BLIP-2 OPT-2.7B + YOLO")
    print("✅ Speech: Google Speech Recognition + pyttsx3")
    print("=" * 70 + "\n")
    
    video_task = asyncio.create_task(video_handler.stream_video())
    await asyncio.sleep(2)
    
    try:
        await audio_receiver()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        global tts_running
        tts_running = False
        video_handler.stop()
        await video_task
        print("✅ Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n🛑 {ROBOT_NAME} stopped by user.")
