"""
Audio Recording Test Script
Records audio from microphone and converts to text using:
1. Google Speech Recognition (online, no setup)
2. Vosk (offline, faster, requires model)
"""

import pyaudio
import wave
import speech_recognition as sr
import os
import time

print("=" * 60)
print("🎤 AUDIO RECORDING TEST SCRIPT")
print("=" * 60)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Record for 5 seconds
WAVE_OUTPUT_FILENAME = "test_recording.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

print("\n📋 Available audio devices:")
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  [{i}] {info['name']} (Channels: {info['maxInputChannels']})")

print("\n" + "=" * 60)
print(f"🎙️  Recording {RECORD_SECONDS} seconds of audio...")
print("   Speak clearly into your microphone!")
print("=" * 60)

# Start recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

frames = []

# Record countdown
for i in range(RECORD_SECONDS, 0, -1):
    print(f"⏱️  {i}...", end=" ", flush=True)
    for _ in range(int(RATE / CHUNK * 1)):  # 1 second worth of chunks
        data = stream.read(CHUNK)
        frames.append(data)

print("\n✅ Recording finished!")

# Stop recording
stream.stop_stream()
stream.close()
audio.terminate()

# Save recording to file
print(f"💾 Saving to {WAVE_OUTPUT_FILENAME}...")
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(f"✅ Saved!")

# Convert to text using speech recognition
recognizer = sr.Recognizer()

print("\n" + "=" * 60)
print("🔄 SPEECH TO TEXT CONVERSION")
print("=" * 60)

# Test 1: Google Speech Recognition
print("\n[1] Testing Google Speech Recognition (online)...")
try:
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)
    
    start_time = time.time()
    text_google = recognizer.recognize_google(audio_data)
    google_time = time.time() - start_time
    
    print(f"✅ Google Result ({google_time:.2f}s):")
    print(f"   📝 '{text_google}'")
except sr.UnknownValueError:
    print("❌ Google could not understand audio")
except sr.RequestError as e:
    print(f"❌ Google API error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Vosk (if available)
print("\n[2] Testing Vosk (offline, fast)...")
try:
    from vosk import Model, KaldiRecognizer
    import json
    
    # Try to find Vosk model
    model_paths = [
        "vosk-model-small-en-us-0.15",
        "vosk-model-en-us-0.42-gigaspeech",
        "vosk-model-en-us-0.22",
        "vosk-model-en-us-0.22-lgraph",
        "model",
        "vosk-model",
        # Also check in parent directory
        "../vosk-model-small-en-us-0.15",
        "../vosk-model-en-us-0.42-gigaspeech",
        "../vosk-model-en-us-0.22",
        # Check in common locations
        r"C:\Users\Abhiram\OneDrive\Desktop\Wall-E\VAMSHI-DONOT-TOUCH\vosk-model-en-us-0.42-gigaspeech"
        r"C:\Users\Abhiram\OneDrive\Desktop\Wall-E\vosk-model-small-en-us-0.15",
        r"C:\Users\Abhiram\OneDrive\Desktop\Wall-E\vosk-model-en-us-0.42-gigaspeech",
        r"C:\Users\Abhiram\OneDrive\Desktop\Wall-E\VAMSHI-DONOT-TOUCH\vosk-model-small-en-us-0.15",
        r"VAMSHI-DONOT-TOUCH\vosk-model-en-us-0.42-gigaspeech"
    ]
    
    print("   Searching for Vosk model in:")
    for path in model_paths:
        if os.path.exists(path):
            print(f"   ✅ Found: {path}")
        else:
            print(f"   ❌ Not found: {path}")
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print(f"\n   ✅ Using model: {model_path}")
        print(f"   📂 Full path: {os.path.abspath(model_path)}")
        print(f"   Loading Vosk model...")
        vosk_model = Model(model_path)
        vosk_rec = KaldiRecognizer(vosk_model, RATE)
        print(f"   ✅ Model loaded successfully!")
        
        # Read audio file
        wf = wave.open(WAVE_OUTPUT_FILENAME, "rb")
        
        start_time = time.time()
        
        # Process audio
        while True:
            data = wf.readframes(CHUNK)
            if len(data) == 0:
                break
            vosk_rec.AcceptWaveform(data)
        
        # Get final result
        result = json.loads(vosk_rec.FinalResult())
        text_vosk = result.get("text", "")
        vosk_time = time.time() - start_time
        
        wf.close()
        
        if text_vosk:
            print(f"✅ Vosk Result ({vosk_time:.2f}s):")
            print(f"   📝 '{text_vosk}'")
            print(f"\n⚡ Vosk was {google_time/vosk_time:.1f}x faster than Google!")
        else:
            print("❌ Vosk could not understand audio")
    else:
        print("\n⚠️  Vosk model not found in any searched location!")
        print("\n   📁 Current directory:", os.getcwd())
        print("\n   To use Vosk:")
        print("   1. Download from: https://alphacephei.com/vosk/models")
        print("      - Recommended: vosk-model-en-us-0.42-gigaspeech.zip (2.3GB, most accurate)")
        print("      - Or smaller: vosk-model-small-en-us-0.15.zip (40MB)")
        print("   2. Extract the ZIP file")
        print("   3. Move the extracted folder to:")
        print(f"      {os.path.join(os.getcwd(), 'vosk-model-en-us-0.42-gigaspeech')}")
        print("\n   Or enter the model path manually:")
        try:
            manual_path = input("   Enter Vosk model path (or press Enter to skip): ").strip()
            if manual_path and os.path.exists(manual_path):
                print(f"\n   🔄 Retrying with: {manual_path}")
                vosk_model = Model(manual_path)
                vosk_rec = KaldiRecognizer(vosk_model, RATE)
                
                # Process audio with manually entered path
                wf = wave.open(WAVE_OUTPUT_FILENAME, "rb")
                start_time = time.time()
                
                while True:
                    data = wf.readframes(CHUNK)
                    if len(data) == 0:
                        break
                    vosk_rec.AcceptWaveform(data)
                
                result = json.loads(vosk_rec.FinalResult())
                text_vosk = result.get("text", "")
                vosk_time = time.time() - start_time
                wf.close()
                
                if text_vosk:
                    print(f"✅ Vosk Result ({vosk_time:.2f}s):")
                    print(f"   📝 '{text_vosk}'")
                    if 'google_time' in locals():
                        print(f"\n⚡ Vosk was {google_time/vosk_time:.1f}x faster than Google!")
                else:
                    print("❌ Vosk could not understand audio")
            elif manual_path:
                print(f"   ❌ Path not found: {manual_path}")
        except KeyboardInterrupt:
            print("\n   Skipped manual input")
        except Exception as e:
            print(f"   ❌ Error with manual path: {e}")
        
except ImportError:
    print("⚠️  Vosk not installed!")
    print("\n   Install with: pip install vosk")
except Exception as e:
    print(f"❌ Vosk error: {e}")

# Test 3: Real-time streaming (like Wall-E uses)
print("\n[3] Testing real-time streaming mode...")
print("   (This is how Wall-E processes audio)")

try:
    # Simulate streaming by reading in chunks
    with open(WAVE_OUTPUT_FILENAME, 'rb') as f:
        # Skip WAV header (44 bytes)
        f.read(44)
        
        # Read all PCM data
        pcm_data = f.read()
    
    # Test with speech_recognition
    audio_data = sr.AudioData(pcm_data, RATE, 2)
    
    start_time = time.time()
    text_streaming = recognizer.recognize_google(audio_data)
    streaming_time = time.time() - start_time
    
    print(f"✅ Streaming Result ({streaming_time:.2f}s):")
    print(f"   📝 '{text_streaming}'")
    
except Exception as e:
    print(f"❌ Streaming test error: {e}")

print("\n" + "=" * 60)
print("🎉 TESTING COMPLETE!")
print("=" * 60)

# Cleanup
print(f"\n🗑️  Cleaning up...")
try:
    os.remove(WAVE_OUTPUT_FILENAME)
    print(f"   Deleted {WAVE_OUTPUT_FILENAME}")
except:
    pass

print("\n💡 Tips:")
print("   • For faster recognition, use Vosk (offline)")
print("   • Speak clearly and at normal pace")
print("   • Reduce background noise for better accuracy")
print("   • Adjust RECORD_SECONDS if you need more time")

print("\n" + "=" * 60)
