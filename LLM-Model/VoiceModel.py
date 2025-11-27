import speech_recognition as sr
from gtts import gTTS
import requests
import playsound
import tempfile
import time

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDUhUL_PsFBXfKSqtitBIVCYGOzmxbUjBw"  # Replace with your actual key safely!
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/text-bison-001:generateText"

def listen_for_wake_word(recognizer, microphone):
    print("Listening for wake word 'WALLE'...")
    with microphone as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Heard: {text}")
        if "walle" in text.lower():
            return True
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    return False

def listen_for_question(recognizer, microphone):
    print("Listening for your question...")
    with microphone as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Question: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    return None

def get_gemini_response(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    payload = {
        "contents": [{"text": prompt}],
        "temperature": 0.7,
        "top_p": 0.8,
        "candidate_count": 1
    }
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        reply = result.get("candidates")[0].get("content").get("text")
        return reply
    else:
        print("API error:", response.text)
        return "Sorry, I couldn't get an answer."

def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        path = fp.name + ".mp3"
        tts.save(path)
        playsound.playsound(path)

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    print("Initializing WALLE assistant...")

    while True:
        if listen_for_wake_word(recognizer, microphone):
            speak("Yes, how can I help you?")
            question = listen_for_question(recognizer, microphone)
            if question:
                answer = get_gemini_response(question)
                print(f"WALLE: {answer}")
                speak(answer)
            time.sleep(1)

if __name__ == "__main__":
    main()
