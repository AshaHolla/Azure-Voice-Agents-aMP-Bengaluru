import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import threading
import time
import queue
import random
import os



endpoint= os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = "o3-mini"
deployment = "o3-mini"
subscription_key = os.getenv("OPENAI_API_KEY")
api_version = "2024-12-01-preview"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_recognition_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

# Audio I/O
audio_input = speechsdk.audio.AudioConfig(use_default_microphone=True)
audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, audio_config=audio_input
)
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, audio_config=audio_output
)

chat_history = [
    {"role": "system", "content": 
     """You are a helpful, real-time voice assistant. 
        Always respond naturally, conversationally. 
        Avoid bullets, numbered lists, line breaks or symbols.
        Keep it clear, concise, and friend-like.
     """}
]

FILLERS = [
    "Okay, got it. Let me think about that.",
    "Alright, let me work that out for you.",
    "Hmm, interesting. Give me a second.",
    "Sure thing, let me figure this out.",
    "Got it, let me check on that."
]

speak_lock = threading.Lock()
tts_queue = queue.Queue()
stop_speaking = False
quit_program = False

# --- TTS Worker ---
def tts_worker():
    """Background worker that speaks items from the queue one at a time."""
    global stop_speaking
    while True:
        text = tts_queue.get()
        if text is None:
            break  # shutdown signal

        with speak_lock:
            if stop_speaking:
                with tts_queue.mutex:
                    tts_queue.queue.clear()
                continue

            print(f"[TTS] Speaking: {text}")
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.Canceled:
                print("Speech synthesis canceled.")

def speak_text_queue(text):
    """Enqueue text for the TTS worker to process."""
    tts_queue.put(text)

def stop_speaking_if_needed():
    """Stop current TTS when user interrupts with speech."""
    global stop_speaking
    stop_speaking = True
    synthesizer.stop_speaking_async()
    with speak_lock:
        while not tts_queue.empty():
            tts_queue.get_nowait()
    print("[Interrupted] Cleared TTS queue.")

# --- AI Response Streaming ---
def stream_ai_response(user_text: str):
    """Stream AI response token by token, batch for natural TTS."""
    global stop_speaking, chat_history
    stop_speaking = False

    # Add user input to session memory
    chat_history.append({"role": "user", "content": user_text})

    response_stream = client.chat.completions.create(
        model=model_name,
        messages=chat_history,
        stream=True
    )

    buffer = ""
    collected_text = ""

    for event in response_stream:
        if stop_speaking:
            print("Stopping response stream...")
            break

        if not event.choices or not hasattr(event.choices[0].delta, "content"):
            continue

        token = event.choices[0].delta.content
        if not token:
            continue

        collected_text += token
        buffer += token

        # Flush buffer at punctuation or ~12 words for natural TTS
        if any(p in buffer for p in [".", "?", "!", ";",":"]):
            speak_text_queue(buffer.strip())
            buffer = ""

    # Flush remaining buffer
    if buffer.strip() and not stop_speaking:
        speak_text_queue(buffer.strip())

    print("\n[Full Response]:", collected_text)

    # Add assistant response to memory
    chat_history.append({"role": "assistant", "content": collected_text})

# --- Event Handlers ---
def recognizing_cb(evt):
    """Partial recognition while user speaks."""
    stop_speaking_if_needed()  # interrupt TTS instantly on new speech

def recognized_cb(evt):
    global quit_program
    if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
        return

    final_text = evt.result.text.strip()

    if "quit" in final_text.lower():
        print("Quitting session")
        quit_program = True
        stop_speaking_if_needed()
        tts_queue.put(None)  # shutdown TTS worker
        speech_recognizer.stop_continuous_recognition()
        return
    
    # Ignore empty or too-short speech
    if not final_text or len(final_text.split()) < 5:
        print(f"[Ignored] Detected speech too short or noise: '{final_text}'")
        return

    print(f"\nUser said: {final_text}")

    # --- Pick a random filler response ---
    filler = random.choice(FILLERS)

    # Speak filler synchronously
    print(f"[Filler] {filler}")
    result = synthesizer.speak_text_async(filler).get()

    # Now stream the AI response (on a separate thread)
    threading.Thread(
        target=stream_ai_response,
        args=(final_text,),
        daemon=True
    ).start()
    
def canceled_cb(evt):
    print(f"CANCELED: {evt.reason} {evt.error_details}")

def session_stopped_cb(evt):
    print("Session stopped")
    speech_recognizer.stop_continuous_recognition()

# --- Main Loop ---
def run_realtime_conversation():
    speak_text_queue("Hi, I'm Chat. Start speaking when ready. Say quit to stop.")

    # Start TTS worker
    threading.Thread(target=tts_worker, daemon=True).start()

    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    speech_recognizer.session_stopped.connect(session_stopped_cb)

    speech_recognizer.start_continuous_recognition_async().get()

    while not quit_program:
        time.sleep(0.5)

if __name__ == "__main__":
    run_realtime_conversation()
