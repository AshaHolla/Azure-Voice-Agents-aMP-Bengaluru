
import azure.cognitiveservices.speech as speechsdk
import openai
import os
from openai import AzureOpenAI

# Azure Speech Service configuration
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_o3")
model_name = "o3-mini"
deployment = "o3-mini"

subscription_key = os.getenv("AZURE_OPENAI_KEY_o3")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Configure Speech Service
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, 
    audio_config=audio_config
)

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, 
    audio_config=audio_config
    )


def speech_to_text()->str:

    result = speech_recognizer.recognize_once_async().get()

    while True:
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_text = result.text
            user_text_repeat = f"Generating response for {user_text}"
            synthesizer.speak_text_async(user_text_repeat).get()
            return user_text  

        elif result.reason == speechsdk.ResultReason.NoMatch:
            notif = "I didn’t catch that. Could you please repeat?"
            print("No speech could be recognized")
            synthesizer.speak_text_async(notif).get()

        elif result.reason == speechsdk.ResultReason.Canceled:
            print(f"Speech Recognition canceled: {result.cancellation_details.reason}")
            synthesizer.speak_text_async("Speech recognition canceled. Stopping now.").get()
            return None  

def get_ai_response(user_text)->str:
    response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise and conversational."},
        {"role": "user", "content": user_text}
    ],
    max_completion_tokens=1000,
    model=deployment
)
    text = response.choices[0].message.content
    return text

def text_to_speech(text):
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, 
        audio_config=audio_config
    )

    # Synthesize speech
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        synthesizer.speak_text_async("Conversion to voice failed").get()


def run_voice_conversation():
    greeting = "My name is Chat. What can I help you with today?"
    synthesizer.speak_text_async(greeting).get()
    while True:
        synthesizer.speak_text_async("Ask your question or say quit to exit").get()
        
        # Step 1: Speech to Text
        user_text = speech_to_text()
        print(user_text)
        if user_text.strip().lower() in ['quit', 'Quit','QUIT']:
            synthesizer.speak_text_async("goodbye").get()
            break
        
        if user_text:
            # Step 2: Get AI Response
            ai_response = get_ai_response(user_text)
            
            # Step 3: Text to Speech
            text_to_speech(ai_response)
        else:
            synthesizer.speak_text_async("Please repeat there was an error processing").get()



if __name__ == "__main__":
    run_voice_conversation()