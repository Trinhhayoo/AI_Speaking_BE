from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
)
from google.cloud import texttospeech
from vertexai.generative_models import HarmCategory, HarmBlockThreshold


def get_gemini_text_response( model: GenerativeModel, prompt: str,
    generation_config: GenerationConfig):
    print(prompt)
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
    responses = model.generate_content(prompt,
    generation_config=generation_config,
    safety_settings=safety_settings)

    
    return responses.text

def AI_Generated_Voice(text: str):
    client = texttospeech.TextToSpeechClient()
    print(text)

    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-A")

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content