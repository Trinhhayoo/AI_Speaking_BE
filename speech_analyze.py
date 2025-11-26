from google.cloud import speech_v1p1beta1 as speech
import io
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speech_to_text_key.json"

client = speech.SpeechClient()

from pydub import AudioSegment
import io

def convert_to_mono(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    if audio.channels != 1:
        audio = audio.set_channels(1)
    return audio.export(format="wav").read()


def transcribe_with_confidence(audio_bytes):
    audio_bytes = convert_to_mono(audio_bytes)
    audio = speech.RecognitionAudio(content=audio_bytes)


    config = speech.RecognitionConfig(
        language_code="en-US",
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        use_enhanced=True,
        model="latest_long",
        enable_automatic_punctuation=True,
        adaptation=speech.SpeechAdaptation(
            phrase_sets=[
                speech.PhraseSet(
                    boost=20.0,
                    phrases=[
                        speech.PhraseSet.Phrase(value="think"),
                        speech.PhraseSet.Phrase(value="market"),
                        speech.PhraseSet.Phrase(value="pronunciation")
                    ]
                )
            ]
        )
    )


    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        alternative = result.alternatives[0]
        print("Transcript:", alternative.transcript)
        confidences = []

        for word_info in alternative.words:
            info = {
                "word": word_info.word,
                "start": word_info.start_time.total_seconds(),
                "end": word_info.end_time.total_seconds(),
                "confidence": word_info.confidence
            }
            confidences.append(info)
    return confidences
