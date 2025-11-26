import logging
from google.cloud import logging as cloud_logging
import os
from vertexai.generative_models import GenerativeModel, Part
logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel
from response_utils import get_gemini_text_response, AI_Generated_Voice
import vertexai
from fastapi.middleware.cors import CORSMiddleware
from vertexai.preview.generative_models import GenerationConfig
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile, FastAPI
import io
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
# from io import BytesIO

log_client = cloud_logging.Client()
log_client.setup_logging()

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("REGION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI()

def load_models():
    model = GenerativeModel("gemini-2.0-flash")
    return model

class Transcript(BaseModel):
    transcript: str

class Audio(BaseModel):
    audio: UploadFile

# Allow frontend origin
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],            
)

safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

@app.post("/describe-audio")
async def describe_audio(audio: UploadFile = File(...)):
    model = load_models()

    # Read bytes from uploaded file
    content = await audio.read()

    audio_part = Part.from_data(
        mime_type=audio.content_type,   # e.g. audio/wav
        data=content
    )

    generation_config = GenerationConfig(
        max_output_tokens=1024,
        temperature=0.7,
    )

    result = model.generate_content(
        ["Return the transcript of the audio clip", audio_part],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    return {"transcript": result.text}


@app.post("/ai-voice")
def generate_ai_voice(req: Transcript):
    model = load_models()

    generation_config = GenerationConfig(
        max_output_tokens=1024,
        temperature=0.7,
    )
    
    ai_text = get_gemini_text_response(model, req.transcript, generation_config, safety_settings)

    audio_content = AI_Generated_Voice(ai_text)

    return {
        "audio": StreamingResponse(io.BytesIO(audio_content), media_type="audio/mpeg"),
        "transcript": ai_text
    }


    