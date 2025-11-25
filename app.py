import logging
from google.cloud import logging as cloud_logging
import os
from vertexai.generative_models import GenerativeModel
logging.basicConfig(level=logging.INFO)
from fastapi import FastAPI
from pydantic import BaseModel
from response_utils import get_gemini_text_response, AI_Generated_Voice
import vertexai
from fastapi.middleware.cors import CORSMiddleware
from vertexai.preview.generative_models import GenerationConfig
from fastapi.responses import StreamingResponse
import io
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


@app.post("/ai-voice")
def generate_ai_voice(req: Transcript):
    model = load_models()

    generation_config = GenerationConfig(
        max_output_tokens=1024,
        temperature=0.7,
    )
    
    ai_text = get_gemini_text_response(model, req.transcript, generation_config)

    audio_content = AI_Generated_Voice(ai_text)

    return StreamingResponse(io.BytesIO(audio_content), media_type="audio/mpeg")


    