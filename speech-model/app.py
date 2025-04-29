# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import io
import torch, torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

app = FastAPI()

# Load Granite-Speech 3.3-8B for ASR
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b").to(device)

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = "en"

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = "granite-speech-3.3-8b"
):
    # Only Granite-Speech is supported here
    if model != "granite-speech-3.3-8b":
        raise HTTPException(400, detail="Only model 'granite-speech-3.3-8b' supported")

    # Read and process audio
    audio_bytes = await file.read()
    speech, sr = torchaudio.load(io.BytesIO(audio_bytes))
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt").to(device)

    # Generate transcription
    generated = model.generate(**inputs)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return TranscriptionResponse(text=text)
