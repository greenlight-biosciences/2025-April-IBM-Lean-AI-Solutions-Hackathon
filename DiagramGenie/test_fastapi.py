from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os

app = FastAPI()

# Load the Whisper model once
model = whisper.load_model("tiny")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe using Whisper
        result = model.transcribe(tmp_path, fp16=False, language="en")

        # Clean up
        os.remove(tmp_path)

        return JSONResponse(content={"text": result.get("text", "").strip()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
