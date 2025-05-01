# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import io
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import date
import uvicorn
import time

app = FastAPI()

# Force CPU-only
device = "cpu"
model_name = "ibm-granite/granite-speech-3.3-8b"

# Load processor and model (on CPU)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)  # already on CPU

API_KEY = "your-secret-api-key@12431242"  # Replace with your actual API key

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = "en"
    generation_time: float

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_choice: str = "granite-speech-3.3-8b",
    x_api_key: str = Depends(verify_api_key)
):
    if model_choice != "granite-speech-3.3-8b":
        raise HTTPException(400, detail="Only model 'granite-speech-3.3-8b' supported")

    # Read and load audio
    audio_bytes = await file.read()
    try:
        speech, sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {str(e)} -- wav, mp3, flac, ogg supported")


    # Mono + 16kHz
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(sr, 16000)(speech)
        sr = 16000

    # Build prompt
    today_str = date.today().isoformat()
    chat = [
        {"role": "system", "content": (
            "Knowledge Cutoff Date: April 2024.\n"
            f"Today's Date: {today_str}.\n"
            "You are Granite, developed by IBM. You are a helpful AI assistant."
        )},
        {"role": "user", "content": "<|audio|>Can you transcribe the speech into a written format?"}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Feature-extract + tokenize (CPU tensors by default)
    inputs = processor(
        prompt,
        speech,
        #sr,
        device=device, 
        return_tensors="pt"
    )
    start_time = time.time()

    # Generate (on CPU)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=1,
        do_sample=True,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Decode just the generated portion
    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0, prompt_len:].unsqueeze(0)
    text = tokenizer.batch_decode(
        generated, skip_special_tokens=True, add_special_tokens=False
    )[0]

    return TranscriptionResponse(text=text, generation_time=generation_time, language="en")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",           # import string: <filename_without_.py>:<app_instance_name>
        host="0.0.0.0",
        port=8000,
        workers=1,           # now supported because we passed an import string
        # reload=True          # optional: auto-reload on code changes
    )
