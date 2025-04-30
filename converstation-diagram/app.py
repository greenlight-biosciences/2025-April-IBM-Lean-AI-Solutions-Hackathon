import io
import asyncio
import streamlit as st
from pydub import AudioSegment
import httpx
import speech_recognition as sr

API_URL = "http://localhost:8005/v1/audio/transcriptions"

async def transcribe_remote(wav_bytes: bytes):
    files = {"file": ("recording.wav", wav_bytes, "audio/wav")}
    data = {"model": "granite-speech-3.3-8b"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(API_URL, files=files, data=data)
        resp.raise_for_status()
        return resp.json()

def transcribe_local(wav_bytes: bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

st.title("Granite Speech 3.3 Transcription via FastAPI or Local")

audio_bytes = st.audio_recorder()  
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    # Resample to 16 kHz mono
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")  
    seg = seg.set_frame_rate(16000).set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)
    
    # Choose transcription method
    use_local = st.checkbox("Use local transcription (for testing)")
    if use_local:
        try:
            result_text = transcribe_local(buf.read())
            st.markdown("**Transcript (Local):**")
            st.write(result_text)
        except Exception as e:
            st.error(f"Local transcription failed: {e}")
    else:
        try:
            result = asyncio.run(transcribe_remote(buf.read()))
            st.markdown("**Transcript (Remote):**")
            st.write(result["text"])
        except Exception as e:
            st.error(f"Remote transcription failed: {e}")
