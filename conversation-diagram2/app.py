import streamlit as st
import whisper
import pyaudio
import numpy as np
import wave
import tempfile
import os
from datetime import datetime
import subprocess
import sys
import threading
import queue
import time
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
import io
from AudioRecorder import AudioRecorder, check_ffmpeg, save_audio_chunk, create_workflow_image

# Initialize the Whisper model with a smaller model for faster processing
@st.cache_resource
def load_model():
    try:
        # Using 'tiny' model for faster transcription
        return whisper.load_model("tiny")
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        st.info("Please make sure FFmpeg is installed on your system.")
        return None

def display_chat_message(message, timestamp, source="User"):
    """Display a chat message with timestamp and source label"""
    # Assign colors based on source
    if source == "User (Manual)":
        bg_color = "#27ae60"  # Green for manual user entries
        label = "User (Manual):"
    elif source == "User (Transcribed)":
        bg_color = "#2c3e50"  # Dark blue for transcribed user messages
        label = "User:"
    elif source == "AI":
        bg_color = "#3498db"  # Light blue for AI messages (placeholder)
        label = "AI:"
    else: # Default fallback
        bg_color = "#7f8c8d"  # Gray for unknown source
        label = f"{source}:"
        
    st.markdown(f"""
        <div style='
            background-color: {bg_color};
            color: white;
            padding: 8px;
            border-radius: 8px;
            margin: 4px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        '>
            <div style='font-size: 0.7em; color: #bdc3c7; margin-bottom: 3px; font-weight: bold;'>
                {label} <span style='color: #95a5a6; font-weight: normal;'>{timestamp}</span>
            </div>
            <div style='font-size: 0.9em;'>{message}</div>
        </div>
    """, unsafe_allow_html=True)

# Callback function to handle manual message submission
def handle_manual_submit(message_text):
    if message_text.strip():
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_transcription = {
            "text": message_text.strip(),
            "timestamp": timestamp,
            "source": "User (Manual)" # Changed from is_manual
        }
        if 'transcriptions' not in st.session_state:
            st.session_state.transcriptions = []
        st.session_state.transcriptions.append(new_transcription)
        # Let clear_on_submit=True handle clearing

def main():
    st.set_page_config(layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Model selection
        model_size = st.selectbox(
            "Select Model Size",
            ["tiny", "base", "small"],
            index=0,
            help="Smaller models are faster but less accurate"
        )
        
        # Recording settings
        st.subheader("Recording Settings")
        chunk_duration = st.slider("Chunk Duration (seconds)", 1.0, 5.0, 3.0, 0.5)
        overlap = st.slider("Overlap (seconds)", 0.5, 2.5, 1.5, 0.5)
        
        # Status
        st.subheader("Status")
        if 'is_recording' in st.session_state:
            if st.session_state.is_recording:
                st.error("Recording in progress...")
            else:
                st.success("Ready to record")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Real-time Audio Transcription")
        
        # Workflow visualization
        st.subheader("Workflow")
        workflow_img = create_workflow_image()
        st.image(workflow_img, use_column_width=True)

        # Convert image to bytes for download
        buf = io.BytesIO()
        workflow_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label=":arrow_down: Download Workflow Image",
            data=byte_im,
            file_name="workflow_diagram.png",
            mime="image/png"
        )
        
        # Record controls
        col_controls = st.columns(2)
        with col_controls[0]:
            if st.button(":microphone: Start Recording", disabled=st.session_state.get('is_recording', False)):
                if 'recorder' not in st.session_state:
                    st.session_state.recorder = AudioRecorder()
                st.session_state.recorder.start_recording()
                st.session_state.is_recording = True
                st.rerun()
        
        with col_controls[1]:
            if st.button(":octagonal_sign: Stop Recording", disabled=not st.session_state.get('is_recording', False)):
                st.session_state.recorder.stop_recording()
                st.session_state.is_recording = False
                st.rerun()
    
    with col2:
        st.title("Transcription")
        
        # Top controls: Toggle and Download button area
        top_col1, top_col2 = st.columns([1, 1])
        with top_col1:
            show_transcript = st.checkbox("Show Transcript", value=True)
        with top_col2:
            download_button_placeholder = st.empty() # Placeholder for download button

        # Conditionally display transcript elements
        if show_transcript:
            # --- Download Button Logic (Moved inside conditional block) ---
            if st.session_state.get('transcriptions', []): # Check if there are transcriptions
                with download_button_placeholder.container(): # Populate the placeholder
                    timestamp_dl = datetime.now().strftime("%Y%m%d_%H%M%S")
                    transcription_text = "\n".join([
                        f"[{msg['timestamp']}] {msg.get('source', 'Unknown')}: {msg['text']}"
                        for msg in st.session_state.transcriptions
                        if isinstance(msg, dict) and "text" in msg and "timestamp" in msg
                    ])
                    st.download_button(
                        label=":arrow_down: Download Transcript",
                        data=transcription_text,
                        file_name=f"transcription_{timestamp_dl}.txt",
                        mime="text/plain",
                        key="download_button"
                    )
            # --- End Download Button Logic ---

            # --- Manual message input form (inside conditional block) ---
            with st.form("manual_message_form", clear_on_submit=True):
                manual_message = st.text_area("Add Manual Message", 
                                            placeholder="Type your message here...",
                                            height=80,
                                            key="manual_input")
                submit_manual = st.form_submit_button(":pencil2: Add Message")

                if submit_manual: # Check if submitted
                    handle_manual_submit(st.session_state.manual_input)
            # --- End Manual message input form ---

            # --- Placeholder container for messages (inside conditional block) ---
            chat_placeholder = st.container()
            with chat_placeholder:
                # Display messages in reverse order (newest first)
                for msg in reversed(st.session_state.get('transcriptions', [])):
                    if isinstance(msg, dict) and "text" in msg and "timestamp" in msg:
                        display_chat_message(
                            msg["text"], 
                            msg["timestamp"],
                            source=msg.get("source", "User (Transcribed)")
                        )
            # --- End Placeholder container ---
        else:
             # Optionally clear the placeholder if transcript is hidden
             download_button_placeholder.empty()

    # Initialize session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'model' not in st.session_state or st.session_state.get('model_size') != model_size:
        st.session_state.model_size = model_size
        st.session_state.model = whisper.load_model(model_size)
    if 'last_transcription' not in st.session_state:
        st.session_state.last_transcription = ""
    
    # Check for FFmpeg
    if not check_ffmpeg():
        st.error("FFmpeg is not installed or not in PATH. Please install FFmpeg to use this application.")
        st.info("You can download FFmpeg from: https://ffmpeg.org/download.html")
        st.info("After installing, make sure to add FFmpeg to your system PATH.")
        st.info("Current PATH: " + os.environ.get('PATH', ''))
        return
    
    # Process audio and transcribe
    if st.session_state.is_recording:
        sample_rate = 16000
        chunk_size = 1024
        
        # Calculate number of chunks needed
        samples_per_chunk = int(chunk_duration * sample_rate / chunk_size)
        overlap_samples = int(overlap * sample_rate / chunk_size)
        
        # Collect audio chunks
        audio_chunks = []
        start_time = time.time()
        
        while time.time() - start_time < chunk_duration and st.session_state.is_recording:
            chunk = st.session_state.recorder.get_audio_chunk()
            if chunk:
                audio_chunks.append(chunk)
        
        if audio_chunks:
            # Combine chunks and transcribe
            audio_data = b''.join(audio_chunks)
            audio_file = save_audio_chunk(audio_data)
            
            if audio_file:
                try:
                    result = st.session_state.model.transcribe(
                        audio_file,
                        fp16=False,
                        language="en"
                    )
                    
                    if result["text"].strip():
                        # Only add new content that hasn't been transcribed before
                        new_text = result["text"]
                        if st.session_state.last_transcription:
                            # Find the overlapping part
                            overlap_text = st.session_state.last_transcription[-len(new_text)//2:]
                            if overlap_text in new_text:
                                # Remove the overlapping part
                                new_text = new_text[new_text.find(overlap_text) + len(overlap_text):]
                        
                        if new_text.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            new_transcription = {
                                "text": new_text,
                                "timestamp": timestamp,
                                "is_manual": False
                            }
                            st.session_state.transcriptions.append(new_transcription)
                            st.session_state.last_transcription = result["text"]
                            # No need to display here, rerun will handle it
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
                finally:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
        
        # Rerun to update the display with new transcriptions
        st.rerun()

if __name__ == "__main__":
    main() 