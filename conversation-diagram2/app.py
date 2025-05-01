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
from langchain_ibm import ChatWatsonx
import queue
import base64
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_openai import AzureChatOpenAI
import time
from PIL import Image
import io, json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from AudioRecorder import AudioRecorder, check_ffmpeg, save_audio_chunk, create_workflow_image
from fastmcp import Client
st.set_page_config(layout="wide")

async def get_workflow_image():
    """Fetch a workflow image from an MCP resource with retries and logging"""

    async with Client("http://localhost:8000/sse") as client:
        # Make sure this is a valid URI to a single image file!
        resource_uri = "file://graph_images"
        response = await client.read_resource(resource_uri)
        for content in response:
            img_bytes = json.loads(content.text)[0]['bytes']
            # convert this img_bytes in str into base64
            img_bytes = base64.b64decode(img_bytes)
            st.session_state.workflow_img = img_bytes
            # st.image(img_bytes, caption="Graphviz Output", use_column_width=True)
            # # return img_bytes

def get_image():
    asyncio.run(get_workflow_image())
    print("Image fetched successfully")
    st.rerun()

# Initialize LLM
# lc_llm = AzureChatOpenAI(
#     model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
#     openai_api_key=os.environ["AZURE_OPENAI_KEY"],
#     openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
# )

WATSONX_APIKEY = os.getenv('WATSONX_APIKEY', "")
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID', "")

lc_llm = ChatWatsonx(
    model_id="ibm/granite-3-3-8b-instruct",#"mistralai/mistral-large",# "ibm/granite-3-8b-instruct",
    url = "https://us-south.ml.cloud.ibm.com",
    apikey = WATSONX_APIKEY,
    project_id = WATSONX_PROJECT_ID,
    params = {
        "decoding_method": "greedy",
        "temperature": 0,
        "min_new_tokens": 5,
        "max_new_tokens": 100000
    }
)

# Sidebar
with st.sidebar:
    st.title("Settings")

    model_size = st.selectbox(
        "Select Model Size",
        ["tiny", "base", "small"],
        index=0,
        help="Smaller models are faster but less accurate"
    )

    st.subheader("Recording Settings")
    chunk_duration = st.slider("Chunk Duration (seconds)", 1.0, 5.0, 3.0, 0.5)
    overlap = st.slider("Overlap (seconds)", 0.5, 2.5, 1.5, 0.5)

    st.subheader("Status")
    if st.session_state.get('is_recording', False):
        st.error("Recording in progress...")
    else:
        st.success("Ready to record")

# Initialize the Whisper model
@st.cache_resource
def load_model():
    try:
        return whisper.load_model("tiny")
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        st.info("Please make sure FFmpeg is installed on your system.")
        return None

# Manual message submission handler
def handle_manual_submit(message_text):
    if message_text.strip():
        timestamp = datetime.now().strftime("%H:%M:%S")
        human_msg = HumanMessage(
            content=message_text.strip(),
            additional_kwargs={
                "timestamp": timestamp,
                "source": "User (Manual)"
            }
        )
        if 'transcriptions' not in st.session_state:
            st.session_state.transcriptions = []
        st.session_state.transcriptions.append(human_msg)
        run_async_task()

# Async query processor
async def process_query():
    async with MultiServerMCPClient(
        {
            "Graphviz": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            
        }
    ) as client:
        
        tools = client.get_tools()
        agent = create_react_agent(lc_llm, tools)

        response = await agent.ainvoke({"messages": st.session_state.transcriptions})
        ai_message = response["messages"][-1]
        # print(ai_message)
        st.session_state.transcriptions.append(ai_message)

        return ai_message.content

# Sync wrapper
def run_async_task():
    return asyncio.run(process_query())

def main():
    # Initialize session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = [
            SystemMessage(
                content=(
                    """Welcome to the Graphviz Diagram Drawing Assistant!

                        I will guide you through the process of creating a diagram step by step. Here's how it works:
                        1. You will describe the diagram you want to create, including its purpose and key components.
                        2. I will propose a plan for the diagram, including the blocks/nodes, connections, and layout.
                        3. You will review and confirm the plan before we proceed.
                        4. Render after each step to show progress, always use the tool and never show the images in chat.
                        5. You are getting a live transcription of an audio stream, so take time and confirm before executing any tools
                        6. Focus on the information that is pertinent to the diagram, and avoid unnecessary details.
                        7. Users may also add a typed in message, use that to make necessary changes.

                        Let's get started! Please describe the diagram you want to create.
                    """
                )
            )
        ]
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'model' not in st.session_state or st.session_state.get('model_size') != model_size:
        st.session_state.model_size = model_size
        st.session_state.model = whisper.load_model(model_size)
    if 'last_transcription' not in st.session_state:
        st.session_state.last_transcription = ""
    if 'workflow_img' not in st.session_state:
        st.session_state.workflow_img = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("Real-time Audio Transcription")
        st.subheader("Workflow")
        if st.session_state.workflow_img is not None:
            st.image(st.session_state.workflow_img, caption="Workflow Diagram", use_column_width=True)
        # workflow_img = create_workflow_image()
        # st.image(workflow_img, use_column_width=True)

        # buf = io.BytesIO()
        # workflow_img.save(buf, format="PNG")
        # byte_im = buf.getvalue()

        # st.download_button(
        #     label=":arrow_down: Download Workflow Image",
        #     data=byte_im,
        #     file_name="workflow_diagram.png",
        #     mime="image/png"
        # )

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
        top_col1, top_col2 = st.columns([1, 1])
        with top_col1:
            show_transcript = st.checkbox("Show Transcript", value=True)
        with top_col2:
            download_button_placeholder = st.empty()

        if show_transcript:
            if st.session_state.get('transcriptions', []):
                with download_button_placeholder.container():
                    timestamp_dl = datetime.now().strftime("%Y%m%d_%H%M%S")
                    transcription_text = "\n".join([
                        f"[{msg.additional_kwargs.get('timestamp', '')}] {msg.additional_kwargs.get('source', 'User')}: {msg.content}"
                        for msg in st.session_state.transcriptions
                    ])
                    st.download_button(
                        label=":arrow_down: Download Transcript",
                        data=transcription_text,
                        file_name=f"transcription_{timestamp_dl}.txt",
                        mime="text/plain",
                        key="download_button"
                    )

            with st.form("manual_message_form", clear_on_submit=True):
                manual_message = st.text_area("Add Manual Message",
                                              placeholder="Type your message here...",
                                              height=80,
                                              key="manual_input")
                submit_manual = st.form_submit_button(":pencil2: Add Message")

                if submit_manual:
                    handle_manual_submit(st.session_state.manual_input)
                    get_image()

            # Chat-style display using st.chat_message
            for msg in st.session_state.get('transcriptions', [])[::-1]:
                if isinstance(msg, HumanMessage):
                    role = "user"
                    source = msg.additional_kwargs.get("source", "User")
                    timestamp = msg.additional_kwargs.get("timestamp", "")
                    with st.chat_message(role):
                        st.markdown(f"**{source}** — *{timestamp}*\n\n{msg.content}")
                if isinstance(msg, AIMessage):
                    role = "assistant"
                    source = msg.additional_kwargs.get("source", "Assistant")
                    timestamp = msg.additional_kwargs.get("timestamp", "")
                    with st.chat_message(role):
                        st.markdown(f"**{source}** — *{timestamp}*\n\n{msg.content}")
        else:
            download_button_placeholder.empty()

    # FFmpeg check
    if not check_ffmpeg():
        st.error("FFmpeg is not installed or not in PATH. Please install FFmpeg to use this application.")
        st.info("You can download FFmpeg from: https://ffmpeg.org/download.html")
        st.info("After installing, make sure to add FFmpeg to your system PATH.")
        st.info("Current PATH: " + os.environ.get('PATH', ''))
        return

    # Audio processing and transcription
    if st.session_state.is_recording:
        sample_rate = 16000
        chunk_size = 1024

        samples_per_chunk = int(chunk_duration * sample_rate / chunk_size)
        overlap_samples = int(overlap * sample_rate / chunk_size)

        audio_chunks = []
        start_time = time.time()

        while time.time() - start_time < chunk_duration and st.session_state.is_recording:
            chunk = st.session_state.recorder.get_audio_chunk()
            if chunk:
                audio_chunks.append(chunk)

        if audio_chunks:
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
                        new_text = result["text"]
                        if st.session_state.last_transcription:
                            overlap_text = st.session_state.last_transcription[-len(new_text)//2:]
                            if overlap_text in new_text:
                                new_text = new_text[new_text.find(overlap_text) + len(overlap_text):]

                        if new_text.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            transcribed_msg = HumanMessage(
                                content=new_text.strip(),
                                additional_kwargs={
                                    "timestamp": timestamp,
                                    "source": "User (Transcribed)"
                                }
                            )
                            st.session_state.transcriptions.append(transcribed_msg)
                            run_async_task() # To make a call the MCP
                            get_image() # To update the image
                            st.session_state.last_transcription = result["text"]
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
                finally:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)

        st.rerun()

if __name__ == "__main__":
    main()
