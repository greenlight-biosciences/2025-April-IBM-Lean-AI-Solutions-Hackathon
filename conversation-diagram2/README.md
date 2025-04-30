# Real-time Audio Transcription App

This is a Streamlit application that allows you to record audio from your microphone and transcribe it in real-time using OpenAI's Whisper model.

## Prerequisites

1. Install FFmpeg:
   - Download FFmpeg from: https://ffmpeg.org/download.html
   - Extract the downloaded file
   - Add the FFmpeg `bin` directory to your system PATH
   - To verify installation, open a new terminal and run: `ffmpeg -version`

## Setup

1. Install `uv` (if not already installed):
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
.venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
uv pip install -r requirements.txt
```

## Running the App

To run the application, use the following command:
```bash
streamlit run app.py
```

## Features

- Record audio from your microphone
- Real-time transcription using OpenAI's Whisper model
- Download transcriptions as text files
- Simple and intuitive user interface

## Requirements

- Python 3.8 or higher
- FFmpeg installed and added to system PATH
- Microphone access
- Internet connection (for downloading the Whisper model)

## Troubleshooting

If you encounter the error "The system cannot find the file specified":
1. Make sure FFmpeg is installed correctly
2. Verify that FFmpeg is in your system PATH
3. Try opening a new terminal and running `ffmpeg -version` to confirm installation

## Note

The first time you run the app, it will download the Whisper model, which might take a few minutes depending on your internet connection. 