import pyaudio
import threading
import queue
import subprocess
import tempfile
import wave
import os
from PIL import Image

def create_workflow_image():
    """Create a simple workflow visualization"""
    # Create a simple image with workflow steps
    img = Image.new('RGB', (600, 400), color='white')
    return img

def save_audio_chunk(audio_data, sample_rate=16000):
    """Save audio data to a temporary WAV file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            return temp_file.name
    except Exception as e:
        # st.error(f"Error saving audio: {str(e)}")
        print(f"Error saving audio: {str(e)}")
        return None

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        # Try to run ffmpeg -version
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.buffer = []

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.recording_thread:
                self.recording_thread.join()

    def _record(self):
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size)
                self.audio_queue.put(data)
            except Exception as e:
                # st.error(f"Error recording audio: {str(e)}")
                print(f"Error recording audio: {str(e)}")
                break

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def __del__(self):
        self.stop_recording()
        self.audio.terminate()