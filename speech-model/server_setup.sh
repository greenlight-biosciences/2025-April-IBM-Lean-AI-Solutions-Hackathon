mkdir ibmhackathon
cd ibmhackathon/
sudo   apt install python3.12-venv
python3 -m venv .venv
.venv/bin/activate
source .venv/bin/activate
pip install https://github.com/huggingface/transformers/archive/main.zip torchaudio peft soundfile fastapi pydantic uvicorn python-multipart
python
python3 test.py 
code app.py
python app.py 