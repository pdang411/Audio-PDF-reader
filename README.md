"# PIPER-Audiobook" 
Audio PDF reader combine with PIPER TTS. This docker build for Docker desktop,Linux or Ras PI linux .
Two container in one docker file using port 5000 for PIPER TTS and the second port 8501 Streamlit. 
PIPER TTS hev a web ui to download voice pakcage to access it your http://your-ip:5000/ui.
<img width="2503" height="1219" alt="piper tts ui" src="https://github.com/user-attachments/assets/086baf12-1a44-486d-a113-7eb8eb2b58b8" />
Streamlit access for Audio PDF reader http://your-ip:8501
<img width="2477" height="1266" alt="audio pdf reader" src="https://github.com/user-attachments/assets/c5aaffd0-2984-4e02-8fdc-63635ac93ae2" />

Docker set up cd into your folder : docker compose up -d 

Make sure your docker desk top is running and linux command apt install docker docker-compose.

PDF reader only. CPU setting designed for ARM and computer with no Nvidia GPU.
