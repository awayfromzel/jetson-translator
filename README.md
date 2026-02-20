# Jetson Translator

An offline, hardware-based speech translation device built on the NVIDIA Jetson Orin Nano.

The system performs speech-to-text (Whisper), machine translation (NLLB-200),
and text-to-speech (Piper) in real time using physical buttons, a rotary encoder,
and an LCD interface.

## Architecture

The system is split into two components:

- Main application (hardware control, ASR, MT)
- TTS microservice (Piper running in a separate environment)

The TTS engine runs as a FastAPI service to isolate dependencies
and avoid conflicts with the main ML stack.

The two components communicate over HTTP (localhost) to maintain environment isolation.

## Processing Pipeline

1. User holds button → recording starts
2. Audio is transcribed using Whisper
3. Text is translated with NLLB-200
4. Translated text is sent to TTS service
5. Audio is synthesized and played
6. LCD displays translated output

## Hardware Components

- NVIDIA Jetson Orin Nano
- USB microphone
- USB speaker
- I2C LCD (16x2)
- Rotary encoder (language selection)
- Two push-buttons (A→B / B→A translation)

## Repository Structure
```
project_root/
├── src/ # Main application logic
├── tts_service/ # FastAPI-based Piper TTS microservice
│ ├── serve.py
│ ├── requirements.txt
│ └── venv/
├── requirements.txt # Main translator environment
└── README.md
```
## How to run
Both services must be running concurrently. The TTS service listens on localhost:5005.

### Start the TTS Service
```bash
cd tts_service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd ..
PIPER_VOICE_DIR=./tts_service/piper_voices \
PIPER_USE_CUDA=0 \
uvicorn tts_service.serve:app --host 127.0.0.1 --port 5005
```

### Start the Main Application (in a separate terminal)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd src
python main.py
```

## Design Decisions

### TTS as a Microservice
Piper dependencies conflict with the main ML stack. Running TTS
in a separate FastAPI service isolates environments and improves reliability.

### Preloading Voices at Startup
All voices are loaded during service startup to eliminate first-request latency.

### Poll-Based GPIO Handling
Buttons use software debouncing and long-press detection
to ensure stable input handling without hardware interrupts.

### Warm-Up Initialization
ASR and MT models are warmed up at startup to avoid slow first inference.









