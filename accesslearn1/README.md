# AccessLearn — Inclusive AI Learning Platform

A unified web application that runs your four accessibility modules
through a single browser interface on your laptop.

---

## 📁 Project Structure

```
accesslearn/
├── app.py                  ← Flask server (run this)
├── requirements.txt        ← All Python dependencies
├── run.sh                  ← macOS/Linux one-click start
├── run.bat                 ← Windows one-click start
├── modules/
│   ├── braille.py          ← Grade-1 Braille converter
│   ├── audio_player.py     ← pygame audio playback
│   ├── tts_engine.py       ← gTTS text-to-speech
│   └── sign_detector.py    ← MediaPipe hand detection
├── templates/
│   └── index.html          ← The full UI
└── outputs/                ← Generated braille files + TTS audio
```

---

## 🚀 Quick Start

### macOS / Linux
```bash
cd accesslearn
bash run.sh
```

### Windows
Double-click `run.bat`

### Manual (any OS)
```bash
cd accesslearn
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## ⚙️ System Requirements

| Requirement | Notes |
|------------|-------|
| Python 3.9+ | https://python.org |
| ffmpeg | Required by pydub for MP3 support |
| Microphone | For voice command mode |
| Webcam | For sign language detection |
| Internet | For Gemini API + gTTS |

**Install ffmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: https://ffmpeg.org/download.html → add to PATH

---

## 🔑 Gemini API Key

The key is pre-configured. To change it, either:
1. Set environment variable: `export GEMINI_API_KEY=your_key`
2. Or edit line in `app.py`: `api_key=os.environ.get("GEMINI_API_KEY", "your_key")`

---

## 🧩 The Four Modes

### 🦯 Blind User Mode
- Upload a WAV/MP3 lecture → click Play
- Click **Start Voice** → say `pause` to ask a question verbally
- Gemini answers are **spoken aloud** via TTS and displayed in **Braille**
- Braille files auto-saved to `outputs/` folder
- Adjust TTS speed with the slider

### 👁️ Partially Sighted Mode
- Same audio playback with larger controls
- Answers displayed in **large, high-contrast text** (20px+)
- Optional: open camera for sign language detection

### 🤟 Deaf User Mode
- Click **Start (10s)** to open webcam
- Show a hand sign → MediaPipe detects it and populates question box
- All answers shown as **text only** (no audio played)
- Live lecture captions display

### 🔇 Mute User Mode
- Audio lecture plays normally
- Type questions via keyboard
- Answers shown in **text + animated sign language alphabet grid**
- TTS audio generated for caregivers/teachers to hear the answer

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'mediapipe'` | `pip install mediapipe` |
| `No module named 'pygame'` | `pip install pygame` |
| TTS fails | `pip install gtts pydub` + install ffmpeg |
| Microphone not found | Check system mic permissions |
| Camera not found | Check system camera permissions |
| Gemini error | Check API key / internet connection |
| `Port 5000 in use` | `kill $(lsof -ti:5000)` or change port in app.py |

---

## 📌 Notes

- The header shows live **capability badges** (✓ green / ✗ red) for each module
- Braille files are saved as `outputs/gemini_answer_YYYYMMDD_HHMMSS_braille.txt`
- TTS audio files are saved as `outputs/answer_YYYYMMDD_HHMMSS.wav` and auto-deleted after playback is optional
- Voice recognition uses Google Speech API (requires internet)
