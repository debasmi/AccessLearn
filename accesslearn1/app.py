"""
AccessLearn — Inclusive AI Learning Platform
Flask backend that connects the HTML UI to all Python modules.
Run: python app.py
"""

import os
import sys
import time
import threading
import tempfile
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS

# ── Add modules folder to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

# ── Lazy imports (so app starts even if a lib is missing) ───────────────────
try:
    import pygame
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    PYGAME_OK = True
except Exception as e:
    print(f"⚠️  pygame not available: {e}")
    PYGAME_OK = False

try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyCAmlqHeDK_95FYvLIgcdV-z6W8Xd8_yak"))
    GEMINI_OK = True
except Exception as e:
    print(f"⚠️  Gemini not available: {e}")
    GEMINI_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except Exception as e:
    print(f"⚠️  gTTS not available: {e}")
    GTTS_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception as e:
    print(f"⚠️  pydub not available: {e}")
    PYDUB_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except Exception as e:
    print(f"⚠️  SpeechRecognition not available: {e}")
    SR_OK = False

try:
    import cv2
    import mediapipe as mp
    CV2_OK = True
except Exception as e:
    print(f"⚠️  OpenCV/MediaPipe not available: {e}")
    CV2_OK = False

from modules.braille import text_to_braille
from modules.audio_player import AudioPlayer
from modules.tts_engine import speak_text_to_file
from modules.sign_detector import detect_sign_from_camera

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Global state ─────────────────────────────────────────────────────────────
STATE = {
    "audio_file": None,
    "is_playing": False,
    "paused": False,
    "position_ms": 0,
    "duration_ms": 0,
    "playback_start": 0,
    "total_paused": 0,
    "pause_start": 0,
    "player": None,
    "audio_thread": None,
    "voice_thread": None,
    "voice_active": False,
    "last_command": "",
    "session_questions": 0,
    "session_answered": 0,
    "braille_files": [],
    "tts_speed": 1.5,
}

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Audio
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/audio/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    save_path = OUTPUTS_DIR / f.filename
    f.save(str(save_path))
    STATE["audio_file"] = str(save_path)

    if PYDUB_OK:
        try:
            audio = AudioSegment.from_file(str(save_path))
            STATE["duration_ms"] = len(audio)
        except:
            STATE["duration_ms"] = 0

    return jsonify({
        "success": True,
        "filename": f.filename,
        "duration_ms": STATE["duration_ms"],
        "path": str(save_path)
    })


@app.route("/api/audio/play", methods=["POST"])
def audio_play():
    if not STATE["audio_file"]:
        return jsonify({"error": "No audio file loaded. Please upload one first."}), 400
    if not PYDUB_OK or not PYGAME_OK:
        return jsonify({"error": "pygame/pydub not installed"}), 500

    STATE["paused"] = False
    STATE["is_playing"] = True

    def _play():
        player = AudioPlayer(STATE["audio_file"])
        STATE["player"] = player
        STATE["playback_start"] = time.time()
        player.play_from_position(int(STATE["position_ms"]))
        while player.is_playing():
            if STATE["paused"]:
                elapsed = (time.time() - STATE["playback_start"]) * 1000
                STATE["position_ms"] = elapsed - STATE["total_paused"]
                player.stop()
                STATE["pause_start"] = time.time()
                while STATE["paused"] and STATE["is_playing"]:
                    time.sleep(0.1)
                if STATE["is_playing"]:
                    pause_dur = (time.time() - STATE["pause_start"]) * 1000
                    STATE["total_paused"] += pause_dur
                    player.play_from_position(int(STATE["position_ms"]))
            elapsed = (time.time() - STATE["playback_start"]) * 1000
            STATE["position_ms"] = min(elapsed - STATE["total_paused"], STATE["duration_ms"])
            time.sleep(0.2)
        STATE["is_playing"] = False

    t = threading.Thread(target=_play, daemon=True)
    STATE["audio_thread"] = t
    t.start()
    return jsonify({"success": True, "status": "playing"})


@app.route("/api/audio/pause", methods=["POST"])
def audio_pause():
    STATE["paused"] = True
    return jsonify({"success": True, "position_ms": STATE["position_ms"]})


@app.route("/api/audio/resume", methods=["POST"])
def audio_resume():
    STATE["paused"] = False
    return jsonify({"success": True})


@app.route("/api/audio/stop", methods=["POST"])
def audio_stop():
    STATE["is_playing"] = False
    STATE["paused"] = False
    STATE["position_ms"] = 0
    STATE["total_paused"] = 0
    if STATE.get("player"):
        STATE["player"].stop()
    return jsonify({"success": True})


@app.route("/api/audio/seek", methods=["POST"])
def audio_seek():
    data = request.json or {}
    pct = float(data.get("percent", 0))
    if STATE["duration_ms"]:
        STATE["position_ms"] = STATE["duration_ms"] * pct / 100
    return jsonify({"success": True, "position_ms": STATE["position_ms"]})


@app.route("/api/audio/status")
def audio_status():
    elapsed = 0
    if STATE["is_playing"] and not STATE["paused"] and STATE["playback_start"]:
        elapsed = (time.time() - STATE["playback_start"]) * 1000 - STATE["total_paused"]
        STATE["position_ms"] = min(elapsed, STATE["duration_ms"])
    return jsonify({
        "is_playing": STATE["is_playing"],
        "paused": STATE["paused"],
        "position_ms": STATE["position_ms"],
        "duration_ms": STATE["duration_ms"],
        "audio_file": os.path.basename(STATE["audio_file"]) if STATE["audio_file"] else None,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Gemini Q&A
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "blind")       # blind | partial | deaf | mute
    tts = data.get("tts", True)
    braille = data.get("braille", True)
    speed = float(data.get("speed", STATE["tts_speed"]))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    STATE["session_questions"] += 1

    # ── Gemini answer ────────────────────────────────────────────────────────
    answer = _ask_gemini(question)
    STATE["session_answered"] += 1

    result = {
        "question": question,
        "answer": answer,
        "mode": mode,
        "braille_file": None,
        "audio_file": None,
    }

    # ── Braille (blind / partial modes) ─────────────────────────────────────
    if braille and mode in ("blind", "partial"):
        braille_text = text_to_braille(answer)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"gemini_answer_{ts}_braille.txt"
        fpath = OUTPUTS_DIR / fname
        fpath.write_text(braille_text, encoding="utf-8")
        STATE["braille_files"].insert(0, fname)
        result["braille"] = braille_text
        result["braille_file"] = fname

    # ── TTS audio (blind / partial / mute modes) ─────────────────────────────
    if tts and GTTS_OK and PYDUB_OK and mode in ("blind", "partial", "mute"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        wav_name = f"answer_{ts}.wav"
        wav_path = OUTPUTS_DIR / wav_name
        ok = speak_text_to_file(answer, str(wav_path), speed=speed)
        if ok:
            result["audio_file"] = wav_name

    return jsonify(result)


def _ask_gemini(question: str) -> str:
    if not GEMINI_OK:
        return "(Gemini not configured — check GEMINI_API_KEY)"
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Voice recognition (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

voice_events = []   # ring buffer of events the frontend polls

@app.route("/api/voice/start", methods=["POST"])
def voice_start():
    if not SR_OK:
        return jsonify({"error": "SpeechRecognition not installed"}), 500
    if STATE["voice_active"]:
        return jsonify({"ok": True, "msg": "already running"})
    STATE["voice_active"] = True
    t = threading.Thread(target=_voice_loop, daemon=True)
    STATE["voice_thread"] = t
    t.start()
    return jsonify({"ok": True})


@app.route("/api/voice/stop", methods=["POST"])
def voice_stop():
    STATE["voice_active"] = False
    return jsonify({"ok": True})


@app.route("/api/voice/events")
def voice_events_poll():
    events = voice_events.copy()
    voice_events.clear()
    return jsonify({"events": events})


def _voice_loop():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        _voice_event("calibrated", "Microphone ready. Say 'pause', 'ask', 'play', or 'quit'.")
    except Exception as e:
        _voice_event("error", f"Microphone error: {e}")
        STATE["voice_active"] = False
        return

    while STATE["voice_active"]:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            command = recognizer.recognize_google(audio).lower().strip()
            _voice_event("command", command)
            STATE["last_command"] = command

            if any(w in command for w in ["pause", "question", "ask"]) and not STATE["paused"]:
                STATE["paused"] = True
                _voice_event("paused", "Paused. Ask your question now (15 seconds).")
                try:
                    with mic as qs:
                        qa = recognizer.listen(qs, timeout=15, phrase_time_limit=10)
                    question = recognizer.recognize_google(qa).strip()
                    _voice_event("question", question)
                    answer = _ask_gemini(question)
                    STATE["session_questions"] += 1
                    STATE["session_answered"] += 1
                    braille_text = text_to_braille(answer)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"gemini_answer_{ts}_braille.txt"
                    (OUTPUTS_DIR / fname).write_text(braille_text, encoding="utf-8")
                    STATE["braille_files"].insert(0, fname)
                    if GTTS_OK and PYDUB_OK:
                        wav_name = f"answer_{ts}.wav"
                        speak_text_to_file(answer, str(OUTPUTS_DIR / wav_name), speed=STATE["tts_speed"])
                        _voice_event("answer", json.dumps({
                            "question": question,
                            "answer": answer,
                            "braille": braille_text,
                            "braille_file": fname,
                            "audio_file": wav_name,
                        }))
                    else:
                        _voice_event("answer", json.dumps({
                            "question": question, "answer": answer,
                            "braille": braille_text, "braille_file": fname,
                        }))
                    time.sleep(2)
                    STATE["paused"] = False
                    _voice_event("resumed", "Resuming lecture.")
                except sr.WaitTimeoutError:
                    STATE["paused"] = False
                    _voice_event("resumed", "No question heard. Resuming.")
                except sr.UnknownValueError:
                    STATE["paused"] = False
                    _voice_event("resumed", "Could not understand. Resuming.")

            elif any(w in command for w in ["play", "resume", "continue"]):
                STATE["paused"] = False
                _voice_event("resumed", "Resuming lecture.")

            elif any(w in command for w in ["quit", "exit", "stop"]):
                STATE["voice_active"] = False
                _voice_event("stopped", "Voice control stopped.")
                break

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            _voice_event("error", str(e))
            time.sleep(1)


def _voice_event(kind: str, data: str):
    voice_events.append({"type": kind, "data": data, "t": time.time()})
    if len(voice_events) > 50:
        voice_events.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Camera / Sign detection
# ─────────────────────────────────────────────────────────────────────────────

sign_result = {"status": "idle", "detected": "", "frame_b64": ""}

@app.route("/api/sign/start", methods=["POST"])
def sign_start():
    data = request.json or {}
    duration = int(data.get("duration", 10))
    if not CV2_OK:
        return jsonify({"error": "OpenCV/MediaPipe not installed"}), 500
    threading.Thread(target=_sign_detect_thread, args=(duration,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/sign/status")
def sign_status():
    return jsonify(sign_result)


def _sign_detect_thread(duration: int):
    sign_result["status"] = "detecting"
    sign_result["detected"] = ""
    detected = detect_sign_from_camera(duration_sec=duration)
    sign_result["detected"] = detected
    sign_result["status"] = "done"

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — Settings & state
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["POST"])
def settings():
    data = request.json or {}
    if "tts_speed" in data:
        STATE["tts_speed"] = float(data["tts_speed"])
    return jsonify({"ok": True})


@app.route("/api/state")
def state():
    return jsonify({
        "session_questions": STATE["session_questions"],
        "session_answered": STATE["session_answered"],
        "braille_files": STATE["braille_files"][:5],
        "voice_active": STATE["voice_active"],
        "is_playing": STATE["is_playing"],
        "paused": STATE["paused"],
        "tts_speed": STATE["tts_speed"],
        "capabilities": {
            "pygame": PYGAME_OK,
            "gemini": GEMINI_OK,
            "gtts": GTTS_OK,
            "pydub": PYDUB_OK,
            "speech_recognition": SR_OK,
            "opencv": CV2_OK,
        }
    })

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎓  AccessLearn — Inclusive AI Learning Platform")
    print("="*55)
    print(f"  pygame          : {'✅' if PYGAME_OK else '❌ pip install pygame'}")
    print(f"  gemini          : {'✅' if GEMINI_OK else '❌ pip install google-generativeai'}")
    print(f"  gTTS            : {'✅' if GTTS_OK else '❌ pip install gtts'}")
    print(f"  pydub           : {'✅' if PYDUB_OK else '❌ pip install pydub'}")
    print(f"  SpeechRecog.    : {'✅' if SR_OK else '❌ pip install SpeechRecognition'}")
    print(f"  OpenCV/MediaPipe: {'✅' if CV2_OK else '❌ pip install opencv-python mediapipe'}")
    print("="*55)
    print("  Open http://localhost:5000 in your browser")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
