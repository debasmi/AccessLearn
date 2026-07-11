import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import os
from gtts import gTTS
import google.generativeai as genai
from pydub import AudioSegment
import tempfile

# ---------------- CONFIG ----------------
genai.configure(api_key="AIzaSyCAmlqHeDK_95FYvLIgcdV-z6W8Xd8_yak")
AUDIO_FILE = "/Users/debasmibasu/Documents/SIS/blind user/lecture1audio.wav"
SIGN_WAIT_TIME = 10  # seconds to wait for sign language question

# ---------------- INITIALIZE ----------------
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

stop_audio = False
paused = False

# ---------------- AUDIO PLAYER ----------------
class AudioPlayer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.full_audio = None
        self.load_audio()

    def load_audio(self):
        try:
            self.full_audio = AudioSegment.from_wav(self.audio_file)
            print(f"🎵 Audio loaded: {len(self.full_audio)/1000:.1f} seconds")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
        return True

    def play_from_position(self, start_ms=0):
        if not self.full_audio:
            return False
        remaining_audio = self.full_audio[start_ms:]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        remaining_audio.export(temp_file.name, format="wav")
        try:
            sound = pygame.mixer.Sound(temp_file.name)
            pygame.mixer.Sound.play(sound)
            def cleanup():
                time.sleep(0.5)
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()
            return True
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            return False

    def stop(self):
        pygame.mixer.stop()

    def is_playing(self):
        return pygame.mixer.get_busy()

def play_audio_continuous():
    """Play lecture audio continuously in background."""
    global stop_audio
    player = AudioPlayer(AUDIO_FILE)
    if not player.full_audio:
        print("❌ Failed to load audio")
        return
    current_pos = 0
    while not stop_audio:
        if not player.is_playing():
            player.play_from_position(int(current_pos))
        time.sleep(0.5)
        current_pos += 500
        if current_pos >= len(player.full_audio):
            break

# ---------------- GEMINI & TTS ----------------
def ask_gemini(question: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "Sorry, could not process your question."

def speak_text(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_mp3_path = tmp_file.name
        tts = gTTS(text=text, lang="en")
        tts.save(temp_mp3_path)
        temp_wav_path = temp_mp3_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(temp_mp3_path)
        audio.export(temp_wav_path, format="wav")
        sound = pygame.mixer.Sound(temp_wav_path)
        pygame.mixer.Sound.play(sound)
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        os.remove(temp_mp3_path)
        os.remove(temp_wav_path)
    except Exception as e:
        print(f"❌ TTS error: {e}")

# ---------------- SIGN LANGUAGE DISPLAY ----------------
def show_sign_language(text: str):
    """Display answer in sign language (one image per letter)."""
    for char in text.upper():
        if char.isalpha():
            img_path = f"./sign_alphabet/{char}.png"
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                cv2.imshow("Sign Language", img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

def handle_question(question: str):
    """Get Gemini answer and display in text, audio, and sign language."""
    if not question:
        question = "No question asked"
    print(f"❓ Question: {question}")
    answer = ask_gemini(question)
    print(f"💡 Answer: {answer}")
    speak_text(answer)
    show_sign_language(answer)

# ---------------- SIGN LANGUAGE QUESTION ----------------
def ask_sign_language_question():
    """Pause audio, open camera, wait for a question 10 sec, resume if none."""
    global paused
    paused = True
    pygame.mixer.pause()
    cap = cv2.VideoCapture(0)
    question_detected = False
    start_time = time.time()

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while time.time() - start_time < SIGN_WAIT_TIME:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if results.multi_hand_landmarks:
                print("✋ Hand detected!")
                question = "HELLO"  # Placeholder for sign language recognition
                handle_question(question)
                question_detected = True
                break

    cap.release()
    cv2.destroyAllWindows()
    paused = False
    pygame.mixer.unpause()
    if not question_detected:
        print("⏰ No question detected. Resuming lecture.")
        handle_question("")  # Respond "No question asked"

# ---------------- TEXT INPUT ----------------
def text_input_mode():
    global stop_audio
    while not stop_audio:
        question = input("❓ Your question: ").strip()
        if question.lower() in ["exit", "quit", "stop"]:
            stop_audio = True
            break
        handle_question(question)

# ---------------- MAIN ----------------
def main():
    global stop_audio
    # Start lecture audio immediately
    audio_thread = threading.Thread(target=play_audio_continuous, daemon=True)
    audio_thread.start()
    time.sleep(1)  # slight delay to ensure audio starts

    # Ask mode after audio has started
    print("🎵 Lecture started.")
    print("Choose input mode:")
    print("1️⃣ Sign Language Mode (camera)")
    print("2️⃣ Text Input Mode (keyboard)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "2":
        text_input_mode()
    else:
        # Stop audio and open camera for 10 seconds
        ask_sign_language_question()
        # Resume audio automatically

    stop_audio = True
    pygame.mixer.quit()
    print("🔚 Application terminated")

if __name__ == "__main__":
    main()