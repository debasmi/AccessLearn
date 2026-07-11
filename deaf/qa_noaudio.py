
import pygame
import speech_recognition as sr
import threading
import time
import os
from gtts import gTTS
import google.generativeai as genai
from pydub import AudioSegment
import tempfile

# 🔑 Configure Gemini API
genai.configure(api_key="AIzaSyCAmlqHeDK_95FYvLIgcdV-z6W8Xd8_yak")

AUDIO_FILE = "/Users/debasmibasu/Documents/SIS/blind user/lecture1audio.wav"

# Initialize pygame mixer
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Global state
is_playing = False
paused = False
stop_audio = False
pause_start_time = 0
total_paused_time = 0
playback_start_time = 0


class AudioPlayer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.full_audio = None
        self.current_position = 0  # in milliseconds
        self.load_audio()
        
    def load_audio(self):
        """Load the full audio file using pydub"""
        try:
            self.full_audio = AudioSegment.from_wav(self.audio_file)
            print(f"🎵 Audio loaded: {len(self.full_audio) / 1000:.1f} seconds")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
        return True
    
    def play_from_position(self, start_ms=0):
        """Play audio from a specific position in milliseconds"""
        if not self.full_audio:
            return False
            
        # Create audio segment from current position to end
        remaining_audio = self.full_audio[start_ms:]
        
        # Export to temporary WAV file for pygame
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        remaining_audio.export(temp_file.name, format="wav")
        
        # Load and play with pygame
        try:
            sound = pygame.mixer.Sound(temp_file.name)
            pygame.mixer.Sound.play(sound)
            
            # Clean up temp file after a short delay
            def cleanup():
                time.sleep(0.5)  # Wait for pygame to load the file
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
        """Stop the audio"""
        pygame.mixer.stop()
    
    def is_playing(self):
        """Check if audio is currently playing"""
        return pygame.mixer.get_busy()
    
    def get_duration_ms(self):
        """Get total audio duration in milliseconds"""
        return len(self.full_audio) if self.full_audio else 0


def play_audio():
    """
    Plays the audio file with proper pause/resume from timestamp functionality.
    """
    global is_playing, paused, stop_audio, pause_start_time, total_paused_time, playback_start_time
    
    player = AudioPlayer(AUDIO_FILE)
    if not player.full_audio:
        print("❌ Failed to load audio file")
        return
    
    current_position = 0  # Current playback position in milliseconds
    total_paused_time = 0
    
    print(f"🎵 Starting audio playback from beginning...")
    playback_start_time = time.time()
    
    # Start playing from the beginning
    if not player.play_from_position(current_position):
        return
    
    is_playing = True
    
    while True:
        if stop_audio:
            player.stop()
            break
        
        if paused and player.is_playing():
            # Calculate current position when pausing
            elapsed_real_time = (time.time() - playback_start_time) * 1000  # Convert to ms
            current_position = elapsed_real_time - total_paused_time
            
            player.stop()
            pause_start_time = time.time()
            
            print(f"🔇 Audio paused at {current_position/1000:.1f} seconds")
            
            # Wait while paused
            while paused and not stop_audio:
                time.sleep(0.1)
            
            if not stop_audio:
                # Calculate how long we were paused
                pause_duration = (time.time() - pause_start_time) * 1000
                total_paused_time += pause_duration
                
                print(f"▶️ Resuming from {current_position/1000:.1f} seconds...")
                
                # Resume from the calculated position
                if current_position < player.get_duration_ms():
                    player.play_from_position(int(current_position))
                else:
                    print("✅ Audio completed")
                    is_playing = False
                    break
        
        # Check if audio finished naturally
        if not player.is_playing() and not paused and is_playing:
            elapsed_real_time = (time.time() - playback_start_time) * 1000
            actual_position = elapsed_real_time - total_paused_time
            
            if actual_position >= player.get_duration_ms() - 1000:  # Within 1 second of end
                print("✅ Audio playback completed")
                is_playing = False
                break
        
        time.sleep(0.1)


def ask_gemini(question: str) -> str:
    """
    Sends question to Gemini API and returns the response text.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"❌ Error with Gemini API: {e}")
        return "Sorry, I couldn't process your question right now."


def speak_text(text: str, answer_speed: float = 1.5):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_mp3_path = tmp_file.name

        # Generate TTS audio
        tts = gTTS(text=text, lang="en")
        tts.save(temp_mp3_path)

        # Convert to wav for pygame
        temp_wav_path = temp_mp3_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(temp_mp3_path)

        # ✅ Apply playback speed only to Gemini answers
        if answer_speed != 1.0:
            audio = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * answer_speed)}
            ).set_frame_rate(audio.frame_rate)

        audio.export(temp_wav_path, format="wav")

        # Play the sped-up audio
        tts_sound = pygame.mixer.Sound(temp_wav_path)
        pygame.mixer.Sound.play(tts_sound)

        while pygame.mixer.get_busy():
            time.sleep(0.1)

        # Cleanup
        try:
            os.remove(temp_mp3_path)
            os.remove(temp_wav_path)
        except:
            pass
    except Exception as e:
        print(f"❌ Error with text-to-speech: {e}")


"""def listen_for_commands():
    '''
    Continuously listens for voice commands.
    '''
    global paused, stop_audio, pause_start_time
    
    recognizer = sr.Recognizer()
    
    # Try to find the best microphone
    mic_list = sr.Microphone.list_microphone_names()
    print(f"🎤 Available microphones: {len(mic_list)}")
    
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return
    
    # Initial calibration
    try:
        with mic as source:
            print("🎤 Calibrating microphone... (please stay quiet)")
            recognizer.adjust_for_ambient_noise(source, duration=3)
            print("🎤 Voice control ready. Say 'pause' to ask questions, 'play' to resume.")
    except Exception as e:
        print(f"❌ Microphone calibration failed: {e}")
        return

    while not stop_audio:
        try:
            # Listen for command with shorter timeout
            with mic as source:
                print("👂 Listening for command...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
            command = recognizer.recognize_google(audio).lower().strip()
            print(f"🗣️ Command heard: '{command}'")

            # More flexible pause detection
            if any(word in command for word in ["pause", "stop", "question", "ask"]) and not paused:
                print("⏸️ Pausing for Q&A mode...")
                paused = True
                
                # Listen for question
                print("🤔 What's your question? (You have 15 seconds)")
                try:
                    with mic as q_source:
                        question_audio = recognizer.listen(q_source, timeout=15, phrase_time_limit=10)
                        question = recognizer.recognize_google(question_audio).strip()
                        print(f"❓ Question: {question}")

                    # Get answer from Gemini
                    '''print("🤖 Getting answer from Gemini...")
                    answer = ask_gemini(question)
                    print(f"💡 Answer: {answer}")

                    # Speak answer
                    speak_text(answer)'''
                    # Get answer from Gemini
                    print("🤖 Getting answer from Gemini...")
                    answer = ask_gemini(question)
                    print(f"💡 Answer: {answer}")

                    # Convert Gemini answer to Braille
                    braille_output = text_to_braille(answer)
                    print("\n⠿ Braille representation:")
                    print(braille_output)
                    print("")
                    
                    with open("gemini_answer_braille.txt", "w", encoding="utf-8") as f:
                        f.write(braille_output)

                    # Speak answer
                    speak_text(answer)


                    # Auto-resume after answer
                    print("▶️ Auto-resuming playback in 2 seconds...")
                    time.sleep(2)
                    paused = False

                except sr.WaitTimeoutError:
                    print("⏰ No question heard, resuming playback...")
                    paused = False
                except sr.UnknownValueError:
                    print("❌ Could not understand the question, resuming playback...")
                    paused = False

            elif any(word in command for word in ["play", "resume", "continue"]) and paused:
                print("▶️ Resuming playback...")
                paused = False
                
            elif any(word in command for word in ["quit", "exit", "stop program"]):
                print("👋 Stopping application...")
                stop_audio = True
                break

        except sr.WaitTimeoutError:
            # Timeout is normal, continue listening
            continue
        except sr.UnknownValueError:
            # Could not understand audio, continue
            continue
        except sr.RequestError as e:
            print(f"❌ Speech Recognition service error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Unexpected error in voice recognition: {e}")
            time.sleep(1)"""


'''def main():
    """Main function to start the application"""
    global stop_audio
    
    print("🎵 Starting Interactive Audio Q&A Application")
    print("=" * 50)
    print("Commands:")
    print("- Say 'pause' to stop audio and ask a question")
    print("- Say 'play' or 'resume' to continue from where you paused")
    print("- Say 'quit' to exit the application")
    print("=" * 50)
    
    try:
        # Start audio playback in a separate thread
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()
        
        # Start voice command listening (main thread)
        listen_for_commands()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        stop_audio = True
        pygame.mixer.quit()
        print("🔚 Application terminated")



'''
'''#larger text output
def handle_question(question: str):
    """Process question: get Gemini answer, convert to Braille, save, and speak."""
    print("\n" + "=" * 80)
    print(f"🧠 QUESTION: {question.upper()}")
    print("=" * 80)

    # Get answer from Gemini
    print("🤖 Getting answer from Gemini...")
    answer = ask_gemini(question)

    # Create bold, large-looking output
    def enlarge_text(text):
        return "\n".join(["".join([char.upper() + " " for char in line]) for line in text.splitlines()])

    print("\n💡 ANSWER (ENLARGED):")
    print("=" * 60)
    print(enlarge_text(answer))
    print("=" * 60)

    # Convert Gemini answer to Braille
    braille_output = text_to_braille(answer)
    print("\n⠿ BRAILLE REPRESENTATION (ENLARGED):")
    print("=" * 60)
    print(enlarge_text(braille_output))
    print("=" * 60)

    # Save Braille to timestamped file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_answer_{timestamp}.braille.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(braille_output)
    print(f"💾 Braille saved as {filename}")

    # Speak answer aloud
    speak_text(answer)

    print("▶️ Auto-resuming playback in 2 seconds...")
    time.sleep(2)
    return'''

def handle_question(question: str):
    """Process question: get Gemini answer, convert to Braille, save, and speak."""
    print(f"🤖 Getting answer from Gemini...")
    answer = ask_gemini(question)
    print(f"💡 Answer: {answer}")

    
    # Auto-resume playback if applicable
    print("▶️ Auto-resuming playback in 2 seconds...")
    time.sleep(2)
    return


def listen_for_commands():
    """
    Continuously listens for voice commands.
    """
    global paused, stop_audio, pause_start_time
    
    recognizer = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    print(f"🎤 Available microphones: {len(mic_list)}")

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return

    # Initial calibration
    try:
        with mic as source:
            print("🎤 Calibrating microphone... (please stay quiet)")
            recognizer.adjust_for_ambient_noise(source, duration=3)
            print("🎤 Voice control ready. Say 'pause' to ask questions, 'play' to resume.")
    except Exception as e:
        print(f"❌ Microphone calibration failed: {e}")
        return

    while not stop_audio:
        try:
            with mic as source:
                print("👂 Listening for command...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower().strip()
            print(f"🗣️ Command heard: '{command}'")

            if any(word in command for word in ["pause", "stop", "question", "ask"]) and not paused:
                print("⏸️ Pausing for Q&A mode...")
                paused = True

                # Listen for question
                print("🤔 What's your question? (You have 15 seconds)")
                try:
                    with mic as q_source:
                        question_audio = recognizer.listen(q_source, timeout=15, phrase_time_limit=10)
                        question = recognizer.recognize_google(question_audio).strip()
                        print(f"❓ Question: {question}")
                    handle_question(question)
                    paused = False

                except sr.WaitTimeoutError:
                    print("⏰ No question heard, resuming playback...")
                    paused = False
                except sr.UnknownValueError:
                    print("❌ Could not understand the question, resuming playback...")
                    paused = False

            elif any(word in command for word in ["play", "resume", "continue"]) and paused:
                print("▶️ Resuming playback...")
                paused = False

            elif any(word in command for word in ["quit", "exit", "stop program"]):
                print("👋 Stopping application...")
                stop_audio = True
                break

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"❌ Speech Recognition service error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Unexpected error in voice recognition: {e}")
            time.sleep(1)


def text_input_mode():
    """Allows user to type questions instead of speaking."""
    global stop_audio
    print("\n⌨️ Text Input Mode — type your questions below.")
    print("Type 'exit' to quit.\n")

    while not stop_audio:
        question = input("❓ Your question: ").strip()
        if question.lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting text input mode...")
            stop_audio = True
            break
        elif question:
            handle_question(question)


def main():
    """Main function to start the application"""
    global stop_audio

    print("🎵 Starting Interactive Audio Q&A Application")
    print("=" * 50)
    print("Choose mode:")
    print("1️⃣  Sign language input")
    print("2️⃣  Text Input Mode (type your questions)")
    print("=" * 50)
    
    mode = input("Enter 1 or 2: ").strip()

    try:
        # Start audio playback in a separate thread
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

        if mode == "2":
            text_input_mode()
        else:
            listen_for_commands()

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        stop_audio = True
        pygame.mixer.quit()
        print("🔚 Application terminated")
if __name__ == "__main__":
    main()