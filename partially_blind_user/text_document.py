import pygame
import speech_recognition as sr
import threading
import time
import os
from gtts import gTTS
import google.generativeai as genai
from pydub import AudioSegment
import tempfile
from braille import text_to_braille

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


# ============================================================================
# FONT DISPLAY SETTINGS - Based on Accessibility Guidelines
# ============================================================================
# Using sans serif font recommendations from the document:
# - Arial, Verdana, Tahoma are recommended sans serif fonts
# - Minimum 11-12pt for standard print
# - 16pt+ for large print accessibility
# ============================================================================

def format_for_display(text: str, large_print: bool = False) -> str:
    """
    Format text with proper spacing and visual structure for accessibility.
    Based on font legibility guidelines for visually impaired users.
    
    Args:
        text: The text to format
        large_print: If True, adds extra spacing for large print format
    
    Returns:
        Formatted text string with appropriate spacing
    """
    if large_print:
        # For large print: add extra line spacing and character spacing
        # Simulates 16pt+ font with better readability
        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            # Add space between characters for better distinction
            spaced_line = ' '.join(line)
            formatted_lines.append(spaced_line)
            formatted_lines.append('')  # Extra line break
        return '\n'.join(formatted_lines)
    else:
        # Standard print format with consistent spacing
        return text


def print_accessible_header(text: str, large_print: bool = False):
    """Print a header with accessibility-friendly formatting"""
    separator = "=" * 80 if not large_print else "= " * 40
    print(f"\n{separator}")
    if large_print:
        print(format_for_display(text.upper(), large_print=True))
    else:
        print(f"  {text.upper()}")
    print(f"{separator}\n")


def print_accessible_section(label: str, content: str, large_print: bool = False):
    """Print a content section with clear visual separation"""
    print(f"\n{label}")
    print("-" * 80 if not large_print else "- " * 40)
    if large_print:
        print(format_for_display(content, large_print=True))
    else:
        print(content)
    print("-" * 80 if not large_print else "- " * 40)


# ============================================================================
# AUDIO PLAYER CLASS
# ============================================================================

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
        """Stop the audio"""
        pygame.mixer.stop()
    
    def is_playing(self):
        """Check if audio is currently playing"""
        return pygame.mixer.get_busy()
    
    def get_duration_ms(self):
        """Get total audio duration in milliseconds"""
        return len(self.full_audio) if self.full_audio else 0


# ============================================================================
# AUDIO PLAYBACK CONTROL
# ============================================================================

def play_audio():
    """Plays the audio file with proper pause/resume from timestamp functionality."""
    global is_playing, paused, stop_audio, pause_start_time, total_paused_time, playback_start_time
    
    player = AudioPlayer(AUDIO_FILE)
    if not player.full_audio:
        print("❌ Failed to load audio file")
        return
    
    current_position = 0
    total_paused_time = 0
    
    print(f"🎵 Starting audio playback from beginning...")
    playback_start_time = time.time()
    
    if not player.play_from_position(current_position):
        return
    
    is_playing = True
    
    while True:
        if stop_audio:
            player.stop()
            break
        
        if paused and player.is_playing():
            elapsed_real_time = (time.time() - playback_start_time) * 1000
            current_position = elapsed_real_time - total_paused_time
            
            player.stop()
            pause_start_time = time.time()
            
            print(f"🔇 Audio paused at {current_position/1000:.1f} seconds")
            
            while paused and not stop_audio:
                time.sleep(0.1)
            
            if not stop_audio:
                pause_duration = (time.time() - pause_start_time) * 1000
                total_paused_time += pause_duration
                
                print(f"▶️ Resuming from {current_position/1000:.1f} seconds...")
                
                if current_position < player.get_duration_ms():
                    player.play_from_position(int(current_position))
                else:
                    print("✅ Audio completed")
                    is_playing = False
                    break
        
        if not player.is_playing() and not paused and is_playing:
            elapsed_real_time = (time.time() - playback_start_time) * 1000
            actual_position = elapsed_real_time - total_paused_time
            
            if actual_position >= player.get_duration_ms() - 1000:
                print("✅ Audio playback completed")
                is_playing = False
                break
        
        time.sleep(0.1)


# ============================================================================
# GEMINI AI INTEGRATION
# ============================================================================

def ask_gemini(question: str) -> str:
    """Sends question to Gemini API and returns the response text."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"❌ Error with Gemini API: {e}")
        return "Sorry, I couldn't process your question right now."


def speak_text(text: str, answer_speed: float = 1.5):
    """Convert text to speech and play it"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_mp3_path = tmp_file.name

        tts = gTTS(text=text, lang="en")
        tts.save(temp_mp3_path)

        temp_wav_path = temp_mp3_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(temp_mp3_path)

        if answer_speed != 1.0:
            audio = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * answer_speed)}
            ).set_frame_rate(audio.frame_rate)

        audio.export(temp_wav_path, format="wav")

        tts_sound = pygame.mixer.Sound(temp_wav_path)
        pygame.mixer.Sound.play(tts_sound)

        while pygame.mixer.get_busy():
            time.sleep(0.1)

        try:
            os.remove(temp_mp3_path)
            os.remove(temp_wav_path)
        except:
            pass
    except Exception as e:
        print(f"❌ Error with text-to-speech: {e}")


# ============================================================================
# QUESTION HANDLING WITH ACCESSIBLE DISPLAY
# ============================================================================

def handle_question(question: str, large_print: bool = False):
    """
    Process question: get Gemini answer, convert to Braille, save, and speak.
    Display uses accessibility-friendly formatting based on font legibility guidelines.
    """
    print_accessible_header(f"Question: {question}", large_print)
    
    # Get answer from Gemini
    print("🤖 Getting answer from Gemini...\n")
    answer = ask_gemini(question)
    
    # Display answer with accessible formatting
    print_accessible_section("💡 ANSWER", answer, large_print)
    
    # Convert to Braille
    braille_output = text_to_braille(answer)
    print_accessible_section("⠿ BRAILLE REPRESENTATION", braille_output, large_print)
    
    # Save Braille to timestamped file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_answer_{timestamp}.braille.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(braille_output)
    print(f"\n💾 Braille saved as: {filename}\n")
    
    # Speak answer aloud
    speak_text(answer)
    
    print("▶️ Auto-resuming playback in 2 seconds...")
    time.sleep(2)


# ============================================================================
# VOICE COMMAND LISTENER
# ============================================================================

def listen_for_commands(large_print: bool = False):
    """Continuously listens for voice commands."""
    global paused, stop_audio, pause_start_time
    
    recognizer = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    print(f"🎤 Available microphones: {len(mic_list)}")

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return

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

                print("🤔 What's your question? (You have 15 seconds)")
                try:
                    with mic as q_source:
                        question_audio = recognizer.listen(q_source, timeout=15, phrase_time_limit=10)
                        question = recognizer.recognize_google(question_audio).strip()
                        print(f"❓ Question: {question}")
                    handle_question(question, large_print)
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


# ============================================================================
# TEXT INPUT MODE
# ============================================================================

def text_input_mode(large_print: bool = False):
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
            handle_question(question, large_print)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to start the application"""
    global stop_audio

    print("\n" + "=" * 80)
    print("🎵 INTERACTIVE AUDIO Q&A APPLICATION")
    print("    Accessible Design Based on Font Legibility Guidelines")
    print("=" * 80)
    print("\nChoose your preferences:")
    print("\n1️⃣  Voice Command Mode (speak your questions)")
    print("2️⃣  Text Input Mode (type your questions)")
    print("\n" + "=" * 80)
    
    mode = input("Enter mode (1 or 2): ").strip()
    
    print("\n" + "-" * 80)
    print("Display Format:")
    print("S  -  Standard Print (11-12pt equivalent spacing)")
    print("L  -  Large Print (16pt+ equivalent spacing)")
    print("-" * 80)
    
    display_format = input("Enter format (S or L): ").strip().upper()
    large_print = (display_format == "L")
    
    format_type = "LARGE PRINT (16pt+ equivalent)" if large_print else "STANDARD PRINT (11-12pt equivalent)"
    print(f"\n✓ Using: {format_type}")
    print("=" * 80 + "\n")

    try:
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

        if mode == "2":
            text_input_mode(large_print)
        else:
            listen_for_commands(large_print)

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        stop_audio = True
        pygame.mixer.quit()
        print("\n🔚 Application terminated")
        print("=" * 80)


if __name__ == "__main__":
    main()