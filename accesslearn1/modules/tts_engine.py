# modules/tts_engine.py  — gTTS + pydub speed control → WAV file

import os
import tempfile

try:
    from gtts import gTTS
    from pydub import AudioSegment
    READY = True
except ImportError:
    READY = False


def speak_text_to_file(text: str, output_wav: str, speed: float = 1.5) -> bool:
    """
    Convert text to speech, apply speed, save as WAV.
    Returns True on success.
    """
    if not READY:
        return False
    try:
        tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_mp3.close()

        tts = gTTS(text=text, lang="en")
        tts.save(tmp_mp3.name)

        audio = AudioSegment.from_mp3(tmp_mp3.name)

        if speed != 1.0:
            audio = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            ).set_frame_rate(audio.frame_rate)

        audio.export(output_wav, format="wav")
        os.unlink(tmp_mp3.name)
        return True
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False


def speak_text_live(text: str, speed: float = 1.5):
    """
    Convert text to speech and play immediately via pygame.
    """
    try:
        import pygame, time
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        if speak_text_to_file(text, tmp_wav.name, speed):
            snd = pygame.mixer.Sound(tmp_wav.name)
            pygame.mixer.Sound.play(snd)
            while pygame.mixer.get_busy():
                time.sleep(0.1)
        try: os.unlink(tmp_wav.name)
        except: pass
    except Exception as e:
        print(f"❌ Live TTS error: {e}")
