# modules/audio_player.py

import os
import time
import tempfile
import threading

try:
    import pygame
    from pydub import AudioSegment
    READY = True
except ImportError:
    READY = False


class AudioPlayer:
    def __init__(self, audio_file: str):
        self.audio_file  = audio_file
        self.full_audio  = None
        if READY:
            self._load()

    def _load(self):
        try:
            ext = os.path.splitext(self.audio_file)[1].lower().lstrip('.')
            fmt = "wav" if ext == "wav" else "mp3"
            self.full_audio = AudioSegment.from_file(self.audio_file, format=fmt)
        except Exception as e:
            print(f"❌ AudioPlayer load error: {e}")

    def play_from_position(self, start_ms: int = 0):
        if not self.full_audio:
            return False
        remaining = self.full_audio[start_ms:]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        remaining.export(tmp.name, format="wav")
        try:
            sound = pygame.mixer.Sound(tmp.name)
            pygame.mixer.Sound.play(sound)

            def _cleanup():
                time.sleep(0.8)
                try: os.unlink(tmp.name)
                except: pass
            threading.Thread(target=_cleanup, daemon=True).start()
            return True
        except Exception as e:
            print(f"❌ AudioPlayer play error: {e}")
            return False

    def stop(self):
        if READY:
            pygame.mixer.stop()

    def is_playing(self) -> bool:
        return READY and pygame.mixer.get_busy()

    def get_duration_ms(self) -> int:
        return len(self.full_audio) if self.full_audio else 0
