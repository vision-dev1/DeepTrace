# Codes by Vision
import os
from moviepy import VideoFileClip
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_audio(self, video_path, output_path="temp_audio.wav"):
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                return None
            video.audio.write_audiofile(output_path, fps=self.sample_rate, logger=None)
            return output_path
        except Exception:
            return None

    def load_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return y, sr
        except Exception:
            return None, None

    def normalize_audio(self, y):
        if y is None:
            return None
        return librosa.util.normalize(y)

    def cleanup(self, path):
        if os.path.exists(path):
            os.remove(path)
