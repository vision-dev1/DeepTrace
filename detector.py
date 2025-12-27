# Codes by Vision
import os
import numpy as np
from video_processor import VideoProcessor
from audio_processor import AudioProcessor
from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from model_handler import ModelHandler
from cli_utils import print_status, print_warning

class DeepTraceDetector:
    def __init__(self):
        self.video_proc = VideoProcessor()
        self.audio_proc = AudioProcessor()
        self.image_proc = ImageProcessor()
        self.feature_ext = FeatureExtractor()
        self.model_hdl = ModelHandler()
        self.v_model = self.model_hdl.load_video_model()
        self.a_model = self.model_hdl.load_audio_model()
        self.i_model = self.model_hdl.load_image_model()

    def analyze(self, file_path):
        results = {"video": None, "audio": None, "image": None}
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
        is_audio = ext in [".wav", ".mp3", ".flac", ".m4a"]
        is_image = ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        if is_video:
            print_status("Checking video content...")
            faces = self.video_proc.extract_faces(file_path)
            if faces:
                v_feats = [self.feature_ext.extract_visual_features(f) for f in faces]
                if self.v_model:
                    preds = self.v_model.predict_proba(v_feats)
                    fake_prob = np.mean(preds[:, 1])
                    results["video"] = {
                        "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
                        "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob
                    }
                else:
                    print_warning("Video model not found. Training required.")
            print_status("Checking audio content...")
            temp_audio = self.audio_proc.extract_audio(file_path)
            if temp_audio:
                results["audio"] = self._analyze_audio(temp_audio)
                self.audio_proc.cleanup(temp_audio)
        elif is_audio:
            print_status("Checking audio file...")
            results["audio"] = self._analyze_audio(file_path)
        elif is_image:
            print_status("Checking image file...")
            images = self.image_proc.process_image(file_path)
            if images:
                i_feats = [self.feature_ext.extract_visual_features(img) for img in images]
                if self.i_model:
                    preds = self.i_model.predict_proba(i_feats)
                    fake_prob = np.mean(preds[:, 1])
                    results["image"] = {
                        "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
                        "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob
                    }
                else:
                    print_warning("Image model not found. Training required.")
        return results

    def _analyze_audio(self, audio_path):
        y, sr = self.audio_proc.load_audio(audio_path)
        if y is not None:
            y = self.audio_proc.normalize_audio(y)
            a_feats = self.feature_ext.extract_audio_features(y, sr)
            if self.a_model:
                pred_proba = self.a_model.predict_proba([a_feats])[0]
                fake_prob = pred_proba[1]
                return {
                    "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
                    "confidence": fake_prob if fake_prob > 0.5 else 1 - fake_prob
                }
            else:
                print_warning("Audio model not found. Training required.")
        return None
