# Codes by Vision
import cv2
import numpy as np
import librosa
from skimage.feature import local_binary_pattern

class FeatureExtractor:
    def __init__(self):
        self.lbp_radius = 3
        self.lbp_points = 8 * self.lbp_radius

    def extract_visual_features(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        color_features = []
        for i in range(3):
            ch = face_img[:, :, i]
            color_features.extend([np.mean(ch), np.std(ch)])
        dct = cv2.dct(np.float32(gray) / 255.0)
        dct_features = dct[:8, :8].flatten()
        return np.concatenate([hist, color_features, dct_features])

    def extract_audio_features(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        return np.concatenate([mfccs_mean, mfccs_std, [sc_mean, sc_std, sr_mean, sr_std, zcr_mean]])
