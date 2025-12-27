# Codes by Vision
import cv2
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_faces(self, video_path, max_frames=50):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        faces = []
        for i in tqdm(range(total_frames), desc="Processing video frames", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(detections) > 0:
                    (x, y, width, height) = detections[0]
                    x, y = max(0, x), max(0, y)
                    x_end, y_end = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
                    face_img = frame[y:y_end, x:x_end]
                    if face_img.size > 0:
                        face_img = cv2.resize(face_img, (224, 224))
                        faces.append(face_img)
                else:
                    # Fallback: if no face detected in frame, use the whole frame (resized)
                    # This ensures we don't return an empty list for valid videos with hard-to-detect faces
                    resized_full = cv2.resize(frame, (224, 224))
                    faces.append(resized_full)
        cap.release()
        return faces
