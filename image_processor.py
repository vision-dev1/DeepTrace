# Codes by Vision
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        processed_images = []
        if len(detections) > 0:
            for (x, y, width, height) in detections:
                x, y = max(0, x), max(0, y)
                x_end, y_end = min(image.shape[1], x + width), min(image.shape[0], y + height)
                face_img = image[y:y_end, x:x_end]
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (224, 224))
                    processed_images.append(face_img)
        if not processed_images:
            resized_full = cv2.resize(image, (224, 224))
            processed_images.append(resized_full)
        return processed_images
