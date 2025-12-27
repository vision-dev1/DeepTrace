# Codes by Vision
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelHandler:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Get the directory of this script (model_handler.py)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(base_dir, "saved_models")
        else:
            self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.video_model_path = os.path.join(self.model_dir, "video_model.pkl")
        self.audio_model_path = os.path.join(self.model_dir, "audio_model.pkl")
        self.image_model_path = os.path.join(self.model_dir, "image_model.pkl")

    def train_video_model(self, X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.video_model_path)
        return model

    def train_audio_model(self, X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.audio_model_path)
        return model

    def train_image_model(self, X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.image_model_path)
        return model

    def load_video_model(self):
        if os.path.exists(self.video_model_path):
            return joblib.load(self.video_model_path)
        return None

    def load_audio_model(self):
        if os.path.exists(self.audio_model_path):
            return joblib.load(self.audio_model_path)
        return None

    def load_image_model(self):
        if os.path.exists(self.image_model_path):
            return joblib.load(self.image_model_path)
        return None

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred)
        }
