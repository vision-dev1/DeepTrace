# Codes by Vision
import argparse
import os
import numpy as np
import glob
from video_processor import VideoProcessor
from audio_processor import AudioProcessor
from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from model_handler import ModelHandler
from cli_utils import display_banner, print_status, print_success, print_error

def train():
    display_banner()
    parser = argparse.ArgumentParser(description="DeepTrace Model Trainer")
    parser.add_argument("--real_dir", required=True, help="Directory containing real video/audio/image files")
    parser.add_argument("--fake_dir", required=True, help="Directory containing fake video/audio/image files")
    parser.add_argument("--type", choices=["video", "audio", "image"], required=True, help="Type of model to train")
    args = parser.parse_args()
    v_proc = VideoProcessor()
    a_proc = AudioProcessor()
    i_proc = ImageProcessor()
    f_ext = FeatureExtractor()
    m_hdl = ModelHandler()
    features = []
    labels = []
    def process_dir(directory, label):
        files = glob.glob(os.path.join(directory, "*"))
        for f in files:
            print_status(f"Processing: {os.path.basename(f)}")
            if args.type == "video":
                faces = v_proc.extract_faces(f, max_frames=20)
                for face in faces:
                    features.append(f_ext.extract_visual_features(face))
                    labels.append(label)
            elif args.type == "audio":
                y, sr = a_proc.load_audio(f)
                if y is not None:
                    features.append(f_ext.extract_audio_features(y, sr))
                    labels.append(label)
            elif args.type == "image":
                images = i_proc.process_image(f)
                if images:
                    for img in images:
                        features.append(f_ext.extract_visual_features(img))
                        labels.append(label)
    print_status(f"Extracting features for REAL {args.type}...")
    process_dir(args.real_dir, 0)
    print_status(f"Extracting features for FAKE {args.type}...")
    process_dir(args.fake_dir, 1)
    if not features:
        print_error("No features extracted. Check your directories and file formats.")
        return
    X = np.array(features)
    y = np.array(labels)
    print_status(f"Training {args.type} model...")
    if args.type == "video":
        model = m_hdl.train_video_model(X, y)
    elif args.type == "audio":
        model = m_hdl.train_audio_model(X, y)
    else:
        model = m_hdl.train_image_model(X, y)
    stats = m_hdl.evaluate(model, X, y)
    print_success(f"Training complete! Accuracy on training set: {stats['accuracy']:.2%}")
    
    # Get the correct path for the specific model type
    model_path = ""
    if args.type == "video": model_path = m_hdl.video_model_path
    elif args.type == "audio": model_path = m_hdl.audio_model_path
    else: model_path = m_hdl.image_model_path
    
    print_success(f"Model saved at: {model_path}")
    print("\nClassification Report:")
    print(stats['report'])

if __name__ == "__main__":
    train()
