# DeepTrace

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Deepfake%20Detection-FF6F00?style=for-the-badge&logo=google-cloud&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Operational-00C853?style=for-the-badge&logo=cachet&logoColor=white" />
</p>


**DeepTrace** is a forensic deepfake detection system designed to analyze **Video**, **Audio**, and **Images**. It uses a hybrid approach of facial feature extraction (LBP, DCT, Color Analysis) and audio signal analysis (MFCC, Spectral Features) to distinguish between real and AI-generated media.

## 🚀 Features
- **Video Detection**: Analyzes frame-by-frame visual artifacts. Handles face detection failures by analyzing full frames.
- **Audio Detection**: Detects synthetic voice patterns, noise, and clipping artifacts.
- **Image Detection**: Identifies pixel-level irregularities common in GANs and diffusion models.
- **Pre-Trained Models**: Comes with pre-loaded models (trained on synthetic datasets) to demonstrate functionality immediately.

## 📂 Project Structure
```
DeepTrace/
├── samples/             # Test files (deepfake video/audio)
├── saved_models/        # Trained .pkl models for Video, Audio, and Image
├── .venv/               # Python virtual environment
├── main.py              # Entry point
├── detector.py          # Core detection logic
├── feature_extractor.py # Feature extraction algorithms
├── model_handler.py     # Model loading and saving
├── *_processor.py       # Helper modules for media processing
├── train.py             # Training script
└── run.sh               # Execution helper script
```

## 🛠️ Step-by-Step Installation

1. **Ensure Python Installed**: Make sure you have Python 3.10 or newer installed on your Linux system.
   ```bash
   python3 --version
   ```

2. **Open the Project Directory**:
   ```bash
   cd ~/Desktop/DeepTrace
   ```

3. **Set Up Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ How to Use

### 1. Analyze a File (Detection)
We provide a helper script `run.sh` that automatically sets up the environment and runs the detector.

**Step 1:** Locate the file you want to analyze (e.g., a video or image).

**Step 2:** Run the following command in your terminal:
```bash
./run.sh path/to/your/file.mp4
```

**Examples:**
```bash
# Analyze the included sample deepfake video
./run.sh samples/test_deepfake.mp4

# Analyze the included sample fake audio
./run.sh samples/test_fake_audio.wav

# Analyze an image from your downloads
./run.sh /home/vision/Downloads/suspect_image.jpg
```

### 2. Retraining the Models
The tool comes with models trained on synthetic data for demonstration. For real-world usage, you should train it on your own dataset.

**Step 1:** Organize your data into directories for Real and Fake samples.
   - `data/real_images/`
   - `data/fake_images/`

**Step 2:** Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

**Step 3:** Run the training command:
   ```bash
   # For Images
   python train.py --real_dir data/real_images --fake_dir data/fake_images --type image
   
   # For Video
   python train.py --real_dir data/real_videos --fake_dir data/fake_videos --type video
   
   # For Audio
   python train.py --real_dir data/real_audio --fake_dir data/fake_audio --type audio
   ```

## 📜 License
This project is licensed under the **Apache License 2.0**.

## 👨‍💻 Author
**Vision KC**
[GITHUB](https://github.com/vision-dev1)
[SITE](https://visionkc.com.np)

