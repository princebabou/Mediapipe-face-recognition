# Face Recognition System

A real-time face recognition system using **MediaPipe** for face detection and **OpenCV's LBPH (Local Binary Patterns Histograms)** algorithm for face recognition.

## Features

- 🎥 Real-time face detection using MediaPipe
- 👤 Face recognition with LBPH algorithm
- 📸 Easy face data collection interface
- 🔄 Train custom models with your own data
- ⚡ Fast and efficient processing

## Technologies Used

- **Python 3.12**
- **OpenCV** - Computer vision and face recognition
- **MediaPipe** - Face detection
- **NumPy** - Numerical operations

## Project Structure

```
face_detection_assignment/
├── capture.py          # Capture face images for training
├── train.py           # Train the LBPH face recognition model
├── predict.py         # Real-time face recognition
├── data/              # Stored face images (gitignored)
├── models/            # Trained models (gitignored)
│   ├── lbph.yml      # Trained LBPH model
│   └── label_map.json # Label to ID mapping
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
https://github.com/princebabou/Mediapipe-face-Recognition.git
cd face-recognition-lbph
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-contrib-python mediapipe numpy
```

## Usage

### Step 1: Capture Face Data

Run the capture script to collect face images for training:

```bash
python capture.py
```

- Enter a label (person's name) when prompted
- Position your face in front of the camera
- The script will automatically capture 50 face images
- Press `q` to quit early
- Repeat for each person you want to recognize

### Step 2: Train the Model

After collecting face data for all people, train the recognition model:

```bash
python train.py
```

This will:
- Load all face images from the `data/` directory
- Train an LBPH face recognizer
- Save the model to `models/lbph.yml`
- Save the label mapping to `models/label_map.json`

### Step 3: Run Face Recognition

Start real-time face recognition:

```bash
python predict.py
```

- The camera will open and detect faces in real-time
- Recognized faces will be labeled with the person's name and confidence score
- Press `q` to quit

## How It Works

1. **Face Detection**: MediaPipe detects faces in the video stream and provides bounding box coordinates
2. **Face Extraction**: Detected faces are cropped, converted to grayscale, and resized to 200x200 pixels
3. **Feature Extraction**: LBPH algorithm extracts local binary pattern features from face images
4. **Recognition**: The trained model compares features and predicts the person's identity with a confidence score

## Configuration

### Camera Selection

If you have multiple cameras, modify the camera index in `capture.py` and `predict.py`:

```python
cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc.
```

### Number of Training Images

In `capture.py`, adjust the target number of images:

```python
target = 50  # Increase for better accuracy
```

### Detection Confidence

In `capture.py` and `predict.py`, adjust MediaPipe detection sensitivity:

```python
min_detection_confidence=0.5  # Range: 0.0 to 1.0
```

## Troubleshooting

### Camera Not Opening

- Ensure your camera is not being used by another application
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- Check camera permissions in your OS settings

### Module Not Found Errors

Make sure all dependencies are installed:

```bash
pip install opencv-contrib-python mediapipe numpy
```

Note: Use `opencv-contrib-python` (not `opencv-python`) for face recognition support.

### Low Recognition Accuracy

- Capture more training images (increase `target` in `capture.py`)
- Ensure good lighting conditions during capture and recognition
- Capture faces from different angles and expressions
- Retrain the model after adding more data

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for face detection
- [OpenCV](https://opencv.org/) for computer vision tools and LBPH face recognition
