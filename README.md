# Pedestrian Counter

This Python app counts pedestrians entering and exiting a venue using a live webcam or a video file. It uses YOLOv8 for pedestrian detection and ByteTrack for tracking. A counting line is defined in the code, and whenever a pedestrian crosses the line, an enter or exit event is counted and visualized.

## Features
- Detects and tracks pedestrians in real-time
- Counts entries and exits based on a defined line
- Works with webcam or video file
- Visualizes bounding boxes, counting line, and live counts

## Requirements
- Python 3.8+

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

### With Webcam
```bash
python main.py
```

### With Video File
```bash
python main.py --video path_to_video.mp4
```

Press `q` to quit the visualization window.

## Customization
- To change the position of the counting line, edit the `LINE` variable in `main.py`. 