# Multi-Person Movement Heatmap Generator

A tool for generating and visualizing movement heatmaps from video, with support for tracking multiple people with different colors.

## Setup

1. Install requirements:
```bash
pip install ultralytics opencv-python numpy
```

2. Download YOLO model:
```bash
yolo detect download yolov8n.pt
```

## Usage

### Collecting Movement Data

Basic usage with webcam:
```bash
python movement_collector.py
```

Options:
```bash
python movement_collector.py --video path/to/video.mp4  # Use video file
python movement_collector.py --width 640 --height 480   # Change resolution
python movement_collector.py --fps 15                   # Adjust detection FPS
python movement_collector.py --duration 60              # Process 60 seconds
python movement_collector.py --intensity 1.5            # Adjust heatmap intensity
python movement_collector.py --threshold 0.2            # Change activity threshold
```

### Visualizing Results

Create static visualization:
```bash
python movement_visualizer.py movement_data/[session_dir]
```

Options:
```bash

# Create trails visualization
python movement_visualizer.py movement_data/[session_dir] --style trails

# Generate timelapse video
python movement_visualizer.py movement_data/[session_dir] --timelapse

# Adjust visualization parameters
python movement_visualizer.py movement_data/[session_dir] \
    --intensity 1.5 \
    --threshold 80 \
    --no-background
```

### Running Video Calls

Start host:
```bash
python video_call.py --is-host
```

Connect client:
```bash
python video_call.py --host [host_ip]
```

## Output Data Structure

Data is saved in `movement_data/[timestamp]`:
- `metadata.json`: Session information
- `background.jpg`: First frame as reference
- `color_[n]_frame_[xxxxx].npy`: Movement data for each tracked person
- Video files: Records of visualization
