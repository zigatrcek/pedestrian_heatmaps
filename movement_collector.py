import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
from datetime import datetime
import time
import argparse
import random

class MovementDataCollector:
    def __init__(self, save_dir='movement_data', model_path='yolov8n.pt', 
                 detection_fps=10, resolution=(300, 300), kernel_size=(31, 31),
                 is_camera=False, video_fps=30.0):
        self.model = YOLO(model_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.target_size = resolution
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255)   # Magenta
        ]
        self.accumulators = {i: np.zeros(self.target_size, dtype=np.float32) 
                           for i in range(len(self.colors))}
        self.background_saved = False
        self.available_colors = list(range(len(self.colors)))
        self.person_colors = {}
        
        self.kernel_size = kernel_size
        self.kernel = self._create_gaussian_kernel(kernel_size)
        
        self.detection_interval = int(30 / detection_fps)
        self.frame_count = 0
        self.last_detections = {}
        
        self.session_dir = self.save_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir.mkdir(exist_ok=True)
        
        self.is_camera = is_camera
        self.video_writer = None
        self.recording_start_time = time.time()
        self.video_fps = video_fps
        self.initialize_video_writer()
        
        self.intensity_scale = 1.0
        self.activity_threshold = 0.1
        
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'frame_count': 0,
            'detection_fps': detection_fps,
            'detection_interval': self.detection_interval,
            'resolution': list(self.target_size),
            'background_path': 'background.jpg',
            'kernel_size': kernel_size,
            'video_fps': video_fps
        }
        self._save_metadata()

    def _assign_person_id(self, box):
        x, y = box[:2]
        min_dist = float('inf')
        matched_id = None

        for pid, last_box in self.last_detections.items():
            dist = np.sqrt((x - last_box[0])**2 + (y - last_box[1])**2)
            if dist < min_dist and dist < 50:
                min_dist = dist
                matched_id = pid

        if matched_id is None:
            if not self.available_colors:
                self.available_colors = list(range(len(self.colors)))
            color_idx = random.choice(self.available_colors)
            self.available_colors.remove(color_idx)
            matched_id = color_idx

        return matched_id

    def _create_gaussian_kernel(self, size):
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(x**2 + y**2))
        return gaussian / gaussian.sum()
    
    def _apply_gaussian_contribution(self, region, accumulator, contribution=1.0):
        y_slice, x_slice = region
        region_height = y_slice.stop - y_slice.start
        region_width = x_slice.stop - x_slice.start
        
        if (region_height, region_width) != self.kernel_size:
            kernel = cv2.resize(self.kernel, (region_width, region_height))
            kernel = kernel / kernel.sum()
        else:
            kernel = self.kernel
        
        accumulator[y_slice, x_slice] += kernel * contribution

    def _create_heatmap_overlay(self, frame):
        blended = frame.astype(np.float32)
        
        for color_idx, accumulator in self.accumulators.items():
            max_val = np.max(accumulator)
            if max_val > 0:
                normalized = (accumulator / max_val * 255).clip(0, 255)
                normalized[normalized < self.activity_threshold * 255] = 0
                
                colored_heatmap = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)
                color = self.colors[color_idx]
                for i in range(3):
                    colored_heatmap[..., i] = normalized * (color[i] / 255)

                mask = (normalized > 0)[..., np.newaxis]
                overlay = colored_heatmap.astype(np.float32) * self.intensity_scale
                blended = np.where(mask, cv2.addWeighted(blended, 1.0, overlay, 0.6, 0), blended)

        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def initialize_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = str(self.session_dir / ('camera_latest.mp4' if self.is_camera else f'recording_{timestamp}.mp4'))

        codecs = [('avc1', 'mp4'), ('h264', 'mp4'), ('hevc', 'mp4'), ('vp09', 'webm')]
        for codec, ext in codecs:
            if ext != 'mp4':
                video_path = str(Path(video_path).with_suffix(f'.{ext}'))
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, self.target_size, True)
            if writer.isOpened():
                self.video_writer = writer
                return

        video_path = str(Path(video_path).with_suffix('.avi'))
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.video_fps,
            self.target_size,
            True
        )

    def check_recording_duration(self):
        if self.is_camera and time.time() - self.recording_start_time >= 60:
            self.initialize_video_writer()
            self.recording_start_time = time.time()
    
    def _save_metadata(self):
        with open(self.session_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _save_background(self, frame):
        if not self.background_saved:
            cv2.imwrite(str(self.session_dir / 'background.jpg'), frame)
            self.background_saved = True
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, self.target_size)
        
        if not self.background_saved:
            self._save_background(frame)
        
        if self.frame_count % self.detection_interval == 0:
            results = self.model(frame, verbose=False)
            if results[0].boxes:
                current_detections = {}
                boxes = results[0].boxes.xywh.cpu()
                
                for box in boxes:
                    person_id = self._assign_person_id(box)
                    current_detections[person_id] = box
                    self._update_accumulator(box, person_id)
                
                self.last_detections = current_detections
        
        elif self.last_detections:
            for person_id, box in self.last_detections.items():
                self._update_accumulator(box, person_id)
        
        visualized_frame = self._create_heatmap_overlay(frame)
        
        if self.video_writer is not None:
            self.video_writer.write(visualized_frame)
            self.check_recording_duration()
        
        self._save_frame_data()
        self.frame_count += 1
        
        return visualized_frame
    
    def _update_accumulator(self, box, person_id):
        x, y, w, h = map(int, box)
        
        pad_x = self.kernel_size[1] // 2
        pad_y = self.kernel_size[0] // 2
        
        x1 = max(pad_x, min(self.target_size[0] - pad_x - 1, int(x - w/2)))
        x2 = max(pad_x, min(self.target_size[0] - pad_x - 1, int(x + w/2)))
        y1 = max(pad_y, min(self.target_size[1] - pad_y - 1, int(y - h/2)))
        y2 = max(pad_y, min(self.target_size[1] - pad_y - 1, int(y + h/2)))
        
        region = (
            slice(y1 - pad_y, y2 + pad_y),
            slice(x1 - pad_x, x2 + pad_x)
        )
        
        self._apply_gaussian_contribution(region, self.accumulators[person_id])
    
    def _save_frame_data(self):
        for color_idx, accumulator in self.accumulators.items():
            np.save(self.session_dir / f'color_{color_idx}_frame_{self.metadata["frame_count"]:06d}.npy', 
                   accumulator)
        
        self.metadata['frame_count'] += 1
        if self.metadata['frame_count'] % 100 == 0:
            self._save_metadata()
    
    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
        self.metadata['end_time'] = datetime.now().isoformat()
        self._save_metadata()

def main():
    parser = argparse.ArgumentParser(description='Collect movement data from video')
    parser.add_argument('--video', type=str, default='0',
                       help='Path to video file or camera index (default: 0 for webcam)')
    parser.add_argument('--duration', type=float,
                       help='Duration to process in seconds')
    parser.add_argument('--width', type=int, default=300,
                       help='Processing width (default: 300)')
    parser.add_argument('--height', type=int, default=300,
                       help='Processing height (default: 300)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Detection FPS (default: 10)')
    parser.add_argument('--kernel-size', type=int, default=31,
                       help='Size of Gaussian kernel (default: 31)')
    parser.add_argument('--output', type=str, default='movement_data',
                       help='Output directory (default: movement_data)')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='Heatmap intensity (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Activity threshold (default: 0.1)')
    
    args = parser.parse_args()
    
    is_camera = args.video.isdigit()
    cap = cv2.VideoCapture(int(args.video) if is_camera else args.video)
    video_fps = 30.0 if is_camera else cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video}")
        return
    
    frame_limit = int(args.duration * video_fps) if args.duration else None
    
    collector = MovementDataCollector(
        save_dir=args.output,
        detection_fps=args.fps,
        resolution=(args.width, args.height),
        kernel_size=(args.kernel_size, args.kernel_size),
        is_camera=is_camera,
        video_fps=video_fps
    )
    
    collector.intensity_scale = args.intensity
    collector.activity_threshold = args.threshold
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (frame_limit and frame_count >= frame_limit):
                break
            
            visualized_frame = collector.process_frame(frame)
            cv2.imshow('Movement Heatmap', visualized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            if args.duration:
                print(f"\rProgress: {frame_count * 100 / frame_limit:.1f}%", end='')
    
    finally:
        print("\nCleaning up...")
        collector.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()