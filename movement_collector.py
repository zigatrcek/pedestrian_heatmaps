import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
from datetime import datetime
import time
import argparse

class MovementDataCollector:
    def __init__(self, save_dir='movement_data', model_path='yolov8n.pt', 
                 detection_fps=10, resolution=(300, 300), kernel_size=(31, 31)):
        """
        Initialize the movement data collector with configurable parameters.
        
        Args:
            save_dir: Directory for saving movement data
            model_path: Path to YOLO model weights
            detection_fps: Detection frequency
            resolution: Processing resolution (width, height)
            kernel_size: Size of the Gaussian kernel for movement accumulation
        """
        self.model = YOLO(model_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.target_size = resolution
        self.accumulator = np.zeros(self.target_size, dtype=np.float32)
        self.background_saved = False
        
        # Create Gaussian kernel for movement accumulation
        self.kernel_size = kernel_size
        self.kernel = self._create_gaussian_kernel(kernel_size)
        
        self.detection_interval = int(30 / detection_fps)
        self.frame_count = 0
        self.last_boxes = None
        
        self.session_dir = self.save_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir.mkdir(exist_ok=True)
        
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'frame_count': 0,
            'detection_fps': detection_fps,
            'detection_interval': self.detection_interval,
            'resolution': list(self.target_size),
            'background_path': 'background.jpg',
            'kernel_size': kernel_size
        }
        self._save_metadata()
    
    def _create_gaussian_kernel(self, size):
        """
        Create a 2D Gaussian kernel for weighted movement accumulation.
        
        This implements what's known as a "spatial probability distribution"
        where movement detection confidence decreases radially from the center.
        """
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        x, y = np.meshgrid(x, y)
        
        # Create 2D Gaussian
        gaussian = np.exp(-(x**2 + y**2))
        
        # Normalize to ensure kernel sums to 1
        return gaussian / gaussian.sum()
    
    def _apply_gaussian_contribution(self, region, contribution=1.0):
        """
        Apply Gaussian-weighted contribution to a region of the accumulator.
        
        Args:
            region: Tuple of slices defining the region to update
            contribution: Base contribution value (typically 1.0)
        """
        y_slice, x_slice = region
        
        # Get region dimensions
        region_height = y_slice.stop - y_slice.start
        region_width = x_slice.stop - x_slice.start
        
        # Resize kernel to match region size
        if (region_height, region_width) != self.kernel_size:
            kernel = cv2.resize(self.kernel, (region_width, region_height))
            kernel = kernel / kernel.sum()  # Renormalize after resize
        else:
            kernel = self.kernel
        
        # Apply weighted contribution
        self.accumulator[y_slice, x_slice] += kernel * contribution
    
    def _save_metadata(self):
        with open(self.session_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _save_background(self, frame):
        """Save the first frame as background reference."""
        if not self.background_saved:
            bg_path = self.session_dir / 'background.jpg'
            cv2.imwrite(str(bg_path), frame)
            self.background_saved = True
    
    def process_frame(self, frame):
        # Resize input frame immediately
        frame = cv2.resize(frame, self.target_size)
        
        # Save first frame as background
        if not self.background_saved:
            self._save_background(frame)
        
        annotated_frame = frame.copy()
        
        if self.frame_count % self.detection_interval == 0:
            results = self.model(frame, classes=[0])
            
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                self.last_boxes = boxes
                self._update_accumulator_and_draw(annotated_frame, boxes)
        
        elif self.last_boxes is not None:
            self._update_accumulator_and_draw(annotated_frame, self.last_boxes)
        
        self._save_frame_data()
        self.frame_count += 1
        
        return annotated_frame
    
    def _update_accumulator_and_draw(self, frame, boxes):
        for box in boxes:
            x, y, w, h = map(int, box)
            
            # Calculate box boundaries with padding for Gaussian kernel
            pad_x = self.kernel_size[1] // 2
            pad_y = self.kernel_size[0] // 2
            
            x1 = max(pad_x, min(self.target_size[0] - pad_x - 1, int(x - w/2)))
            x2 = max(pad_x, min(self.target_size[0] - pad_x - 1, int(x + w/2)))
            y1 = max(pad_y, min(self.target_size[1] - pad_y - 1, int(y - h/2)))
            y2 = max(pad_y, min(self.target_size[1] - pad_y - 1, int(y + h/2)))
            
            # Create extended region for Gaussian application
            region = (
                slice(y1 - pad_y, y2 + pad_y),
                slice(x1 - pad_x, x2 + pad_x)
            )
            
            # Apply Gaussian-weighted contribution
            self._apply_gaussian_contribution(region)
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def _save_frame_data(self):
        frame_path = self.session_dir / f'frame_{self.metadata["frame_count"]:06d}.npy'
        np.save(frame_path, self.accumulator)
        self.metadata['frame_count'] += 1
        
        if self.metadata['frame_count'] % 100 == 0:
            self._save_metadata()
    
    def cleanup(self):
        self.metadata['end_time'] = datetime.now().isoformat()
        self._save_metadata()

def main():
    parser = argparse.ArgumentParser(description='Collect movement data from video')
    
    parser.add_argument('--video', type=str, default='0',
                       help='Path to video file or camera index (default: 0 for webcam)')
    parser.add_argument('--duration', type=float,
                       help='Duration to process in seconds (default: process entire video)')
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
    
    args = parser.parse_args()
    
    # Handle video source
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video}")
        return
    
    # Calculate frame limit if duration specified
    frame_limit = None
    if args.duration:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_limit = int(args.duration * video_fps)
    
    collector = MovementDataCollector(
        save_dir=args.output,
        detection_fps=args.fps,
        resolution=(args.width, args.height),
        kernel_size=(args.kernel_size, args.kernel_size)
    )
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (frame_limit and frame_count >= frame_limit):
                break
            
            annotated_frame = collector.process_frame(frame)
            cv2.imshow('Movement Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
            # Show progress for video files
            if args.duration:
                progress = (frame_count / frame_limit) * 100
                print(f"\rProgress: {progress:.1f}%", end='')
    
    finally:
        print("\nCleaning up...")
        collector.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()