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
                 detection_fps=10, resolution=(300, 300), kernel_size=(31, 31),
                 is_camera=False, video_fps=30.0):
        """
        Initialize the movement data collector with configurable parameters.
        
        Args:
            save_dir: Directory for saving movement data
            model_path: Path to YOLO model weights
            detection_fps: Detection frequency
            resolution: Processing resolution (width, height)
            kernel_size: Size of the Gaussian kernel for movement accumulation
            is_camera: Boolean indicating if input is from camera
            video_fps: Frame rate for output video (default: 30.0)
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
        
        # Video recording parameters
        self.is_camera = is_camera
        self.video_writer = None
        self.recording_start_time = time.time()
        self.video_fps = video_fps
        self.initialize_video_writer()
        
        # Initialize visualization parameters
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
    
    def _create_gaussian_kernel(self, size):
        """Create a 2D Gaussian kernel for weighted movement accumulation."""
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        x, y = np.meshgrid(x, y)
        
        gaussian = np.exp(-(x**2 + y**2))
        return gaussian / gaussian.sum()
    
    def _apply_gaussian_contribution(self, region, contribution=1.0):
        """Apply Gaussian-weighted contribution to a region of the accumulator."""
        y_slice, x_slice = region
        
        region_height = y_slice.stop - y_slice.start
        region_width = x_slice.stop - x_slice.start
        
        if (region_height, region_width) != self.kernel_size:
            kernel = cv2.resize(self.kernel, (region_width, region_height))
            kernel = kernel / kernel.sum()
        else:
            kernel = self.kernel
        
        self.accumulator[y_slice, x_slice] += kernel * contribution
    
    def _create_heatmap_overlay(self, frame):
        """Create a colored heatmap overlay from the current accumulator state."""
        # Normalize accumulator to 0-255 range with dynamic scaling
        max_val = np.max(self.accumulator)
        if max_val > 0:
            normalized = (self.accumulator / max_val * 255).clip(0, 255)
        else:
            normalized = self.accumulator
        
        # Threshold low activity
        normalized[normalized < self.activity_threshold * 255] = 0
        
        # Convert to uint8 and apply colormap
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_HOT)
        
        # Create mask for overlay
        mask = (normalized > 0)[..., np.newaxis]
        
        # Blend with original frame
        blended = frame.astype(np.float32)
        overlay = heatmap.astype(np.float32) * self.intensity_scale
        blended = cv2.addWeighted(blended, 1.0, overlay, 0.6, 0)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def initialize_video_writer(self):
        """Initialize or reinitialize the video writer with H.264 encoding."""
        if self.video_writer is not None:
            self.video_writer.release()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.is_camera:
            video_path = str(self.session_dir / 'camera_latest.mp4')
        else:
            video_path = str(self.session_dir / f'recording_{timestamp}.mp4')

        # Try different codec options in order of preference
        codecs = [
            ('avc1', 'mp4'),  # H.264 in MP4
            ('h264', 'mp4'),  # Alternative H.264 name
            ('hevc', 'mp4'),  # H.265/HEVC
            ('vp09', 'webm')  # VP9 in WebM (fallback for some systems)
        ]

        for codec, ext in codecs:
            if ext != 'mp4':
                # Update extension if using alternative container format
                video_path = str(Path(video_path).with_suffix(f'.{ext}'))
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                self.video_fps,  # Use input video's FPS
                self.target_size,
                True  # isColor
            )
            
            if writer.isOpened():
                self.video_writer = writer
                print(f"Successfully initialized video writer with codec: {codec}")
                return

        # If no codec worked, fall back to basic MJPG (should work everywhere but larger files)
        if self.video_writer is None or not self.video_writer.isOpened():
            video_path = str(Path(video_path).with_suffix('.avi'))
            self.video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'MJPG'),
                self.video_fps,
                self.target_size,
                True
            )
            print("Falling back to MJPG codec")

    def check_recording_duration(self):
        """Check if we need to start a new recording (camera mode only)."""
        if self.is_camera:
            current_time = time.time()
            if current_time - self.recording_start_time >= 60:  # 60 seconds = 1 minute
                self.initialize_video_writer()
                self.recording_start_time = current_time
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.session_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _save_background(self, frame):
        """Save the first frame as background reference."""
        if not self.background_saved:
            bg_path = self.session_dir / 'background.jpg'
            cv2.imwrite(str(bg_path), frame)
            self.background_saved = True
    
    def process_frame(self, frame):
        """
        Process a single frame from the video stream.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Frame with visualization overlay
        """
        # Resize input frame immediately
        frame = cv2.resize(frame, self.target_size)
        
        # Save first frame as background
        if not self.background_saved:
            self._save_background(frame)
        
        if self.frame_count % self.detection_interval == 0:
            results = self.model(frame, classes=[0])
            
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                self.last_boxes = boxes
                self._update_accumulator(boxes)
        
        elif self.last_boxes is not None:
            self._update_accumulator(self.last_boxes)
        
        # Create heatmap overlay
        visualized_frame = self._create_heatmap_overlay(frame)
        
        # Save frame to video
        if self.video_writer is not None:
            self.video_writer.write(visualized_frame)
            self.check_recording_duration()
        
        self._save_frame_data()
        self.frame_count += 1
        
        return visualized_frame
    
    def _update_accumulator(self, boxes):
        """Update the accumulator with new detections."""
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
    
    def _save_frame_data(self):
        """Save current frame data to disk."""
        frame_path = self.session_dir / f'frame_{self.metadata["frame_count"]:06d}.npy'
        np.save(frame_path, self.accumulator)
        self.metadata['frame_count'] += 1
        
        if self.metadata['frame_count'] % 100 == 0:
            self._save_metadata()
    
    def cleanup(self):
        """Cleanup resources and save final metadata."""
        if self.video_writer is not None:
            self.video_writer.release()
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
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='Heatmap intensity (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Activity threshold (default: 0.1)')
    
    args = parser.parse_args()
    
    # Handle video source
    is_camera = args.video.isdigit()
    if is_camera:
        cap = cv2.VideoCapture(int(args.video))
        video_fps = 30.0  # Standard camera FPS
    else:
        cap = cv2.VideoCapture(args.video)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Input video FPS: {video_fps}")
    
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
        kernel_size=(args.kernel_size, args.kernel_size),
        is_camera=is_camera,
        video_fps=video_fps
    )
    
    # Set visualization parameters
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