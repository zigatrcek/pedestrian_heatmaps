import cv2
import numpy as np
from pathlib import Path
import json
import argparse

class MovementVisualizer:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self._load_metadata()
        self._load_background()
        
        self.width = self.metadata['resolution'][0]
        self.height = self.metadata['resolution'][1]
        print(f"Detected resolution: {self.width}x{self.height}")
    
    def _load_metadata(self):
        try:
            with open(self.session_dir / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
                if 'resolution' not in self.metadata:
                    raise ValueError("Session metadata missing resolution information")
        except FileNotFoundError:
            raise RuntimeError(f"No metadata found in {self.session_dir}")
    
    def _load_background(self):
        bg_path = self.session_dir / self.metadata['background_path']
        
        if bg_path.exists():
            self.background = cv2.imread(str(bg_path))
            if self.background.shape[:2] != (self.metadata['resolution'][1], 
                                           self.metadata['resolution'][0]):
                self.background = cv2.resize(self.background, 
                                          tuple(self.metadata['resolution']))
        else:
            print("Warning: Background image not found, using black background")
            self.background = np.zeros((self.metadata['resolution'][1], 
                                      self.metadata['resolution'][0], 3), 
                                     dtype=np.uint8)
    
    def create_visualization(self, style='heatmap', colormap=cv2.COLORMAP_HOT,
                           blur_kernel=(15, 15), intensity_scale=1.0,
                           activity_threshold=75, no_background=False):
        """
        Create artistic visualization using additive blending for movement patterns.
        
        Args:
            style: 'heatmap' or 'trails'
            colormap: OpenCV colormap for visualization
            blur_kernel: Tuple defining the smoothing kernel size
            intensity_scale: Scaling factor for movement intensity
            activity_threshold: Percentile threshold for activity
            no_background: If True, renders movement patterns on black background
        """
        final_frame = f'frame_{self.metadata["frame_count"]-1:06d}.npy'
        accumulator = np.load(self.session_dir / final_frame)
        
        if accumulator.shape != (self.height, self.width):
            accumulator = cv2.resize(accumulator, (self.width, self.height))
        
        # Apply activity threshold
        threshold_value = np.percentile(accumulator, activity_threshold)
        accumulator[accumulator < threshold_value] = 0
        
        # Normalize to float32 for precise calculations
        normalized = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        
        if style == 'trails':
            _, binary = cv2.threshold(normalized, 1, 255, cv2.THRESH_BINARY)
            visualization = cv2.GaussianBlur(binary, blur_kernel, 0)
        else:
            visualization = cv2.GaussianBlur(normalized, blur_kernel, 0)
        
        # Apply colormap to uint8 version
        visualization_uint8 = visualization.astype(np.uint8)
        colored = cv2.applyColorMap(visualization_uint8, colormap)
        
        if no_background:
            # For no-background mode, just mask the colored visualization
            mask = (visualization > 0)[..., np.newaxis]
            black_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            result = np.where(mask, colored * intensity_scale, black_bg)
        else:
            # Standard additive blending with background
            colored_float = colored.astype(np.float32)
            background_float = self.background.astype(np.float32)
            mask = (visualization > 0)[..., np.newaxis]
            blended = background_float + (colored_float * intensity_scale * mask)
            result = np.clip(blended, 0, 255, casting='unsafe').astype(np.uint8)
        
        return result.astype(np.uint8)
    
    def _visualize_frame(self, accumulator, activity_threshold=75,
                        intensity_scale=1.0, no_background=False):
        if accumulator.shape != (self.height, self.width):
            accumulator = cv2.resize(accumulator, (self.width, self.height))
        
        threshold_value = np.percentile(accumulator, activity_threshold)
        accumulator[accumulator < threshold_value] = 0
            
        normalized = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_HOT)
        
        if no_background:
            mask = (normalized > 0)[..., np.newaxis]
            black_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            result = np.where(mask, colored * intensity_scale, black_bg)
        else:
            colored_float = colored.astype(np.float32)
            background_float = self.background.astype(np.float32)
            mask = (normalized > 0)[..., np.newaxis]
            blended = background_float + (colored_float * intensity_scale * mask)
            result = np.clip(blended, 0, 255, casting='unsafe').astype(np.uint8)
        
        return result.astype(np.uint8)
    
    def create_timelapse(self, output_path, frame_stride=10, activity_threshold=75,
                        intensity_scale=1.0, no_background=False):
        """Create timelapse video of movement accumulation."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use more compatible codec for MP4
        output_path = str(Path(output_path).with_suffix('.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        print(f"Creating timelapse video: {output_path}")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                            (self.width, self.height))
        
        if not out.isOpened():
            # Fallback to alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, 30.0,
                                (self.width, self.height))
            if not out.isOpened():
                raise RuntimeError("Failed to initialize video writer")
        
        try:
            frame_count = self.metadata['frame_count']
            for i in range(0, frame_count, frame_stride):
                print(f"\rProcessing frame {i}/{frame_count} ({i/frame_count*100:.1f}%)", end='')
                accumulator = np.load(
                    self.session_dir / f'frame_{i:06d}.npy'
                )
                frame = self._visualize_frame(
                    accumulator, 
                    activity_threshold=activity_threshold,
                    intensity_scale=intensity_scale,
                    no_background=no_background
                )
                out.write(frame)
        finally:
            out.release()
            print("\nTimelapse creation complete!")
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize movement data')
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--style', choices=['heatmap', 'trails'], 
                       default='heatmap',
                       help='Visualization style')
    parser.add_argument('--colormap', type=int, default=cv2.COLORMAP_HOT,
                       help='OpenCV colormap index (default: HOT)')
    parser.add_argument('--threshold', type=float, default=75,
                       help='Activity threshold percentile (0-100)')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='Movement intensity scale (0-5.0)')
    parser.add_argument('--output', default='visualization.mp4',
                       help='Output path for visualization')
    parser.add_argument('--timelapse', action='store_true',
                       help='Create timelapse video')
    parser.add_argument('--no-background', action='store_true',
                       help='Render movement patterns on black background')
    
    args = parser.parse_args()
    
    visualizer = MovementVisualizer(args.session_dir)
    
    if args.timelapse:
        visualizer.create_timelapse(
            args.output,
            activity_threshold=args.threshold,
            intensity_scale=args.intensity,
            no_background=args.no_background
        )
    else:
        if args.output.endswith('.mp4'):
            args.output = args.output[:-4] + '.png'
        visualization = visualizer.create_visualization(
            style=args.style,
            colormap=args.colormap,
            activity_threshold=args.threshold,
            intensity_scale=args.intensity,
            no_background=args.no_background
        )
        cv2.imwrite(args.output, visualization)

if __name__ == "__main__":
    main()
