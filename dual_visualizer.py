import cv2
import argparse
from movement_collector import MovementDataCollector

class DualVisualizer:
    def __init__(self, video_path, resolution=(300, 300), detection_fps=10):
        # Initialize camera
        self.cap_camera = cv2.VideoCapture(0)
        self.cap_camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Initialize video
        self.cap_video = cv2.VideoCapture(video_path)
        
        # Initialize movement collectors
        self.collector_camera = MovementDataCollector(
            save_dir='local_data',
            detection_fps=detection_fps,
            resolution=resolution,
            is_camera=True
        )
        
        self.collector_video = MovementDataCollector(
            save_dir='remote_data',
            detection_fps=detection_fps,
            resolution=resolution,
            is_camera=False
        )
        
        # Initialize display windows
        cv2.namedWindow('Local Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
    
    def run(self):
        try:
            while True:
                # Process camera feed
                ret_camera, frame_camera = self.cap_camera.read()
                if ret_camera:
                    frame_camera = cv2.resize(frame_camera, (300, 300))
                    visualized_camera = self.collector_camera.process_frame(frame_camera)
                    cv2.imshow('Local Camera Feed', visualized_camera)
                
                # Process video feed
                ret_video, frame_video = self.cap_video.read()
                if ret_video:
                    frame_video = cv2.resize(frame_video, (300, 300))
                    visualized_video = self.collector_video.process_frame(frame_video)
                    cv2.imshow('Video Feed', visualized_video)
                else:
                    # Reset video to beginning when it ends
                    self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.collector_camera.cleanup()
        self.collector_video.cleanup()
        self.cap_camera.release()
        self.cap_video.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Run dual visualization with camera and video')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--width', type=int, default=300,
                       help='Video width (default: 300)')
    parser.add_argument('--height', type=int, default=300,
                       help='Video height (default: 300)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Detection FPS (default: 10)')
    
    args = parser.parse_args()
    
    visualizer = DualVisualizer(
        video_path=args.video_path,
        resolution=(args.width, args.height),
        detection_fps=args.fps
    )
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        visualizer.cleanup()

if __name__ == '__main__':
    main()