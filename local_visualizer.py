import cv2
from movement_collector import MovementDataCollector

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    
    # Initialize movement collector
    collector = MovementDataCollector(
        save_dir='local_data',
        detection_fps=10,
        resolution=(300, 300),
        is_camera=True
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame through movement collector
            visualized_frame = collector.process_frame(frame)
            cv2.imshow('Local Feed with Movement Visualization', visualized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        collector.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()