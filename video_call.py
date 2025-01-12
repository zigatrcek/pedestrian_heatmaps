import cv2
import numpy as np
import socket
import threading
import pickle
import zlib
from movement_collector import MovementDataCollector

class VideoCallNode:
    def __init__(self, is_host=False, host='localhost', port=65432, 
                 resolution=(300, 300), detection_fps=10):
        """
        Initialize a video call node that can both send and receive video streams.
        
        Args:
            is_host: Whether this instance is the host (True) or client (False)
            host: Host address to connect to/listen on
            port: Port number for the connection
            resolution: Video resolution (width, height)
            detection_fps: Detection framerate for movement visualization
        """
        self.resolution = resolution
        self.is_host = is_host
        self.host = host
        self.port = port
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Initialize movement collector
        self.collector = MovementDataCollector(
            save_dir='call_data',
            detection_fps=detection_fps,
            resolution=resolution,
            is_camera=True
        )
        
        # Initialize networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if is_host:
            self.sock.bind((host, port))
            self.sock.listen(1)
        
        # Initialize display windows
        cv2.namedWindow('Local Feed (With Visualization)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Remote Feed', cv2.WINDOW_NORMAL)
        
        self.running = False
        self.remote_frame = None
    
    def start(self):
        """Start the video call."""
        self.running = True
        
        if self.is_host:
            print(f"Waiting for connection on {self.host}:{self.port}...")
            self.conn, self.addr = self.sock.accept()
            print(f"Connected to {self.addr}")
        else:
            print(f"Connecting to {self.host}:{self.port}...")
            self.sock.connect((self.host, self.port))
            self.conn = self.sock
        
        # Start send and receive threads
        self.send_thread = threading.Thread(target=self._send_frames)
        self.recv_thread = threading.Thread(target=self._receive_frames)
        
        self.send_thread.start()
        self.recv_thread.start()
        
        # Main display loop
        try:
            while self.running:
                if self.remote_frame is not None:
                    cv2.imshow('Remote Feed', self.remote_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
        finally:
            self.stop()
    
    def _send_frames(self):
        """Capture, process, and send local frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Resize frame
            frame = cv2.resize(frame, self.resolution)
            
            # Process frame through movement collector
            visualized_frame = self.collector.process_frame(frame)
            cv2.imshow('Local Feed (With Visualization)', visualized_frame)
            
            # Compress and send frame
            try:
                data = pickle.dumps(frame)
                compressed = zlib.compress(data)
                
                # Send data size first, then data
                size = len(compressed)
                self.conn.sendall(size.to_bytes(4, byteorder='big'))
                self.conn.sendall(compressed)
            except (socket.error, IOError) as e:
                print(f"Error sending frame: {e}")
                self.stop()
                break
    
    def _receive_frames(self):
        """Receive and display remote frames."""
        while self.running:
            try:
                # Receive data size
                size_data = self._recvall(4)
                if not size_data:
                    break
                size = int.from_bytes(size_data, byteorder='big')
                
                # Receive frame data
                frame_data = self._recvall(size)
                if not frame_data:
                    break
                
                # Decompress and unpickle frame
                decompressed = zlib.decompress(frame_data)
                frame = pickle.loads(decompressed)
                
                self.remote_frame = frame
                
            except (socket.error, IOError, pickle.UnpicklingError) as e:
                print(f"Error receiving frame: {e}")
                self.stop()
                break
    
    def _recvall(self, n):
        """Helper function to receive n bytes or return None if EOF is hit."""
        data = bytearray()
        while len(data) < n:
            try:
                packet = self.conn.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.error:
                return None
        return data
    
    def stop(self):
        """Stop the video call and cleanup resources."""
        self.running = False
        
        if hasattr(self, 'conn'):
            self.conn.close()
        self.sock.close()
        self.cap.release()
        self.collector.cleanup()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Start a video call node')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host address (default: localhost)')
    parser.add_argument('--port', type=int, default=65432,
                       help='Port number (default: 65432)')
    parser.add_argument('--is-host', action='store_true',
                       help='Run as host (server)')
    parser.add_argument('--width', type=int, default=300,
                       help='Video width (default: 300)')
    parser.add_argument('--height', type=int, default=300,
                       help='Video height (default: 300)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Detection FPS (default: 10)')
    
    args = parser.parse_args()
    
    node = VideoCallNode(
        is_host=args.is_host,
        host=args.host,
        port=args.port,
        resolution=(args.width, args.height),
        detection_fps=args.fps
    )
    
    try:
        node.start()
    except KeyboardInterrupt:
        node.stop()

if __name__ == '__main__':
    main()