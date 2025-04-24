import cv2
import time
import os
import threading
import queue

# Create output directory
os.makedirs('video', exist_ok=True)

# Initialize the RTSP stream with FFMPEG backend
rtsp_url = 'rtsp://admin:123admin@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to connect to IP Camera")
    exit()

# Set RTSP properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)  # Increase buffer size
cap.set(cv2.CAP_PROP_FPS, 30)  # Force FPS (adjust to match camera)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Use H264 for RTSP

# Get frame dimensions and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if not detected
print(f"Recording at {width}x{height}, {fps} FPS")

# Define the codec and create VideoWriter object
output_file = 'video/video2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Verify VideoWriter
if not out.isOpened():
    print("Error: VideoWriter failed to initialize")
    cap.release()
    exit()

# Set recording duration (60 seconds)
record_duration = 1800

# Queue for frames
frame_queue = queue.Queue(maxsize=100)
stop_thread = False
frame_count = 0

# Thread to capture frames
def capture_frames():
    global frame_count, stop_thread
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        try:
            frame_queue.put_nowait((ret, frame))
            frame_count += 1
        except queue.Full:
            print("Queue full, dropping frame")
            continue

# Start capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

print("Recording started...")
start_time = time.time()
display_count = 0

while (time.time() - start_time) < record_duration:
    try:
        ret, frame = frame_queue.get(timeout=0.1)  # Get frame from queue
        if not ret:
            continue
        
        # Write frame to output file
        out.write(frame)
        
        # Display every 10th frame to reduce load
        display_count += 1
        if display_count % 10 == 0:
            cv2.imshow("IP Camera Stream", frame)
        
        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except queue.Empty:
        continue

# Cleanup
stop_thread = True
capture_thread.join()
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Recording stopped. Video saved as {output_file}")
print(f"Captured {frame_count} frames, expected {int(fps * record_duration)} frames")
