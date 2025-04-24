import json
import cv2
import os
import csv
from ultralytics import YOLO
import time
import base64
from imutils.video import VideoStream

USE_BUILD_IN_CAMERA = True

# Create directory for saving faces if it doesn't exist
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)


# Function to save ID and name to CSV file
def save_to_csv(person_id, person_name, photo_path, csv_file="data.csv"):
    """Save person's ID and name to CSV file"""
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Photo"])

    # Check if ID already exists
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 1 and str(person_id) == row[0]:
                    print("Already exist...")
                    return False

    # Save new entry to CSV
    with open(csv_file, "a", newline="") as f:
        print("Saving new record...")
        writer = csv.writer(f)
        writer.writerow([person_id, person_name, photo_path])
    return True


def start(
    person_id, person_name, capture_count=50, delay=0.1, print_web=lambda x: print(x)
):
    face_counter = 0
    frame_counter = 0
    skip_frames = 3
    last_time = 0

    print_web("Loading model, please wait...")

    # Load YOLO face detection model
    facemodel = YOLO("yolov8n-face.pt")

    # Open video stream
    rtsp_url = (
        "rtsp://admin:123admin@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0"
    )
    if USE_BUILD_IN_CAMERA:
        cap = VideoStream(0).start()
    else:
        cap = VideoStream(rtsp_url).start()

    # Create a subdirectory inside captured_images with name_id format
    name_id = f"{person_name}_{person_id}"
    photo_path = (
        f"/file/{save_dir}/{person_name}_{person_id}/{person_name}_{person_id}_1.jpg"
    )
    person_save_dir = os.path.join(save_dir, name_id)
    os.makedirs(person_save_dir, exist_ok=True)

    # Save ID and name to CSV
    if not save_to_csv(person_id, person_name, photo_path):
        print_web(f"Error: ID {person_id} already exists!")
        cap.release()
        cv2.destroyAllWindows()
        return

    print_web(f"Capturing {capture_count} faces for {person_name} (ID: {person_id})...")

    while True:
        video = cap.read()
        if video is None:
            print_web("Error: Unable to read frame from video stream.")
            break

        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue

        # Resize for display
        video = cv2.resize(video, (600, 400))

        # Copy frame before drawing bounding boxes
        video_copy = video.copy()

        # Run YOLO inference for face detection
        face_result = facemodel.predict(video, conf=0.60, imgsz=224)

        # Draw bounding boxes for detected faces and save images
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(video, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Capture face if we haven't reached the desired count yet
                if face_counter < capture_count and time.time() > last_time + delay:
                    face = video_copy[
                        y1:y2, x1:x2
                    ]  # Use clean frame without rectangles
                    if face.size == 0:  # Avoid saving empty images
                        continue
                    face_resized = cv2.resize(face, (160, 160))
                    face_filename = f"a_{person_id}_{face_counter + 1}.jpg"
                    save_path = os.path.join(person_save_dir, face_filename)
                    cv2.imwrite(save_path, face_resized)
                    face_counter += 1
                    print_web(f"Captured face {face_counter}/{capture_count}")
                    last_time = time.time()

        # Display the frame
        # cv2.imshow('Face Detection', video)
        _, buffer = cv2.imencode(".jpg", video)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        print_web(json.dumps({"cam_frame": jpg_as_text}))

        # Exit if all faces are captured
        if face_counter >= capture_count:
            print_web(
                f"Completed capturing {capture_count} faces for {person_name}, camera stopped."
            )
            cap.stop()
            break

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print_web("Capture interrupted by user, camera stopped.")
            cap.stop()
            break
