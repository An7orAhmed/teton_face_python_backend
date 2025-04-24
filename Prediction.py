import json
import cv2
import os
import base64
import pandas as pd
import datetime
from ultralytics import YOLO
import numpy as np
import threading
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import faiss
import queue
import time
from collections import defaultdict
from imutils.video import VideoStream

USE_BUILD_IN_CAMERA = True
DISPLAY_ENABLED = False

running_flag = False
thread = None

# Variables for debounce control
last_push_time = 0
debounce_delay = 1  # Used for web push
unknown_debounce_delay = 2  # For unknown face saving
debounce_timer = None
last_detected_name = None
last_unknown_save_time = defaultdict(float)  # Track last save time for unknown faces

# Configure device
device = torch.device('cpu')
print(f"Using device: {device}")

# Load precomputed ArcFace embeddings and IDs
train_embeddings = np.load("train_embeddings.npy")
train_Ids = np.load("train_Ids.npy")
print("Precomputed ArcFace embeddings and IDs loaded.")

# Convert embeddings to float32 for faiss and normalize
train_embeddings = train_embeddings.astype('float32')
train_embeddings_normed = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

# Build faiss index
embedding_dim = train_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(train_embeddings_normed)
print("Faiss index built for similarity search.")

# Initialize models
arcface_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
yolo_model = YOLO('yolov8n-face.pt').to(device)

def web_push_notification(print_web, Id, name, frame, force_push=False):
    """Push to web with debounce delay, resetting if ID changes or force_push is True."""
    global last_push_time, last_detected_name, debounce_timer

    def send():
        global last_push_time
        last_push_time = time.time()
        ct = datetime.datetime.now().strftime('%H:%M:%S')
        cd = datetime.datetime.now().strftime('%Y-%m-%d')
        # Save to known_faces/date/images/ for recognized faces
        if Id != "unknown":
            img_dir = f"known_faces/{cd}"
            os.makedirs(img_dir, exist_ok=True)
            img_path = f"{img_dir}/{Id}_{name}.jpg"
            cv2.imwrite(img_path, frame)
        else:
            # Unknown faces handled separately with debounce
            img_dir = f"unknown_faces/{cd}"
            os.makedirs(img_dir, exist_ok=True)
            img_path = f"{img_dir}/unknown_{int(last_push_time)}.jpg"
            if time.time() - last_unknown_save_time[img_path] >= unknown_debounce_delay:
                cv2.imwrite(img_path, frame)
                last_unknown_save_time[img_path] = time.time()
                print(f"Unknown face saved to: {img_path}")
            else:
                print(f"Unknown face save skipped due to debounce: {img_path}")
                return  # Skip notification if image not saved

        record = {"type": "FACE_DETECT", "id": Id, "name": name, "img_path": img_path, "time": ct, "date": cd}
        print_web(json.dumps(record))
        print(f"Web notification sent for ID: {Id}, Name: {name}")

    # Reset debounce if name changes or force_push is True
    if last_detected_name != name or force_push:
        last_detected_name = name
        last_push_time = 0
        send()  # Send immediately for new detections
    elif time.time() - last_push_time >= debounce_delay:
        send()
    else:
        if debounce_timer:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(debounce_delay, send)
        debounce_timer.start()
        print(f"Notification for ID: {Id} delayed by {debounce_delay} seconds.")

def start(print_web):
    global running_flag

    # Attendance tracking
    col_names = ['Name', 'ID', 'InTime', 'OutTime']
    attendance_list = []
    attended_ids = set()
    attendance_queue = queue.Queue()
    last_save_time = time.time()

    # Load dataset
    dataset = pd.read_csv("data.csv")
    dataset['ID'] = dataset['ID'].astype(int)
    id_to_name = dict(zip(dataset['ID'], dataset['Name']))
    print("Dataset loaded and ID-to-name dictionary created.")

    # Voting system variables
    current_votes = defaultdict(int)
    last_vote_process_time = time.time()
    vote_threshold = 1

    def save_attendance_worker():
        while True:
            try:
                data = attendance_queue.get(timeout=60)
                if data is None:
                    break
                temp_list, file_path = data
                if temp_list:
                    temp_df = pd.DataFrame(temp_list, columns=col_names)
                    temp_df['ID'] = temp_df['ID'].astype(int)
                    if os.path.exists(file_path):
                        existing = pd.read_csv(file_path)
                        # Update Out Time for existing IDs
                        for _, row in temp_df.iterrows():
                            if row['ID'] in existing['ID'].values:
                                existing.loc[existing['ID'] == row['ID'], 'Out Time'] = row['Out Time']
                            else:
                                existing = pd.concat([existing, row.to_frame().T], ignore_index=True)
                        temp_df = existing
                    temp_df.to_csv(file_path, index=False)
                    print(f"Attendance saved to: {file_path}")
                attendance_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving attendance: {e}")

    attendance_saver = threading.Thread(target=save_attendance_worker, daemon=True)
    attendance_saver.start()

    # Initialize RTSP video stream
    rtsp_url = 'rtsp://admin:123admin@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0'
    if USE_BUILD_IN_CAMERA:
        cap = VideoStream(0).start()
    else:
        cap = VideoStream(rtsp_url).start()
    time.sleep(2.0)
    print("RTSP stream initialized.")

    if DISPLAY_ENABLED:
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', 800, 600)

    frame_skip = 0
    frame_counter = 0
    yolo_conf = 0.6

    def extract_embeddings(face_images):
        try:
            batch_tensors = []
            for face_img in face_images:
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_img_rgb)
                face_resized = face_pil.resize((160, 160))
                face_tensor = torch.tensor(np.array(face_resized)).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(face_tensor)
            batch_tensors = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = arcface_model(batch_tensors).detach().cpu().numpy()
            print(f"Embeddings generated for {len(face_images)} faces.")
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return None

    def predict_with_arcface_faiss(embedding, index, train_Ids):
        if embedding is None:
            return None, None, None
        embedding = embedding.astype('float32')
        embedding_normed = embedding / np.linalg.norm(embedding)
        embedding_normed = embedding_normed.reshape(1, -1)
        distances, indices = index.search(embedding_normed, 1)
        max_similarity = distances[0][0]
        max_similarity_idx = indices[0][0]
        predicted_id = train_Ids[max_similarity_idx]
        return predicted_id, max_similarity, None

    while running_flag:
        frame = cap.read()
        if frame is None:
            print("Error: Failed to capture frame. Check RTSP stream connection.")
            break
        print("Frame captured successfully.")

        frame_counter += 1
        if frame_counter % (frame_skip + 1) != 0:
            continue

        processing_frame = cv2.resize(frame, (720, 480))
        display_frame = processing_frame.copy() if DISPLAY_ENABLED else None

        results = yolo_model(processing_frame, imgsz=160, conf=yolo_conf, verbose=False)
        print(f"YOLO detection completed, boxes found: {len(results[0].boxes)}")

        face_images = []
        face_coords = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            face_img = processing_frame[y1:y2, x1:x2]
            face_images.append(face_img)
            face_coords.append((x1, y1, x2, y2))

        if face_images:
            print(f"Number of faces detected: {len(face_images)}")
            embeddings = extract_embeddings(face_images)
            if embeddings is None:
                continue
            for i, embedding in enumerate(embeddings):
                x1, y1, x2, y2 = face_coords[i]
                predicted_id, max_similarity, _ = predict_with_arcface_faiss(embedding, index, train_Ids)
                if predicted_id is None:
                    continue

                if 0.7 <= max_similarity <= 1.0:
                    Id = int(predicted_id)
                    name = id_to_name.get(Id, "Unknown")
                    vote_key = (Id, x1, y1)
                    current_votes[vote_key] += 1
                    print(f"Vote added for {vote_key}, current votes: {current_votes[vote_key]}, Similarity: {max_similarity:.4f}")
                    display_name = f"{name} | {Id}"
                    print_name = str(Id)
                    color = (0, 255, 0)

                elif max_similarity <= 0.4:
                    Id = "unknown"
                    name = "Unknown"
                    vote_key = (Id, x1, y1)
                    current_votes[vote_key] += 1
                    print(f"Vote added for {vote_key}, current votes: {current_votes[vote_key]}, Similarity: {max_similarity:.4f}")
                    display_name = "Unknown"
                    print_name = "unknown"
                    color = (0, 0, 255)

                else:
                    display_name = ""
                    print_name = "null"
                    color = (0, 255, 255)

                print(f"Max Similarity: {max_similarity:.4f}, Predicted ID: {print_name}")

                if DISPLAY_ENABLED:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
                    if display_name:
                        cv2.putText(display_frame, f"{display_name}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

                # Push web notification immediately when threshold is first met
                if (0.7 <= max_similarity <= 1.0 or max_similarity <= 0.4) and current_votes[vote_key] == vote_threshold:
                    web_push_notification(print_web, Id, name, frame, force_push=True)

        current_time = time.time()
        if current_time - last_vote_process_time >= 2.0:
            print(f"Processing votes (2 sec window): {dict(current_votes)}")
            for vote_key, count in current_votes.items():
                id = vote_key[0]
                if count >= vote_threshold and id not in attended_ids:
                    name = id_to_name.get(id, "Unknown")
                    current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
                    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                    # Set both In Time and Out Time to current time for new entries
                    attendance_list.append([name, id, current_time_str, current_time_str])
                    attended_ids.add(id)
                    print(f"Attendance marked for: {name} (ID: {id}) with {count} votes")
                elif count < vote_threshold:
                    print(f"Ignoring {vote_key} with {count} votes (below threshold {vote_threshold})")
            current_votes.clear()
            last_vote_process_time = current_time

        current_save_time = time.time()
        if attendance_list and (current_save_time - last_save_time >= debounce_delay):
            file_path = f"Attendance/{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            print(f"Attempting to save attendance to: {file_path}")
            os.makedirs("Attendance", exist_ok=True)
            temp_list = attendance_list.copy()
            attendance_list.clear()
            attendance_queue.put((temp_list, file_path))
            last_save_time = current_save_time

        _, buffer = cv2.imencode(".jpg", processing_frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        print_web(json.dumps({"cam_frame": jpg_as_text}))

        if DISPLAY_ENABLED:
            cv2.imshow('Face Recognition', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if attendance_list:
        file_path = f"Attendance/{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
        print(f"Final save attempt to: {file_path}")
        os.makedirs("Attendance", exist_ok=True)
        attendance_queue.put((attendance_list, file_path))

    attendance_queue.put(None)
    attendance_saver.join()
    cap.stop()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()

def run(print_web):
    global thread, running_flag
    if not running_flag:
        running_flag = True
        thread = threading.Thread(target=start, args=[print_web])
        thread.start()
        print_web("Camera started.")

def close(print_web):
    global thread, running_flag
    if running_flag or thread:
        running_flag = False
        if thread:
            thread.join()
        print_web("Camera stopped.")