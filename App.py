from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from flask_cors import CORS
import csv
import Capture
import Prediction
import Train
import os

DATA_CSV = "data.csv"
ATTENDANCE_DIR = "Attendance"
PROFILE_PHOTO_DIR = "profile_photo"
TRAIN_DIR = "captured_images"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

capture_running = False
recognition_running = False

def print_web(msg):
    socketio.emit("status_update", {"message": msg})

# send images to frontend
@app.route("/file/<base_folder>/<path:file_path>")
def serve_file(base_folder, file_path):
    try:
        directory = os.path.join(os.getcwd(), base_folder, os.path.dirname(file_path))
        filename = os.path.basename(file_path)

        return send_from_directory(directory, filename)
    except FileNotFoundError:
        os.abort(404, description="File not found")

# Add face data
@app.route("/add_face", methods=["POST"])
def add_face():
    global capture_running, recognition_running
    if capture_running:
        return jsonify({"error": "Capture is running"}), 400
    if recognition_running:
        return jsonify({"error": "Recognition is running"}), 400
    try:
        unique_id = request.form.get("uniqueId")
        name = request.form.get("name")
        delay = int(request.form.get("captureDelay")) / 1000.0
        frame = int(request.form.get("captureCount"))
        photo = request.files.get("photo")

        if not all([unique_id, name, delay, frame, photo]):
            return jsonify({"error": "Missing required fields"}), 400
        
        if os.path.exists(DATA_CSV):
            with open(DATA_CSV, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) > 1 and str(unique_id) == row[0]:
                        return jsonify({"error": "ID already in database."}), 400

        # Make a folder for this user
        folder_name = f"{name}_{unique_id}"
        user_folder = os.path.join(TRAIN_DIR, folder_name)
        os.makedirs(user_folder, exist_ok=True)

        # Save the uploaded photo
        ext = os.path.splitext(photo.filename)[1]
        filename = secure_filename(f"{unique_id}_{name}{ext}")
        file_path = os.path.join(PROFILE_PHOTO_DIR, filename)
        photo.save(file_path)

        print_web("Capturing started...")
        capture_running = True
        Capture.start(unique_id, name, frame, delay, print_web)
        capture_running = False

        return jsonify({"message": "Face data saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Train model
@app.route("/start_training")
def train_model():
    global capture_running, recognition_running
    if capture_running:
        return jsonify({"error": "Capture is running"}), 400
    if recognition_running:
        return jsonify({"error": "Recognition is running"}), 400
    Train.start(print_web)
    return jsonify({"status": "started"})

# Start recognition
@app.route("/start_recognition")
def start_recognition():
    global recognition_running, capture_running
    if capture_running:
        return jsonify({"error": "Capture is running"}), 400
    if recognition_running:
        return jsonify({"error": "Recognition is already running"}), 400
    recognition_running = True
    Prediction.run(print_web)
    return jsonify({"status": "started"})

# Stop recognition
@app.route("/stop_recognition")
def stop_recognition():
    global recognition_running, capture_running
    if capture_running:
        return jsonify({"error": "Capture is running"}), 400
    if not recognition_running:
        return jsonify({"error": "Recognition is not running"}), 400
    recognition_running = False
    Prediction.close(print_web)
    return jsonify({"status": "stopped"})

# get all faces
@app.route("/faces")
def get_faces():
    data = []
    try:
        with open(DATA_CSV, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:
                    record = {"ID": row[0], "Name": row[1], "Photo": row[2]}
                    data.append(record)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "data.csv not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# get statistic
@app.route("/stats")
def get_stats():
    base_folder = os.path.join(os.getcwd(), "captured_images")
    if not os.path.exists(base_folder):
        return jsonify({"error": "Folder not found"}), 404

    folder_count = 0
    jpg_count = 0

    for root, dirs, files in os.walk(base_folder):
        if root == base_folder:
            folder_count += len(dirs)
        jpg_count += sum(1 for file in files if file.lower().endswith(".jpg"))

    return jsonify(
        {
            "trainedCount": folder_count,
            "totalImages": jpg_count,
            "isCapturing": capture_running,
            "isRecognizing": recognition_running,
        }
    )


# get recognized record
@app.route("/recognized")
def get_recognized():
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "Missing date parameter"}), 400

    file_path = os.path.join(ATTENDANCE_DIR, f"{date}.csv")
    if not os.path.exists(file_path):
        return jsonify({'error': 'no data found'}) 

    recognized = []
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row["Name"]
            id_ = row["ID"]
            photo = f"/{PROFILE_PHOTO_DIR}/{id_}_{name}.jpg"
            recognized.append({
                "name": name,
                "id": id_,
                "inTime": row["InTime"],
                "outTime": row["OutTime"],
                "photo": photo
            })

    return jsonify(recognized)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
