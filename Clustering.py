# DBSCAN clustering and visualization script
import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.cluster import DBSCAN
import shutil
import cv2
from ultralytics import YOLO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

# Initialize ArcFace model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Initialize YOLO for face detection (optional)
yolo_model = YOLO("yolov8n-face.pt").to(device)

# Directory containing the images
image_dir = "Previous Iamages/captured_images/unknown"
output_base_dir = "Previous Iamages/captured_images/clustered_unknown"


# Function to detect and crop face
def detect_face(image_path):
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb, imgsz=160, conf=0.6, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            face_img = img_rgb[y1:y2, x1:x2]
            return Image.fromarray(face_img)
        print(f"No valid face detected in {image_path}")
        return None
    except Exception as e:
        print(f"Error detecting face in {image_path}: {e}")
        return None


# Function to extract ArcFace embeddings
def extract_embeddings(image_paths, use_face_detection=True):
    embeddings = []
    valid_image_paths = []

    for image_path in image_paths:
        try:
            # Load image
            if use_face_detection:
                img = detect_face(image_path)
                if img is None:
                    continue
            else:
                img = Image.open(image_path).convert("RGB")

            # Preprocess image
            img = img.resize((160, 160))  # ArcFace input size
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Extract embedding
            embedding = arcface_model(img_tensor).detach().cpu().numpy().flatten()
            embeddings.append(embedding)
            valid_image_paths.append(image_path)
            print(f"Processed: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), valid_image_paths


# Function to compute cluster quality
def compute_cluster_quality(embeddings, labels):
    unique_labels = set(labels) - {-1}  # Exclude noise
    intra_distances = []
    inter_distances = []

    for label in unique_labels:
        cluster_embeddings = embeddings[labels == label]
        if len(cluster_embeddings) > 1:
            # Compute pairwise cosine distances
            dist_matrix = cosine_distances(cluster_embeddings)
            # Get upper triangular indices (excluding diagonal)
            i, j = np.triu_indices(len(cluster_embeddings), k=1)
            intra_dist = dist_matrix[i, j]
            if len(intra_dist) > 0:
                intra_distances.extend(intra_dist)

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels:
            if label2 <= label1:
                continue
            cluster1 = embeddings[labels == label1]
            cluster2 = embeddings[labels == label2]
            if len(cluster1) > 0 and len(cluster2) > 0:
                inter_dist = cosine_distances(cluster1, cluster2).flatten()
                inter_distances.extend(inter_dist)

    if intra_distances and inter_distances:
        print(f"Average intra-cluster distance: {np.mean(intra_distances):.4f}")
        print(f"Average inter-cluster distance: {np.mean(inter_distances):.4f}")
    else:
        print("Insufficient data to compute cluster quality.")


# Function to visualize clusters with t-SNE
def visualize_clusters(embeddings, labels, output_path="cluster_visualization.png"):
    if len(embeddings) < 2:
        print("Not enough embeddings to visualize.")
        return

    tsne = TSNE(n_components=2, metric="cosine", random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        color = "gray" if label == -1 else plt.cm.tab20(label % 20)
        label_name = "Noise" if label == -1 else f"unknown_{label + 1}"
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6,
        )

    plt.legend()
    plt.title("t-SNE Visualization of Face Clusters")
    plt.savefig(output_path)
    plt.close()
    print(f"Cluster visualization saved to {output_path}")


# Function to cluster embeddings and organize images
def cluster_images(use_face_detection=True):
    # Get list of image files
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.startswith("unknown_Unknown_") and f.endswith(".jpg")
    ]

    if not image_paths:
        print("No images found in the specified directory.")
        return

    # Extract embeddings
    embeddings, valid_image_paths = extract_embeddings(image_paths, use_face_detection)

    if len(embeddings) == 0:
        print("No valid embeddings extracted.")
        return

    # Normalize embeddings for cosine similarity
    embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Cluster using DBSCAN (cosine distance)
    clustering = DBSCAN(eps=0.2, min_samples=1, metric="cosine").fit(embeddings_normed)
    labels = clustering.labels_

    # Compute cluster quality
    compute_cluster_quality(embeddings_normed, labels)

    # Visualize clusters
    visualize_clusters(embeddings_normed, labels)

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Organize and rename images into cluster folders
    for image_path, label in zip(valid_image_paths, labels):
        # Determine cluster folder
        cluster_name = f"unknown_{label + 1}" if label != -1 else "noise"
        cluster_dir = os.path.join(output_base_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        # Generate new filename
        cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith(".jpg")]
        new_filename = f"unknown_{label + 1 if label != -1 else 'noise'}_{len(cluster_files) + 1}.jpg"
        dest_path = os.path.join(cluster_dir, new_filename)

        # Copy the image
        shutil.copy(image_path, dest_path)
        print(f"Assigned {image_path} to {cluster_name}/{new_filename}")

    # Print clustering summary
    unique_labels = set(labels)
    print(f"\nClustering completed:")
    for label in unique_labels:
        if label == -1:
            print(f"Noise: {np.sum(labels == -1)} images")
        else:
            print(f"unknown_{label + 1}: {np.sum(labels == label)} images")


# Run the clustering
if __name__ == "__main__":
    cluster_images(use_face_detection=True)


# HDBSCAN clustering and visualization script

# import os
# import numpy as np
# from PIL import Image
# import torch
# from facenet_pytorch import InceptionResnetV1
# import shutil
# import cv2
# from ultralytics import YOLO
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_distances
# from hdbscan import HDBSCAN

# # Initialize ArcFace model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# arcface_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Initialize YOLO for face detection (optional)
# yolo_model = YOLO('yolov8n-face.pt').to(device)

# # Directory containing the images
# image_dir = "Previous Iamages/captured_images/unknown"
# output_base_dir = "Previous Iamages/captured_images/clustered_unknown"

# # Function to detect and crop face
# def detect_face(image_path):
#     try:
#         img = cv2.imread(image_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = yolo_model(img_rgb, imgsz=160, conf=0.6, verbose=False)
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             if (x2 - x1) < 30 or (y2 - y1) < 30:
#                 continue
#             face_img = img_rgb[y1:y2, x1:x2]
#             return Image.fromarray(face_img)
#         print(f"No valid face detected in {image_path}")
#         return None
#     except Exception as e:
#         print(f"Error detecting face in {image_path}: {e}")
#         return None

# # Function to extract ArcFace embeddings
# def extract_embeddings(image_paths, use_face_detection=True):
#     embeddings = []
#     valid_image_paths = []

#     for image_path in image_paths:
#         try:
#             # Load image
#             if use_face_detection:
#                 img = detect_face(image_path)
#                 if img is None:
#                     continue
#             else:
#                 img = Image.open(image_path).convert('RGB')

#             # Preprocess image
#             img = img.resize((160, 160))  # ArcFace input size
#             img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
#             img_tensor = img_tensor.unsqueeze(0).to(device)

#             # Extract embedding
#             embedding = arcface_model(img_tensor).detach().cpu().numpy().flatten()
#             embeddings.append(embedding)
#             valid_image_paths.append(image_path)
#             print(f"Processed: {image_path}")
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")

#     return np.array(embeddings), valid_image_paths

# # Function to compute cluster quality (using cosine distances)
# def compute_cluster_quality(embeddings, labels):
#     unique_labels = set(labels) - {-1}  # Exclude noise
#     intra_distances = []
#     inter_distances = []

#     for label in unique_labels:
#         cluster_embeddings = embeddings[labels == label]
#         if len(cluster_embeddings) > 1:
#             dist_matrix = cosine_distances(cluster_embeddings)
#             i, j = np.triu_indices(len(cluster_embeddings), k=1)
#             intra_dist = dist_matrix[i, j]
#             if len(intra_dist) > 0:
#                 intra_distances.extend(intra_dist)

#     for i, label1 in enumerate(unique_labels):
#         for label2 in unique_labels:
#             if label2 <= label1:
#                 continue
#             cluster1 = embeddings[labels == label1]
#             cluster2 = embeddings[labels == label2]
#             if len(cluster1) > 0 and len(cluster2) > 0:
#                 inter_dist = cosine_distances(cluster1, cluster2).flatten()
#                 inter_distances.extend(inter_dist)

#     if intra_distances and inter_distances:
#         print(f"Average intra-cluster distance (cosine): {np.mean(intra_distances):.4f}")
#         print(f"Average inter-cluster distance (cosine): {np.mean(inter_distances):.4f}")
#     else:
#         print("Insufficient data to compute cluster quality.")

# # Function to visualize clusters with t-SNE
# def visualize_clusters(embeddings, labels, output_path="cluster_visualization.png"):
#     if len(embeddings) < 2:
#         print("Not enough embeddings to visualize.")
#         return

#     tsne = TSNE(n_components=2, metric='euclidean', random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings)

#     plt.figure(figsize=(10, 8))
#     unique_labels = set(labels)
#     for label in unique_labels:
#         mask = labels == label
#         color = 'gray' if label == -1 else plt.cm.tab20(label % 20)
#         label_name = 'Noise' if label == -1 else f'unknown_{label + 1}'
#         plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=[color], label=label_name, alpha=0.6)

#     plt.legend()
#     plt.title("t-SNE Visualization of Face Clusters")
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Cluster visualization saved to {output_path}")

# # Function to cluster embeddings and organize images
# def cluster_images(use_face_detection=True):
#     # Get list of image files
#     image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
#                    if f.startswith("unknown_Unknown_") and f.endswith(".jpg")]

#     if not image_paths:
#         print("No images found in the specified directory.")
#         return

#     # Extract embeddings
#     embeddings, valid_image_paths = extract_embeddings(image_paths, use_face_detection)

#     if len(embeddings) == 0:
#         print("No valid embeddings extracted.")
#         return

#     # Normalize embeddings for cosine similarity
#     embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#     # Cluster using HDBSCAN (Euclidean distance on normalized embeddings)
#     clustering = HDBSCAN(min_cluster_size=2, metric='euclidean').fit(embeddings_normed)
#     labels = clustering.labels_

#     # Compute cluster quality (using cosine distances for consistency)
#     compute_cluster_quality(embeddings_normed, labels)

#     # Visualize clusters
#     visualize_clusters(embeddings_normed, labels)

#     # Create output directory
#     os.makedirs(output_base_dir, exist_ok=True)

#     # Organize and rename images into cluster folders
#     for image_path, label in zip(valid_image_paths, labels):
#         # Determine cluster folder
#         cluster_name = f"unknown_{label + 1}" if label != -1 else "noise"
#         cluster_dir = os.path.join(output_base_dir, cluster_name)
#         os.makedirs(cluster_dir, exist_ok=True)

#         # Generate new filename
#         cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith('.jpg')]
#         new_filename = f"unknown_{label + 1 if label != -1 else 'noise'}_{len(cluster_files) + 1}.jpg"
#         dest_path = os.path.join(cluster_dir, new_filename)

#         # Copy the image
#         shutil.copy(image_path, dest_path)
#         print(f"Assigned {image_path} to {cluster_name}/{new_filename}")

#     # Print clustering summary
#     unique_labels = set(labels)
#     print(f"\nClustering completed:")
#     for label in unique_labels:
#         if label == -1:
#             print(f"Noise: {np.sum(labels == -1)} images")
#         else:
#             print(f"unknown_{label + 1}: {np.sum(labels == label)} images")

# # Run the clustering
# if __name__ == "__main__":
#     cluster_images(use_face_detection=True)
