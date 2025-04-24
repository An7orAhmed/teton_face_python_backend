import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

# Initialize ArcFace model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_model = (
    InceptionResnetV1(pretrained="vggface2").eval().to(device)
)  # Pre-trained ArcFace model


# Function to extract images and labels (updated to extract embeddings)
def getImagesAndLabels(path, print_web=None):
    """
    Reads images from subdirectories in the specified path and extracts ArcFace embeddings.
    """
    embeddings = []
    Ids = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        imagePaths = [
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if f.endswith(("jpg", "png", "jpeg"))
        ]

        for imagePath in imagePaths:
            try:
                # Load image
                img = Image.open(imagePath).convert("RGB")

                # Resize to ArcFace input size (no face detection, assuming cropped faces)
                face = img.resize((160, 160))  # Resize to match ArcFace input size
                face_tensor = (
                    torch.tensor(np.array(face)).permute(2, 0, 1).float() / 255.0
                )  # Convert to tensor and normalize
                face_tensor = face_tensor.unsqueeze(0).to(device)  # Add batch dimension

                # Extract embedding using ArcFace
                embedding = arcface_model(face_tensor).detach().cpu().numpy().flatten()
                if embedding is None:
                    if print_web:
                        print_web(f"Skipping {imagePath}: Cannot extract embedding.")
                    continue

                # Extract ID from filename
                filename = os.path.basename(imagePath)
                filename_parts = filename.split("_")

                if len(filename_parts) < 3:
                    if print_web:
                        print_web(f"Skipping {filename}: Incorrect format.")
                    continue

                try:
                    Id = int(filename_parts[1])
                except ValueError:
                    if print_web:
                        print_web(f"Skipping {filename}: ID should be a number.")
                    continue

                embeddings.append(embedding)
                Ids.append(Id)
                if print_web:
                    print_web(f"Processed: {filename} | ID: {Id}")

            except Exception as e:
                if print_web:
                    print_web(f"Error processing {imagePath}: {e}")

    return np.array(embeddings), Ids


# Function to train (store embeddings)
def start(print_web=lambda x: print(x)):
    """
    Extracts embeddings for training images using ArcFace and saves them.
    """
    training_images_path = "augmented_images"
    embeddings_file = "train_embeddings.npy"
    ids_file = "train_Ids.npy"

    # Check if the embedding files exist and delete them
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
        print_web(f"Existing embeddings file '{embeddings_file}' deleted.")
    if os.path.exists(ids_file):
        os.remove(ids_file)
        print_web(f"Existing IDs file '{ids_file}' deleted.")

    # Load images and extract embeddings
    embeddings, Ids = getImagesAndLabels(training_images_path, print_web)

    if len(embeddings) == 0 or len(Ids) == 0:
        print_web("No images found for training.")
        return None, None

    # Save embeddings and IDs
    np.save(embeddings_file, embeddings)
    np.save(ids_file, Ids)

    print_web(f"Embeddings and IDs saved as '{embeddings_file}' and '{ids_file}'")
    print_web(f"Training completed with {len(embeddings)} embeddings processed.")

    return embeddings, Ids
