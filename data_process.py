'''
import cv2
import os

# Folder path
input_folder = "captured_images1/shahin_101"
output_folder = "captured_images1/shahin_101_cropped"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define crop margins (adjust as needed)
crop_margin = 4  # Crop 10 pixels from each side

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Add more formats if needed
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            h, w = img.shape[:2]
            cropped_img = img[crop_margin:h-crop_margin, crop_margin:w-crop_margin]

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped and saved: {output_path}")

print("Cropping complete!")
'''
































'''
import cv2
import os

# Root folder path
root_folder = "captured_images1"

# Define crop margins (adjust as needed)
crop_margin = 4  # Crop 4 pixels from each side

# Process each folder in captured_images1
for folder_name in os.listdir(root_folder):
    input_folder = os.path.join(root_folder, folder_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(input_folder):
        continue
    
    print(f"Processing folder: {input_folder}")

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Case-insensitive check
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                h, w = img.shape[:2]
                # Ensure there's enough space to crop
                if h > crop_margin * 2 and w > crop_margin * 2:
                    cropped_img = img[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
                    
                    # Save the cropped image back to the same folder (overwriting original)
                    output_path = os.path.join(input_folder, filename)
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Cropped and saved: {output_path}")
                else:
                    print(f"Skipping '{filename}': Image too small to crop with margin {crop_margin}")
            else:
                print(f"Failed to load image: {img_path}")

print("Cropping complete for all folders!")
'''





































import os

def rename_files(source_dir="captured_images1"):
    """
    Rename files from 'ID_Name_Number.jpg' to 'Name_ID_Number.jpg' within each folder.
    """
    if not os.path.exists(source_dir):
        print(f"Directory '{source_dir}' does not exist.")
        return

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Extract Name and ID from folder name (e.g., 'Babu_110')
        if "_" not in folder_name:
            print(f"Skipping '{folder_name}': Invalid format (expected 'Name_ID').")
            continue

        name_part, id_part = folder_name.split("_")
        try:
            id_num = int(id_part)
        except ValueError:
            print(f"Skipping '{folder_name}': ID '{id_part}' is not a number.")
            continue

        # Process files in the folder
        for filename in os.listdir(folder_path):
            old_file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(old_file_path) or not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # Parse current filename (e.g., '110_Babu_1.jpg')
            filename_parts = filename.split("_")
            if len(filename_parts) < 3:
                print(f"Skipping '{filename}': Incorrect format (expected 'ID_Name_Number.ext').")
                continue

            old_id, old_name, num_ext = filename_parts[0], filename_parts[1], filename_parts[2]
            file_num, file_ext = os.path.splitext(num_ext)[0], os.path.splitext(num_ext)[1]

            # Verify ID consistency with folder
            if old_id != id_part:
                print(f"Warning: File ID '{old_id}' does not match folder ID '{id_part}' in '{filename}'.")

            # New filename (e.g., 'Babu_110_1.jpg')
            new_filename = f"{old_name}_{old_id}_{file_num}{file_ext}"
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed file: '{filename}' -> '{new_filename}'")

if __name__ == "__main__":
    rename_files()

