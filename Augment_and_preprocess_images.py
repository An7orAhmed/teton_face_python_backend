import os
import cv2
import numpy as np
import albumentations as A

def augment_preprocess():
    input_dir = 'captured_images'  # Updated to match your directory
    output_dir = 'augmented_images'
    os.makedirs(output_dir, exist_ok=True)

    # Define mean and std range
    mean_range = (0, 0.25)
    std_range = (0.1, 0.2)

    # Function to sample random mean and std
    def sample_gaussian_params():
        mean = np.random.uniform(*mean_range)
        std = np.random.uniform(*std_range)
        return mean, std

    # Custom transform to apply GaussNoise with dynamic mean and std
    class RandomGaussNoise(A.ImageOnlyTransform):
        def __init__(self, always_apply=False, p=0.2):
            super().__init__(always_apply, p)

        def apply(self, img, **params):
            mean, std = sample_gaussian_params()
            return A.GaussNoise(var_limit=(std ** 2, std ** 2), mean=mean, per_channel=True, p=1.0)(image=img)['image']

    # Define the augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.Rotate(limit=20, p=0.3),
        A.HorizontalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), contrast_limit=(-0.3, 0.2), p=0.5),
        RandomGaussNoise(p=0.2),
        A.Blur(blur_limit=(1, 7), p=0.25),
        A.CLAHE(clip_limit=1.0, p=0.2),
        A.Affine(shear=0.3, p=0.25),
        A.RandomScale(scale_limit=0.2, p=0.2),
        A.Resize(100, 100)
    ])

    total_images = 0
    augmented_count = 0

    # Check if input directory exists and list its contents
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
    print(f"Scanning directory: {input_dir}")
    print(f"Subdirectories found: {os.listdir(input_dir)}")

    # Iterate through subdirectories in captured_images1 (e.g., shahin_101)
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Skipping non-directory: {subdir}")
            continue

        print(f"Processing subdirectory: {subdir}")
        print(f"Files in {subdir}: {os.listdir(subdir_path)}")

        # Create corresponding subdirectory in augmented_images
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Get list of existing images in the input subdirectory
        existing_images = set(os.listdir(subdir_path))

        # Clean up unused augmented images in the output subdirectory
        augmented_images = os.listdir(output_subdir)
        for augmented_file in augmented_images:
            base_identifier = '_'.join(augmented_file.split('_')[:-2])  # e.g., Shahin_101
            matching_files = [f for f in existing_images if base_identifier in f]
            if not matching_files:
                os.remove(os.path.join(output_subdir, augmented_file))
                print(f"Deleted unused augmented file: {augmented_file}")

        # Process each image in the subdirectory
        for img_file in os.listdir(subdir_path):
            if img_file.lower().endswith(('.jpg', '.jpeg')):
                if "_aug_" in img_file:
                    print(f"Skipping already augmented image: {img_file}")
                    continue

                total_images += 1

                # Extract identifier from the filename (e.g., Shahin_101_1.jpg -> Shahin_101_1)
                parts = img_file.split('_')
                if len(parts) >= 3:
                    identifier = f"{parts[0]}_{parts[1]}_{parts[2].split('.')[0]}"  # e.g., Shahin_101_1
                else:
                    identifier = os.path.splitext(img_file)[0]

                img_path = os.path.join(subdir_path, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Could not read image: {img_path}")
                    continue

                # Generate 5 augmented images
                for i in range(1, 31):
                    try:
                        augmented = augmentation_pipeline(image=image)
                        augmented_image = augmented['image']

                        # Convert from RGB to BGR before saving
                        #augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

                        # Output format: Shahin_101_1_aug_N.jpg
                        output_file_name = f"{identifier}_aug_{i}.jpg"
                        output_path = os.path.join(output_subdir, output_file_name)
                        cv2.imwrite(output_path, augmented_image)
                        augmented_count += 1
                    except Exception as e:
                        print(f"Error augmenting {img_file}: {e}")
                        continue

                print(f"Generated 5 augmented images for: {identifier}")

    print(f"Total input images processed: {total_images}")
    print(f"Total augmented images generated: {augmented_count}")
    print("Augmentation and preprocessing complete.")

augment_preprocess()

