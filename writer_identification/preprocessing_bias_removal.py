"""
Text-Dependent Offline Writer Identification 시스템을 위한
데이터 전처리 파이프라인: 정렬 편향성 제거 및 증강
"""
import cv2
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def remove_background_noise(image_path, target_rgb=(202, 235, 253), threshold=30):
    """
    Remove background noise from scanned images and convert to binary format.
    Creates black text on white background.

    Args:
        image_path (str): Path to the input image
        target_rgb (tuple): RGB values of background noise to remove
        threshold (int): Threshold for color similarity

    Returns:
        numpy.ndarray: Binarized image with background removed (white bg, black text)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply binary threshold to create binary image
    # THRESH_BINARY_INV: Values below threshold become 255 (white), above become 0 (black)
    # This creates a binary image with black text on white background (as needed for handwriting)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary


def find_stroke_bounding_box(binary_image):
    """
    Find the bounding box of stroke pixels (black pixels) in the binary image.
    
    Args:
        binary_image (numpy.ndarray): Binary image (0 for stroke, 255 for background)
    
    Returns:
        tuple: (x, y, w, h) of the bounding box, or None if no stroke found
    """
    # Find coordinates of all black pixels (stroke pixels)
    stroke_coords = np.column_stack(np.where(binary_image == 0))  # Rows, cols
    
    if stroke_coords.size == 0:
        return None  # No stroke pixels found
    
    # Get min/max coordinates
    y_min = np.min(stroke_coords[:, 0])
    y_max = np.max(stroke_coords[:, 0])
    x_min = np.min(stroke_coords[:, 1])
    x_max = np.max(stroke_coords[:, 1])
    
    # Calculate width and height
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return x_min, y_min, w, h


def extract_stroke_region(binary_image, bbox):
    """
    Extract the stroke region defined by the bounding box.
    
    Args:
        binary_image (numpy.ndarray): Binary image
        bbox (tuple): (x, y, w, h) of the bounding box
    
    Returns:
        numpy.ndarray: Extracted stroke region
    """
    x, y, w, h = bbox
    return binary_image[y:y+h, x:x+w]


def place_stroke_randomly(canvas_size, stroke_region, original_bbox):
    """
    Place the stroke region randomly within the canvas while avoiding boundary issues.
    
    Args:
        canvas_size (tuple): (width, height) of the canvas
        stroke_region (numpy.ndarray): The stroke region to place
        original_bbox (tuple): Original bounding box (x, y, w, h)
    
    Returns:
        numpy.ndarray: Canvas with randomly placed stroke
    """
    canvas_w, canvas_h = canvas_size
    stroke_h, stroke_w = stroke_region.shape
    
    # Calculate maximum possible random offsets to avoid boundary issues
    max_x_offset = max(0, canvas_w - stroke_w)
    max_y_offset = max(0, canvas_h - stroke_h)
    
    # Calculate a reasonable range for random placement
    # Ensure the stroke doesn't get placed too close to edges in a way that would be unnatural
    margin_x = min(stroke_w // 4, max_x_offset // 4) if max_x_offset > 0 else 0
    margin_y = min(stroke_h // 4, max_y_offset // 4) if max_y_offset > 0 else 0
    
    # Randomly select new position within valid range
    x_offset = random.randint(margin_x, max(max_x_offset - margin_x, margin_x)) if max_x_offset > 0 else 0
    y_offset = random.randint(margin_y, max(max_y_offset - margin_y, margin_y)) if max_y_offset > 0 else 0
    
    # Create a new canvas with white background
    new_canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    
    # Place the stroke region at the new random position
    new_canvas[y_offset:y_offset + stroke_h, x_offset:x_offset + stroke_w] = stroke_region
    
    return new_canvas


def apply_data_augmentation(image):
    """
    Apply data augmentation with geometric transforms and elastic deformation.

    Args:
        image (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Augmented image
    """
    # Define transforms with milder rotation
    transform = A.Compose([
        A.Rotate(limit=3, p=0.7),  # Small rotation: ±3°
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=3, p=0.7),  # Small scaling: ±5%
        A.ElasticTransform(alpha=1, sigma=50, p=0.6),  # Elastic deformation for stroke distortion
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),  # Mild grid distortion
        A.GaussNoise(var_limit=(10, 30), p=0.2),  # Add Gaussian noise
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        A.Blur(blur_limit=2, p=0.1),
    ])

    # Apply transforms
    augmented = transform(image=image)
    return augmented['image']


def preprocess_image_with_bias_removal(image_path, canvas_size=(300, 64)):
    """
    Complete preprocessing pipeline: bias removal -> random placement -> augmentation.
    
    Args:
        image_path (str): Path to input image
        canvas_size (tuple): (width, height) of the canvas
    
    Returns:
        numpy.ndarray: Preprocessed and augmented image
    """
    # Step A.1: Remove background noise and binarization
    binary_img = remove_background_noise(image_path)
    
    # Step A.2: Find stroke bounding box
    stroke_bbox = find_stroke_bounding_box(binary_img)
    if stroke_bbox is None:
        print(f"Warning: No stroke found in {image_path}, returning original binary image")
        # Resize to canvas size if no stroke found
        resized = cv2.resize(binary_img, canvas_size, interpolation=cv2.INTER_AREA)
        return resized
    
    # Step A.3: Extract stroke region
    stroke_region = extract_stroke_region(binary_img, stroke_bbox)
    
    # Step A.4: Place stroke randomly within canvas
    canvas_img = place_stroke_randomly(canvas_size, stroke_region, stroke_bbox)
    
    # Step B: Apply data augmentation
    augmented_img = apply_data_augmentation(canvas_img)
    
    return augmented_img


def preprocess_dataset_with_bias_removal(data_dir, output_dir, canvas_size=(300, 64), num_augmentations_per_image=3):
    """
    Process entire dataset by applying bias removal preprocessing to all images.
    
    Args:
        data_dir (str): Directory containing class subdirectories (H-, M-, Z-)
        output_dir (str): Directory to save processed images
        canvas_size (tuple): Size of the canvas 
        num_augmentations_per_image (int): Number of augmented versions to create per original image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_dir in class_dirs:
        input_class_path = os.path.join(data_dir, class_dir)
        output_class_path = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Process all images in the class directory
        for img_file in os.listdir(input_class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_img_path = os.path.join(input_class_path, img_file)
                
                try:
                    # Generate multiple augmented versions of each image
                    for aug_idx in range(num_augmentations_per_image):
                        processed_img = preprocess_image_with_bias_removal(input_img_path, canvas_size)
                        
                        # Create output filename with augmentation index
                        name, ext = os.path.splitext(img_file)
                        output_img_name = f"{name}_aug_{aug_idx}{ext}"
                        output_img_path = os.path.join(output_class_path, output_img_name)
                        
                        cv2.imwrite(output_img_path, processed_img)
                        print(f"Processed: {input_img_path} -> {output_img_path} (Aug #{aug_idx+1})")
                except Exception as e:
                    print(f"Error processing {input_img_path}: {str(e)}")


def normalize_for_model_input(image, target_size=(224, 224)):
    """
    Resize and normalize image for model input.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for model input (224, 224 for AlexNet)
    
    Returns:
        numpy.ndarray: Normalized image ready for model input
    """
    # Resize to target size
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to 3 channels for RGB
    if len(resized.shape) == 2:
        resized = np.stack([resized] * 3, axis=-1)
    
    # Normalize to [0, 1] then apply ImageNet normalization
    resized = resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization (mean and std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized = (resized - mean) / std
    
    # Convert to tensor format (H, W, C) -> (C, H, W)
    normalized = np.transpose(normalized, (2, 0, 1))
    
    return normalized


if __name__ == "__main__":
    # Example usage
    input_data_dir = "/home/car/mv/cut_parts/splitREc"
    output_data_dir = "/home/car/mv/writer_identification/processed_data_bias_removed"
    
    print("Starting data preprocessing with bias removal...")
    preprocess_dataset_with_bias_removal(
        input_data_dir, 
        output_data_dir, 
        canvas_size=(300, 64),  # Use appropriate canvas size
        num_augmentations_per_image=3  # Create 3 augmented versions per original image
    )
    print("Preprocessing with bias removal completed!")
    
    # Example of how to normalize a single image for model input
    sample_path = os.path.join(input_data_dir, "H-", "rectangle_10.png")
    if os.path.exists(sample_path):
        bias_removed_img = preprocess_image_with_bias_removal(sample_path)
        normalized_for_model = normalize_for_model_input(bias_removed_img)
        print(f"Sample image shape after preprocessing and normalization: {normalized_for_model.shape}")