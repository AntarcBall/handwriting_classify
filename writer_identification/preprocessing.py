import cv2
import numpy as np
from PIL import Image
import os


def remove_background_noise(image_path, target_rgb=(202, 235, 253), threshold=30):
    """
    Remove background noise from scanned images and convert to binary format.
    
    Args:
        image_path (str): Path to the input image
        target_rgb (tuple): RGB values of background noise to remove
        threshold (int): Threshold for color similarity
    
    Returns:
        numpy.ndarray: Binarized image with background removed
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define the target color range for background removal
    lower_bound = np.array([max(0, val - threshold) for val in target_rgb])
    upper_bound = np.array([min(255, val + threshold) for val in target_rgb])
    
    # Create mask for background pixels
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply binary threshold (convert to black and white)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Apply the mask to remove background noise
    # White areas in the mask will be set to white (255) in the binary image
    binary[mask > 0] = 255
    
    # Invert if needed so that handwriting is white on black background
    # (This may depend on your specific data - adjust as needed)
    # Uncomment the next line if you want handwriting to be white on black:
    # binary = 255 - binary
    
    return binary


def normalize_and_pad_image(image, target_width=None, target_height=64):
    """
    Normalize and pad images to maintain aspect ratio while having consistent dimensions.
    
    Args:
        image (numpy.ndarray): Input image
        target_width (int): Target width (if None, will be calculated based on aspect ratio)
        target_height (int): Target height (default 64)
    
    Returns:
        numpy.ndarray: Padded and normalized image
    """
    h, w = image.shape
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    
    if target_width is None:
        target_width = new_width
    
    # Resize image while maintaining aspect ratio
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Create a canvas of target dimensions
    padded = np.full((target_height, target_width), 255, dtype=np.uint8)  # White background
    
    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    padded[:, x_offset:x_offset + new_width] = resized
    
    return padded


def preprocess_image(image_path, target_width=None, target_height=64):
    """
    Complete preprocessing pipeline: noise removal -> normalization -> padding
    
    Args:
        image_path (str): Path to input image
        target_width (int): Target width of the final image
        target_height (int): Target height of the final image
    
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Step 1: Remove background noise and binarize
    binary_img = remove_background_noise(image_path)
    
    # Step 2: Normalize and pad
    processed_img = normalize_and_pad_image(binary_img, target_width, target_height)
    
    return processed_img


def process_dataset(data_dir, output_dir, target_height=64):
    """
    Process entire dataset by applying preprocessing pipeline to all images.
    
    Args:
        data_dir (str): Directory containing class subdirectories (H-, M-, Z-)
        output_dir (str): Directory to save processed images
        target_height (int): Target height for normalization
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
                output_img_path = os.path.join(output_class_path, img_file)
                
                try:
                    processed_img = preprocess_image(input_img_path, target_height=target_height)
                    cv2.imwrite(output_img_path, processed_img)
                    print(f"Processed: {input_img_path} -> {output_img_path}")
                except Exception as e:
                    print(f"Error processing {input_img_path}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    input_data_dir = "/home/car/mv/cut_parts/splitREc"
    output_data_dir = "/home/car/mv/writer_identification/processed_data"
    
    print("Starting data preprocessing...")
    process_dataset(input_data_dir, output_data_dir)
    print("Preprocessing completed!")