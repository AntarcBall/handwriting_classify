import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing_bias_removal import preprocess_image_with_bias_removal


def visualize_bias_removal_effect():
    """
    Visualize the effect of positional bias removal by showing 
    original images vs multiple randomly positioned versions
    """
    data_dir = "/home/car/mv/cut_parts/splitREc"
    
    # Find a sample image from each class
    sample_images = {}
    for class_dir in ['H-', 'M-', 'Z-']:
        class_path = os.path.join(data_dir, class_dir)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_images[class_dir] = os.path.join(class_path, img_file)
                    break
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for row, (class_dir, img_path) in enumerate(sample_images.items()):
        # Load original image
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Show original in first column
        axes[row, 0].imshow(original, cmap='gray')
        axes[row, 0].set_title(f'Original\n{class_dir}', fontsize=10)
        axes[row, 0].axis('off')
        
        # Create 4 different randomly positioned versions
        for col in range(1, 5):
            processed_img = preprocess_image_with_bias_removal(img_path, canvas_size=(300, 64))
            
            axes[row, col].imshow(processed_img, cmap='gray')
            axes[row, col].set_title(f'Random Position #{col}\n{class_dir}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    output_path = "positional_bias_removal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Positional bias removal visualization saved as {output_path}")


def show_augmentation_examples():
    """
    Show examples of the full augmentation pipeline
    """
    data_dir = "/home/car/mv/cut_parts/splitREc/H-"  # Use one class for example
    
    # Find a sample image
    sample_img_path = None
    for img_file in os.listdir(data_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_img_path = os.path.join(data_dir, img_file)
            break
    
    if not sample_img_path:
        print("No sample image found!")
        return
        
    # Create multiple augmented versions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Show original in the first subplot
    original = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')  # Leave this empty
    
    # Show 4 augmented versions
    for i in range(4):
        processed = preprocess_image_with_bias_removal(sample_img_path, canvas_size=(300, 64))
        
        # Show in top row (positions 1-4)
        col_pos = i + 1
        axes[0, col_pos].imshow(processed, cmap='gray')
        axes[0, col_pos].set_title(f'Augmented #{i+1}', fontsize=12)
        axes[0, col_pos].axis('off')
        
        # Apply another random augmentation for comparison
        processed2 = preprocess_image_with_bias_removal(sample_img_path, canvas_size=(300, 64))
        axes[1, col_pos].imshow(processed2, cmap='gray')
        axes[1, col_pos].set_title(f'Augmented #{i+1} (diff)', fontsize=12)
        axes[1, col_pos].axis('off')
    
    plt.tight_layout()
    
    output_path = "augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Augmentation examples saved as {output_path}")


if __name__ == "__main__":
    print("Creating visualizations for positional bias removal...")
    
    visualize_bias_removal_effect()
    print()
    
    show_augmentation_examples()
    print()
    
    print("Visualizations completed!")