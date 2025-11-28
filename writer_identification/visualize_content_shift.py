import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from gradcam import load_model_for_gradcam, GradCAMWrapper
from model import WriterIdentificationModel


def apply_content_shifting_augmentation(image_path, num_examples=4):
    """
    Apply content-shifting augmentation to an image and return original + shifted versions
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert grayscale to 3-channel
    image_rgb = np.stack([image] * 3, axis=-1)
    
    # Find bounding box of non-background content (assuming black on white)
    if len(image_rgb.shape) == 3:
        gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image_rgb
    
    # Invert to find text regions (assuming text is dark)
    _, thresh = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of text
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    original = image_rgb.copy()
    shifted_images = [original]  # Include original as first example
    
    if contours:
        # Get bounding box of all text regions
        all_points = np.concatenate(contours, axis=0)
        x, y, w_cont, h_cont = cv2.boundingRect(all_points)
        
        h, w = image_rgb.shape[:2]
        
        for i in range(num_examples - 1):  # Generate shifted versions
            # Calculate max possible shifts based on content size and image size
            max_x_shift = int(min(x, w - (x + w_cont)) * 0.3)  # 30% of possible shift
            max_y_shift = int(min(y, h - (y + h_cont)) * 0.3)
            
            # Random shifts
            x_shift = np.random.randint(-max_x_shift, max_x_shift + 1)
            y_shift = np.random.randint(-max_y_shift, max_y_shift + 1)
            
            # Create transformation matrix for shifting
            M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            
            # Apply shift
            shifted_img = cv2.warpAffine(image_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(255, 255, 255))  # White background
            
            shifted_images.append(shifted_img)
    else:
        # If no content found, duplicate the original
        for i in range(num_examples - 1):
            shifted_images.append(original)
    
    return shifted_images


def preprocess_image_for_gradcam(image_np):
    """Preprocess numpy image for model inference"""
    # Define transforms (same as used in training)
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Apply transforms
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def create_gradcam_for_shifted_images(model, image_path, class_names=['H', 'M', 'Z']):
    """Create GradCAM for original and shifted versions of an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get shifted versions of the image
    shifted_versions = apply_content_shifting_augmentation(image_path, num_examples=4)
    
    results = []
    
    for i, img in enumerate(shifted_versions):
        # Preprocess for model
        image_tensor = preprocess_image_for_gradcam(img).to(device)
        
        # Get model prediction
        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Get original dimensions for proper overlay
        orig_h, orig_w = img.shape[:2]
        
        # Get target layer and create GradCAM
        target_layer = model.alexnet.features[-1]
        cam = GradCAMWrapper(model, target_layer)
        
        # Generate CAM
        grayscale_cam = cam.generate_cam(image_tensor, target_class=None)
        
        # Resize the CAM to match original image dimensions
        cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))
        
        # Normalize CAM to [0, 1]
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
        
        # Normalize original image to [0, 255] if needed
        if img.max() <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img
        
        # If image is grayscale, convert to RGB for overlay
        if len(img_display.shape) == 2:
            img_display = np.stack([img_display] * 3, axis=-1)
        
        # Overlay the heatmap on the image
        overlay = cv2.addWeighted(img_display, 0.7, heatmap, 0.3, 0)
        
        # Get prediction info
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        results.append({
            'image_type': 'Original' if i == 0 else f'Shifted #{i}',
            'image': img_display,
            'overlay': overlay,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    return results


def create_content_shift_visualization():
    """Create a visualization showing content shifting with GradCAM"""
    model_path = "/home/car/mv/writer_identification/models/best_writer_model.pth"
    
    if not os.path.exists(model_path):
        print("Trained model not found! Please run the training script first.")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_model_for_gradcam(model_path)
    class_names = ['H', 'M', 'Z']
    
    # Find an image to visualize
    data_dir = "/home/car/mv/writer_identification/processed_data"
    sample_img_path = None
    
    for class_dir in ['H-', 'M-', 'Z-']:
        class_path = os.path.join(data_dir, class_dir)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_img_path = os.path.join(class_path, img_file)
                    break
        if sample_img_path:
            break
    
    if not sample_img_path:
        print("No images found to visualize!")
        return
    
    print(f"Creating visualization for: {sample_img_path}")
    
    # Create GradCAM results for original and shifted images
    results = create_gradcam_for_shifted_images(model, sample_img_path, class_names)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Show original and shifted images
    for i, result in enumerate(results):
        axes[0, i].imshow(result['image'])
        axes[0, i].set_title(f"{result['image_type']}\nClass: {result['predicted_class']}, "
                            f"Conf: {result['confidence']:.3f}", fontsize=10)
        axes[0, i].axis('off')
        
        # Show GradCAM overlay
        axes[1, i].imshow(result['overlay'])
        axes[1, i].set_title(f"GradCAM - {result['predicted_class']}\n"
                            f"H:{result['probabilities'][0]:.2f}, "
                            f"M:{result['probabilities'][1]:.2f}, "
                            f"Z:{result['probabilities'][2]:.2f}", fontsize=9)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    output_path = "content_shift_gradcam_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Content shift visualization saved as {output_path}")
    
    # Print details for each version
    print("\nDetailed results:")
    for i, result in enumerate(results):
        print(f"{result['image_type']}:")
        print(f"  Predicted: {result['predicted_class']} (conf: {result['confidence']:.3f})")
        print(f"  Probabilities - H: {result['probabilities'][0]:.3f}, "
              f"M: {result['probabilities'][1]:.3f}, "
              f"Z: {result['probabilities'][2]:.3f}")
        print()


if __name__ == "__main__":
    create_content_shift_visualization()