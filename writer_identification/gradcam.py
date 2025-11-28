import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import WriterIdentificationModel
import os
from PIL import Image


class GradCAMWrapper:
    """
    Wrapper class for GradCAM visualization on the Writer Identification model
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained PyTorch model
            target_layer: The layer to compute gradients with respect to
        """
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAM(model=model, target_layers=[target_layer])
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map for the input
        
        Args:
            input_tensor: Input tensor to the model
            target_class: Target class for which to generate the CAM. 
                         If None, uses the predicted class.
        
        Returns:
            numpy array: The CAM heatmap
        """
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=target_class)
        
        # Take the first image in the batch
        return grayscale_cam[0, :]
    
    def overlay_cam_on_image(self, input_image, cam, alpha=0.7):
        """
        Overlay the CAM heatmap on the original image
        
        Args:
            input_image: Original input image (numpy array)
            cam: CAM heatmap (numpy array)
            alpha: Transparency factor for overlay
        
        Returns:
            numpy array: Image with heatmap overlay
        """
        # Normalize CAM to [0, 1]
        cam = np.float32(cam) - np.min(cam)
        cam = cam / np.max(cam)
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Convert input image to RGB if it's grayscale
        if len(input_image.shape) == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif input_image.shape[0] == 3:  # Channel-first format, convert to HWC
            input_image = np.transpose(input_image, (1, 2, 0))
        
        # Normalize input image to [0, 1] if it's in [-1, 1] or [0, 1] range
        if input_image.max() <= 1.0:
            input_image = (input_image * 255).astype(np.uint8)
        elif input_image.min() < 0:
            input_image = ((input_image + 1) * 127.5).astype(np.uint8)
        
        # Overlay the heatmap on the input image
        overlay = cv2.addWeighted(input_image, alpha, heatmap, 1 - alpha, 0)
        
        return overlay


def visualize_gradcam(model, test_image_path, class_names=['H', 'M', 'Z'], save_path=None):
    """
    Visualize GradCAM for a specific image
    
    Args:
        model: Trained model
        test_image_path: Path to test image
        class_names: Names of classes for display
        save_path: Path to save the visualization image
    """
    # Load and preprocess the test image
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {test_image_path}")
    
    # Use the same preprocessing as in training/augmentation
    # Convert grayscale to 3-channel
    image_rgb = np.stack([image] * 3, axis=-1)

    # Define transforms (same as used in training)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Apply transforms
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    # Move to device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Get the target layer for GradCAM (usually a layer in the features)
    # For AlexNet, the last conv layer is classifier[0] -> features[12]
    target_layer = model.alexnet.features[-1]  # Last convolutional layer

    # Create GradCAM wrapper
    gradcam_wrapper = GradCAMWrapper(model, target_layer)
    
    # Get model prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()
        predicted_prob = torch.softmax(output, dim=1).max().item()
    
    print(f"Predicted class: {class_names[predicted_class]} (index: {predicted_class})")
    print(f"Prediction probability: {predicted_prob:.4f}")
    
    # Generate GradCAM for the predicted class
    cam = gradcam_wrapper.generate_cam(image_tensor, target_class=None)
    
    # Load original image again for visualization (before preprocessing)
    original_image_bgr = cv2.imread(test_image_path)
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    orig_h, orig_w = original_image_rgb.shape[:2]

    # Resize the CAM to match original image dimensions
    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    # Overlay CAM on original image using original dimensions
    overlay_image = gradcam_wrapper.overlay_cam_on_image(original_image_rgb, cam_resized)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'GradCAM Overlay\nPredicted: {class_names[predicted_class]} ({predicted_prob:.2f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return cam, overlay_image


def visualize_gradcam_for_dataset(model, data_dir, class_names=['H', 'M', 'Z'], num_samples_per_class=2):
    """
    Generate GradCAM visualizations for multiple samples from the dataset
    
    Args:
        model: Trained model
        data_dir: Directory containing class subdirectories
        class_names: Names of classes for display
        num_samples_per_class: Number of samples to visualize per class
    """
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Take first few samples for visualization
        for i, img_file in enumerate(image_files[:num_samples_per_class]):
            if i >= num_samples_per_class:
                break
                
            img_path = os.path.join(class_path, img_file)
            save_path = f"gradcam_{class_dir}_{img_file.replace('.', '_')}.png"
            
            print(f"\nGenerating GradCAM for: {img_path}")
            try:
                visualize_gradcam(model, img_path, class_names, save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")


def load_model_for_gradcam(model_path, num_classes=3):
    """
    Load a trained model for GradCAM analysis
    
    Args:
        model_path: Path to the saved model checkpoint
        num_classes: Number of classes in the model
    
    Returns:
        Loaded model
    """
    model = WriterIdentificationModel(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model


if __name__ == "__main__":
    # Example usage
    model_path = "/home/car/mv/writer_identification/models/best_writer_model.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        model = load_model_for_gradcam(model_path)
        
        # Visualize GradCAM for a sample from the processed dataset
        test_data_dir = "/home/car/mv/writer_identification/processed_data"
        
        # Find a sample image
        for class_dir in sorted(os.listdir(test_data_dir)):
            class_path = os.path.join(test_data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_img_path = os.path.join(class_path, img_file)
                        print(f"Generating GradCAM for sample: {sample_img_path}")
                        
                        # Generate visualization for the first sample
                        visualize_gradcam(model, sample_img_path, 
                                        class_names=['H', 'M', 'Z'],
                                        save_path=f"gradcam_sample_{class_dir}_{img_file.replace('.', '_')}.png")
                        break
                break
    else:
        print("No trained model found. Please run the training script first.")