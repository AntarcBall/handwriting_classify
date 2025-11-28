import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from gradcam import load_model_for_gradcam, GradCAMWrapper
from model import WriterIdentificationModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F


def preprocess_image_for_gradcam(image_path):
    """Preprocess image for model inference"""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert grayscale to 3-channel
    image_rgb = np.stack([image] * 3, axis=-1)
    
    # Define transforms (same as used in training)
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Apply transforms
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def create_gradcam_overlay(model, image_path, class_names=['H', 'M', 'Z']):
    """Create GradCAM overlay and get prediction info"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Preprocess image
    image_tensor = preprocess_image_for_gradcam(image_path).to(device)
    
    # Get model prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    # Load original image for overlay
    original_image_bgr = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    orig_h, orig_w = original_image_rgb.shape[:2]
    
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
    if original_image_rgb.max() <= 1.0:
        original_image_rgb = (original_image_rgb * 255).astype(np.uint8)
    
    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(original_image_rgb, 0.7, heatmap, 0.3, 0)
    
    # Get prediction info
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    return overlay, predicted_class, confidence, probabilities, original_image_rgb


def create_montage():
    """Create a vertical montage of 6 GradCAM visualizations (2 per class)"""
    model_path = "/home/car/mv/writer_identification/models/best_writer_model.pth"
    
    if not os.path.exists(model_path):
        print("Trained model not found! Please run the training script first.")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_model_for_gradcam(model_path)
    class_names = ['H', 'M', 'Z']
    
    # Find 2 images per class
    data_dir = "/home/car/mv/writer_identification/processed_data"
    images_to_process = []
    
    for class_name in class_names:
        class_dir = f"{class_name}-"
        class_path = os.path.join(data_dir, class_dir)
        
        if os.path.exists(class_path):
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Take up to 2 images per class
            for img_file in img_files[:2]:
                img_path = os.path.join(class_path, img_file)
                images_to_process.append((img_path, class_dir))
                
                if len(images_to_process) % 2 == 0:  # 2 images per class
                    break
    
    print(f"Processing {len(images_to_process)} images...")
    
    # Create subplots for the montage
    fig, axes = plt.subplots(len(images_to_process), 1, figsize=(12, 4*len(images_to_process)))
    
    if len(images_to_process) == 1:
        axes = [axes]
    
    for idx, (img_path, class_dir) in enumerate(images_to_process):
        try:
            # Create GradCAM overlay
            overlay, pred_class, confidence, probabilities, original = create_gradcam_overlay(
                model, img_path, class_names
            )
            
            # Display the overlay on the subplot
            axes[idx].imshow(overlay)
            axes[idx].axis('off')
            
            # Create title with prediction info
            title = (f"File: {os.path.basename(img_path)} | "
                    f"Predicted: {pred_class} (confidence: {confidence:.3f})\n"
                    f"Probabilities - H: {probabilities[0]:.3f}, "
                    f"M: {probabilities[1]:.3f}, "
                    f"Z: {probabilities[2]:.3f}")
            
            axes[idx].set_title(title, fontsize=12)
            
            print(f"Processed {os.path.basename(img_path)} -> {pred_class} (conf: {confidence:.3f})")
            print(f"  Probabilities: H={probabilities[0]:.3f}, M={probabilities[1]:.3f}, Z={probabilities[2]:.3f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            axes[idx].text(0.5, 0.5, f"Error: {str(e)}", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[idx].transAxes, fontsize=12, color='red')
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the montage
    output_path = "gradcam_montage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nMontage saved as {output_path}")
    print(f"Total images in montage: {len(images_to_process)}")


if __name__ == "__main__":
    create_montage()