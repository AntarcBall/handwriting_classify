import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import WriterIdentificationModel


class UnknownWriterDetector:
    """
    Class to implement thresholding for unknown writer detection.
    This addresses the critical flaw of softmax always choosing one of the 3 classes.
    """
    def __init__(self, model, threshold=0.9):
        """
        Args:
            model: Trained writer identification model
            threshold: Confidence threshold below which a sample is considered 'unknown'
        """
        self.model = model
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference using the same transforms as during training
        
        Args:
            image_path: Path to the input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
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
        
        return image_tensor.to(self.device)
    
    def predict_with_threshold(self, image_path, class_names=['H', 'M', 'Z']):
        """
        Predict the writer with thresholding to detect unknown writers
        
        Args:
            image_path: Path to the input image
            class_names: Names of the known classes
        
        Returns:
            tuple: (prediction_result, confidence_score, class_probabilities)
                   prediction_result can be a class name or 'Unknown'
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image_path)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Find the maximum probability and corresponding class
        max_prob = np.max(probabilities)
        predicted_class_idx = np.argmax(probabilities)
        
        # Apply thresholding
        if max_prob >= self.threshold:
            result = class_names[predicted_class_idx]
        else:
            result = "Unknown"
        
        return result, max_prob, {class_names[i]: prob for i, prob in enumerate(probabilities)}
    
    def predict_batch(self, image_paths, class_names=['H', 'M', 'Z']):
        """
        Predict for a batch of images with thresholding
        
        Args:
            image_paths: List of image paths
            class_names: Names of the known classes
        
        Returns:
            list: List of prediction results for each image
        """
        results = []
        for img_path in image_paths:
            try:
                result, confidence, all_probs = self.predict_with_threshold(img_path, class_names)
                results.append({
                    'image_path': img_path,
                    'prediction': result,
                    'confidence': confidence,
                    'probabilities': all_probs
                })
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': str(e)
                })
        return results
    
    def find_optimal_threshold(self, validation_loader, class_names=['H', 'M', 'Z']):
        """
        Find the optimal threshold for unknown writer detection based on validation data
        
        Args:
            validation_loader: DataLoader containing validation images
            class_names: Names of the known classes
        
        Returns:
            float: Optimal threshold value
        """
        all_confidences = []
        all_labels = []
        
        # Collect confidence scores for known writers
        with torch.no_grad():
            for data, target in validation_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0].cpu().numpy()
                
                all_confidences.extend(max_probs)
                all_labels.extend(target.cpu().numpy())
        
        # Find the minimum confidence among known writers (this could be used as threshold)
        if len(all_confidences) > 0:
            # Calculate threshold as a percentile to balance between false positives and negatives
            # Using 5th percentile as a starting point for threshold
            optimal_threshold = np.percentile(all_confidences, 5)  # 5th percentile
            print(f"Calculated optimal threshold: {optimal_threshold:.3f}")
            return optimal_threshold
        else:
            # Default threshold if no validation data
            return 0.9
    
    def set_threshold(self, threshold):
        """
        Set a new confidence threshold
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        print(f"Threshold updated to: {self.threshold}")


def create_unknown_detector(model_path, threshold=0.9, num_classes=3):
    """
    Create an unknown writer detector from a saved model
    
    Args:
        model_path: Path to the saved model checkpoint
        threshold: Confidence threshold for unknown detection
        num_classes: Number of classes in the model
    
    Returns:
        UnknownWriterDetector: The configured detector
    """
    # Load the trained model
    model = WriterIdentificationModel(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create detector
    detector = UnknownWriterDetector(model, threshold=threshold)
    
    return detector


def evaluate_threshold_performance(detector, test_image_paths, true_labels, class_names=['H', 'M', 'Z']):
    """
    Evaluate the performance of the threshold-based detection on known vs unknown writers
    
    Args:
        detector: UnknownWriterDetector instance
        test_image_paths: Paths to test images
        true_labels: True labels for known writers, 'Unknown' for unknown writers
        class_names: Names of known classes
    
    Returns:
        dict: Performance metrics
    """
    correct_known = 0  # Correctly identified known writers
    total_known = 0    # Total number of known writers in test set
    correct_unknown = 0  # Correctly identified unknown writers
    total_unknown = 0    # Total number of unknown writers in test set
    
    all_predictions = []
    
    for img_path, true_label in zip(test_image_paths, true_labels):
        pred, conf, probs = detector.predict_with_threshold(img_path, class_names)
        all_predictions.append((img_path, true_label, pred, conf))
        
        if true_label in class_names:  # Known writer
            total_known += 1
            if pred == true_label:
                correct_known += 1
        else:  # Unknown writer
            total_unknown += 1
            if pred == "Unknown":
                correct_unknown += 1
    
    # Calculate metrics
    metrics = {}
    if total_known > 0:
        metrics['known_accuracy'] = correct_known / total_known
    else:
        metrics['known_accuracy'] = 0.0
    
    if total_unknown > 0:
        metrics['unknown_accuracy'] = correct_unknown / total_unknown
    else:
        metrics['unknown_accuracy'] = 0.0
    
    if total_known + total_unknown > 0:
        metrics['overall_accuracy'] = (correct_known + correct_unknown) / (total_known + total_unknown)
    else:
        metrics['overall_accuracy'] = 0.0
    
    metrics['total_known'] = total_known
    metrics['total_unknown'] = total_unknown
    metrics['correct_known'] = correct_known
    metrics['correct_unknown'] = correct_unknown
    metrics['predictions'] = all_predictions
    
    return metrics


if __name__ == "__main__":
    # Example usage
    model_path = "/home/car/mv/writer_identification/models/best_writer_model.pth"
    
    # Create unknown writer detector with default threshold
    detector = create_unknown_detector(model_path, threshold=0.9)
    
    # Test on a sample image
    sample_image_path = "/home/car/mv/writer_identification/processed_data/H-/rectangle_10.png"
    
    print(f"Testing unknown writer detection on: {sample_image_path}")
    result, confidence, probabilities = detector.predict_with_threshold(
        sample_image_path, 
        class_names=['H', 'M', 'Z']
    )
    
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.4f}")
    print("Class probabilities:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Try with different thresholds
    for threshold in [0.5, 0.7, 0.9]:
        detector.set_threshold(threshold)
        result, confidence, _ = detector.predict_with_threshold(
            sample_image_path, 
            class_names=['H', 'M', 'Z']
        )
        print(f"With threshold {threshold}, prediction: {result} (confidence: {confidence:.4f})")