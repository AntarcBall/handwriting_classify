import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ContentShiftTransform:
    """
    Custom transform to randomly shift the content (handwriting) within image boundaries
    This prevents the model from learning from position tendencies
    """
    def __init__(self, max_shift_ratio=0.3):
        """
        Args:
            max_shift_ratio: Maximum shift as a ratio of image dimensions (0.0 to 1.0)
        """
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, image):
        img = image.copy()
        h, w = img.shape[:2]
        
        # Find bounding box of non-background content (assuming black on white)
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img
        
        # Invert to find text regions (assuming text is dark)
        _, thresh = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of text
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all text regions
            all_points = np.concatenate(contours, axis=0)
            x, y, w_cont, h_cont = cv2.boundingRect(all_points)
            
            # Calculate max possible shifts based on content size and image size
            max_x_shift = int(min(x, w - (x + w_cont)) * self.max_shift_ratio)
            max_y_shift = int(min(y, h - (y + h_cont)) * self.max_shift_ratio)
            
            # Random shifts
            x_shift = np.random.randint(-max_x_shift, max_x_shift + 1)
            y_shift = np.random.randint(-max_y_shift, max_y_shift + 1)
            
            # Create transformation matrix for shifting
            M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            
            # Apply shift
            shifted_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(255, 255, 255))  # White background
            
            return shifted_img
        else:
            # If no content found, return original
            return img


class WriterIdentificationDataset(Dataset):
    """
    Custom dataset for writer identification with data augmentation
    """
    def __init__(self, data_dir, transform=None, augmentation_transform=None):
        """
        Args:
            data_dir (str): Directory containing class subdirectories (H-, M-, Z-)
            transform (callable, optional): Transform to be applied on images (for normalization)
            augmentation_transform (callable, optional): Transform for data augmentation
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        
        # Get all class directories and create class mapping
        self.class_dirs = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        self.class_dirs.sort()  # Ensure consistent ordering
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_dirs)}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(data_dir, class_dir)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image from {img_path}")
        
        # Convert grayscale to 3-channel for compatibility with pretrained models
        image = np.stack([image] * 3, axis=-1)
        
        # Apply augmentation if available
        if self.augmentation_transform:
            # Check if our custom transform is included
            if isinstance(self.augmentation_transform, A.Compose):
                # Apply albumentations transforms
                augmented = self.augmentation_transform(image=image)
                image = augmented['image']
            else:
                image = self.augmentation_transform(image=image)['image']
        
        # Apply normalization transform
        if self.transform:
            # If image is already a tensor, we need to be careful
            if isinstance(image, torch.Tensor):
                # If image is already a tensor, return it as is
                pass
            else:
                # Apply normalization transforms to numpy array
                normalized = self.transform(image=image)
                image = normalized['image']

        label = self.labels[idx]
        return image, label


def get_base_transforms(input_size=(224, 224)):
    """
    Basic transforms for normalization without augmentation
    """
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2(),
    ])


def get_augmentation_transforms(input_size=(224, 224)):
    """
    Transforms with data augmentation for training (without normalization)
    """
    # Compose transforms that include shifting content within the image
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Rotate(limit=20, p=0.8),  # Rotate by up to 20 degrees
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.8),
        A.Transpose(p=0.2),  # Transpose the image sometimes
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=15, p=0.5), # Additional shifting
        A.ElasticTransform(alpha=2, sigma=80, p=0.7),  # High elastic distortion
        A.GridDistortion(num_steps=10, distort_limit=0.3, p=0.6),  # High grid distortion
        A.GaussNoise(std_range=(0.1, 0.3), p=0.5),  # Add more Gaussian noise
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Blur(blur_limit=5, p=0.3),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),  # High optical distortion
        A.MotionBlur(blur_limit=7, p=0.3),  # Motion blur
        A.MedianBlur(blur_limit=5, p=0.2),  # Median blur for extra texture
    ])


def create_datasets(data_dir, input_size=(224, 224)):
    """
    Create train, validation, and test datasets with appropriate transforms
    
    Args:
        data_dir (str): Directory containing class subdirectories
        input_size (tuple): Size to resize images to
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Get transforms
    train_transform = get_base_transforms(input_size)
    val_test_transform = get_base_transforms(input_size)
    
    # Create full dataset
    full_dataset = WriterIdentificationDataset(
        data_dir=data_dir,
        transform=train_transform,  # Will be applied after augmentation
        augmentation_transform=None  # Initially no augmentation
    )
    
    # Split dataset indices
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split1 = int(0.6 * dataset_size)  # 60% for training
    split2 = int(0.8 * dataset_size)  # 20% for validation, 20% for test
    
    import random
    random.shuffle(indices)
    
    train_indices = indices[:split1]
    val_indices = indices[split1:split2]
    test_indices = indices[split2:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(
            data_dir=data_dir,
            transform=get_base_transforms(input_size),  # Base transforms
            augmentation_transform=get_augmentation_transforms(input_size)  # Augmentation
        ),
        train_indices
    )
    
    val_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(
            data_dir=data_dir,
            transform=get_base_transforms(input_size),
            augmentation_transform=None  # No augmentation for validation
        ),
        val_indices
    )
    
    test_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(
            data_dir=data_dir,
            transform=get_base_transforms(input_size),
            augmentation_transform=None  # No augmentation for test
        ),
        test_indices
    )
    
    return train_dataset, val_dataset, test_dataset


def augment_single_image(image_path, num_augmented=5, input_size=(224, 224)):
    """
    Generate augmented versions of a single image for data augmentation
    
    Args:
        image_path (str): Path to the input image
        num_augmented (int): Number of augmented versions to create
        input_size (tuple): Size to resize images to
    
    Returns:
        list: List of augmented images
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert grayscale to 3-channel for compatibility
    image = np.stack([image] * 3, axis=-1)
    
    # Create augmentation transforms
    aug_transform = get_augmentation_transforms(input_size)
    
    augmented_images = []
    for _ in range(num_augmented):
        augmented = aug_transform(image=image)
        augmented_images.append(augmented['image'])
    
    return augmented_images


if __name__ == "__main__":
    # Example usage
    data_dir = "/home/car/mv/writer_identification/processed_data"
    
    print("Creating datasets with augmentation...")
    train_dataset, val_dataset, test_dataset = create_datasets(data_dir)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Example: Get a sample from the training set
    sample_img, sample_label = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")