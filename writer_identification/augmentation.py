import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


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
            augmented = self.augmentation_transform(image=image)
            image = augmented['image']

        # Apply normalization transform
        if self.transform:
            # Convert tensor back to numpy array if it's already a tensor due to augmentation
            if torch.is_tensor(image):
                image = image.permute(1, 2, 0).numpy()  # Convert from C, H, W to H, W, C
            transformed = self.transform(image=image)
            image = transformed['image']

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
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Rotate(limit=20, p=0.8),  # Rotate by up to 20 degrees
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.8),
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
    train_transform = get_augmentation_transforms(input_size)
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