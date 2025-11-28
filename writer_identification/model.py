import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class WriterIdentificationModel(nn.Module):
    """
    AlexNet-based model for writer identification with transfer learning.
    Modified to have 3 output classes (H, M, Z).
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(WriterIdentificationModel, self).__init__()
        
        # Load pretrained AlexNet model
        self.alexnet = models.alexnet(pretrained=pretrained)
        
        # Get the number of features from the last classifier layer
        num_features = self.alexnet.classifier[6].in_features
        
        # Replace the final classifier layer with one that matches our number of classes
        # AlexNet classifier is a sequential of layers [0-5] + [6] (final layer)
        # We replace only the last layer (index 6) which is the classifier
        self.alexnet.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)
    
    def get_features(self, x):
        """
        Extract features before the final classification layer
        """
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier[:6](x)  # All layers except the final classifier
        return x


def create_model(num_classes=3, pretrained=True, learning_rate=1e-4):
    """
    Create the model and optimizer with specified parameters
    
    Args:
        num_classes (int): Number of output classes (3 for H, M, Z)
        pretrained (bool): Whether to use pretrained weights
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        tuple: (model, optimizer)
    """
    model = WriterIdentificationModel(num_classes=num_classes, pretrained=pretrained)
    
    # Use a lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    return model, optimizer


def get_device():
    """
    Get the appropriate device (GPU if available, otherwise CPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Test the model
    model, optimizer = create_model(num_classes=3)
    device = get_device()
    model = model.to(device)
    
    print("Model created successfully!")
    print(f"Model is running on: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")