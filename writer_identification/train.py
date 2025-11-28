import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json

from model import create_model, get_device
from augmentation import create_datasets


def train_model(data_dir, num_epochs=50, batch_size=16, learning_rate=1e-4, model_save_path=None):
    """
    Train the writer identification model
    
    Args:
        data_dir (str): Directory containing class subdirectories (H-, M-, Z-)
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        model_save_path (str): Path to save the trained model
    
    Returns:
        tuple: (trained_model, training_history)
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(data_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model and optimizer
    print("Creating model...")
    model, optimizer = create_model(num_classes=3, pretrained=True, learning_rate=learning_rate)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 5 == 0:  # Print progress every 5 batches
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Save best model
        if avg_val_acc > history['best_val_acc']:
            history['best_val_acc'] = avg_val_acc
            history['best_epoch'] = epoch + 1
            
            if model_save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': avg_val_acc,
                }, model_save_path)
                print(f"Saved best model at epoch {epoch+1} with validation accuracy: {avg_val_acc:.4f}")
        
        print(f'Epoch: {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, '
              f'Best Val Acc: {history["best_val_acc"]:.4f} at epoch {history["best_epoch"]}')
    
    print("Training completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
    
    return model, history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test dataset
    
    Args:
        model: Trained model
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot accuracy difference (overfitting indicator)
    plt.subplot(1, 3, 3)
    acc_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
    plt.plot(epochs, acc_diff, label='Train-Val Accuracy Gap', color='red')
    plt.title('Overfitting Indicator')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to run the complete training pipeline
    """
    data_dir = "/home/car/mv/writer_identification/processed_data"
    model_save_path = "/home/car/mv/writer_identification/models/best_writer_model.pth"
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"Starting training with data from: {data_dir}")
    
    # Train the model
    trained_model, history = train_model(
        data_dir=data_dir,
        num_epochs=30,  # Reduced for initial testing
        batch_size=8,   # Reduced for limited data
        learning_rate=1e-4,
        model_save_path=model_save_path
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print("Training history saved to training_history.json")
    
    # Load the best model for evaluation
    device = get_device()
    best_model, _ = create_model(num_classes=3)
    best_model = best_model.to(device)
    
    # Load best checkpoint
    checkpoint = torch.load(model_save_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and loader for evaluation
    _, _, test_dataset = create_datasets(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Evaluate the model
    print("Evaluating the best model...")
    metrics = evaluate_model(best_model, test_loader, device)
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    
    # Save evaluation metrics
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Evaluation metrics saved to evaluation_metrics.json")


if __name__ == "__main__":
    main()