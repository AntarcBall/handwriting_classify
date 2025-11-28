"""
Generate architecture and pipeline diagrams for the Writer Identification System
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def create_overall_pipeline_diagram():
    """
    Create a comprehensive pipeline diagram showing the complete workflow
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Writer Identification System - Complete Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Color scheme
    input_color = '#E3F2FD'
    preprocess_color = '#FFF9C4'
    augment_color = '#F0F4C3'
    model_color = '#FFCCBC'
    output_color = '#C8E6C9'
    special_color = '#E1BEE7'
    
    # 1. Data Input Stage
    box1 = FancyBboxPatch((0.5, 7), 2.5, 1.2, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 7.8, 'Raw Image\nData', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(1.75, 7.3, 'H-, M-, Z-\nclasses', ha='center', va='center', fontsize=8)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((3, 7.6), (4.2, 7.6), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow1)
    
    # 2. Preprocessing Stage
    box2 = FancyBboxPatch((4.2, 6.5), 3, 2.2, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=preprocess_color, linewidth=2)
    ax.add_patch(box2)
    ax.text(5.7, 8.3, 'Preprocessing', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Preprocessing steps
    steps = [
        '1. Background Noise Removal',
        '2. Binarization (B&W)',
        '3. Normalize & Pad (64px height)',
        '4. Maintain Aspect Ratio'
    ]
    for i, step in enumerate(steps):
        ax.text(5.7, 7.9 - i*0.35, step, ha='center', va='center', fontsize=7)
    
    # Arrow 2
    arrow2 = FancyArrowPatch((7.2, 7.6), (8.4, 7.6), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow2)
    
    # 3. Data Augmentation (Training Only)
    box3 = FancyBboxPatch((8.4, 5.8), 3.2, 3.6, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=augment_color, linewidth=2)
    ax.add_patch(box3)
    ax.text(10, 9.1, 'Data Augmentation', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(10, 8.7, '(Training Only)', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Augmentation techniques
    aug_steps = [
        '• Rotation (±20°)',
        '• Shift/Scale/Rotate',
        '• Elastic Transform',
        '• Grid Distortion',
        '• Gaussian Noise',
        '• Brightness/Contrast',
        '• Motion Blur',
        '• Optical Distortion'
    ]
    for i, step in enumerate(aug_steps):
        ax.text(10, 8.2 - i*0.3, step, ha='center', va='center', fontsize=7)
    
    # Arrow 3
    arrow3 = FancyArrowPatch((11.6, 7.6), (12.8, 7.6), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow3)
    
    # 4. Model Training/Inference
    box4 = FancyBboxPatch((12.8, 6.2), 2.7, 2.8, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=model_color, linewidth=2)
    ax.add_patch(box4)
    ax.text(14.15, 8.7, 'AlexNet Model', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    model_info = [
        'Transfer Learning',
        'Pretrained on ImageNet',
        'Modified Final Layer',
        '3 Output Classes (H, M, Z)',
        'Adam Optimizer',
        'Cross-Entropy Loss'
    ]
    for i, info in enumerate(model_info):
        ax.text(14.15, 8.3 - i*0.35, info, ha='center', va='center', fontsize=7)
    
    # Arrow 4 (down)
    arrow4 = FancyArrowPatch((14.15, 6.2), (14.15, 5.2), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow4)
    
    # 5. Output & Analysis
    box5 = FancyBboxPatch((12.8, 3.5), 2.7, 1.5, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(box5)
    ax.text(14.15, 4.6, 'Prediction Output', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(14.15, 4.15, 'Class probabilities\n(H, M, Z)', ha='center', va='center', fontsize=8)
    
    # Arrow 5 (down)
    arrow5 = FancyArrowPatch((14.15, 3.5), (14.15, 2.5), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow5)
    
    # 6. Unknown Writer Detection
    box6 = FancyBboxPatch((12.8, 1), 2.7, 1.3, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=special_color, linewidth=2)
    ax.add_patch(box6)
    ax.text(14.15, 2.0, 'Threshold Check', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(14.15, 1.55, 'Confidence ≥ threshold?\nYes: Classify | No: Unknown', 
            ha='center', va='center', fontsize=7)
    
    # Side: Data Split Info
    split_box = FancyBboxPatch((0.3, 3.5), 2.8, 2, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='#E8EAF6', linewidth=2)
    ax.add_patch(split_box)
    ax.text(1.7, 5.2, 'Data Split', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(1.7, 4.75, '60% Training', ha='center', va='center', fontsize=9)
    ax.text(1.7, 4.4, '20% Validation', ha='center', va='center', fontsize=9)
    ax.text(1.7, 4.05, '20% Testing', ha='center', va='center', fontsize=9)
    
    # Side: Visualization Tools
    viz_box = FancyBboxPatch((0.3, 1), 2.8, 2, boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor='#FCE4EC', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(1.7, 2.7, 'Analysis Tools', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(1.7, 2.3, '• GradCAM Visualization', ha='left', va='center', fontsize=8)
    ax.text(1.7, 2.0, '• Confusion Matrix', ha='left', va='center', fontsize=8)
    ax.text(1.7, 1.7, '• Metrics (Acc, P, R, F1)', ha='left', va='center', fontsize=8)
    ax.text(1.7, 1.4, '• Training History', ha='left', va='center', fontsize=8)
    
    # Side: Application
    app_box = FancyBboxPatch((4, 0.5), 3.5, 1.5, boxstyle="round,pad=0.1", 
                             edgecolor='red', facecolor='#FFEBEE', linewidth=3)
    ax.add_patch(app_box)
    ax.text(5.75, 1.7, 'Application:', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='red')
    ax.text(5.75, 1.3, 'Proxy Attendance Detection', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(5.75, 0.9, 'Verify if signature matches\nregistered writers', 
            ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Pipeline diagram saved as 'pipeline_diagram.png'")
    plt.show()


def create_model_architecture_diagram():
    """
    Create a detailed model architecture diagram
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'AlexNet Model Architecture for Writer Identification', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1.2, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 8.5, 'Input', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.25, 8.1, '224×224×3', ha='center', va='center', fontsize=8)
    ax.text(1.25, 7.8, 'RGB Image', ha='center', va='center', fontsize=8)
    
    # Arrow
    arrow1 = FancyArrowPatch((2, 8.1), (2.8, 8.1), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow1)
    
    # Feature Extractor (AlexNet Features)
    feature_box = FancyBboxPatch((2.8, 5.5), 8, 4.2, boxstyle="round,pad=0.15", 
                                 edgecolor='blue', facecolor='#E8EAF6', linewidth=3)
    ax.add_patch(feature_box)
    ax.text(6.8, 9.4, 'Feature Extractor (AlexNet.features)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='blue')
    
    # Convolutional layers
    conv_layers = [
        ('Conv1', '11×11, 64', '55×55×64'),
        ('MaxPool', '3×3', '27×27×64'),
        ('Conv2', '5×5, 192', '27×27×192'),
        ('MaxPool', '3×3', '13×13×192'),
        ('Conv3', '3×3, 384', '13×13×384'),
        ('Conv4', '3×3, 256', '13×13×256'),
        ('Conv5', '3×3, 256', '13×13×256'),
        ('MaxPool', '3×3', '6×6×256'),
    ]
    
    y_start = 8.9
    for i, (name, kernel, output) in enumerate(conv_layers):
        y_pos = y_start - i * 0.4
        
        # Layer box
        layer_box = FancyBboxPatch((3, y_pos - 0.15), 7.4, 0.28, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='black', facecolor='#BBDEFB', linewidth=1)
        ax.add_patch(layer_box)
        
        ax.text(3.5, y_pos, name, ha='left', va='center', fontsize=8, fontweight='bold')
        ax.text(6, y_pos, f'Kernel: {kernel}', ha='center', va='center', fontsize=7)
        ax.text(8.8, y_pos, f'Output: {output}', ha='center', va='center', fontsize=7)
    
    # Arrow to avgpool
    arrow2 = FancyArrowPatch((6.8, 5.5), (6.8, 4.8), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow2)
    
    # AvgPool
    avgpool_box = FancyBboxPatch((5.5, 4.2), 2.6, 0.5, boxstyle="round,pad=0.05", 
                                 edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(avgpool_box)
    ax.text(6.8, 4.45, 'AdaptiveAvgPool2d', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    ax.text(6.8, 4.25, 'Output: 6×6×256', ha='center', va='center', fontsize=7)
    
    # Arrow to flatten
    arrow3 = FancyArrowPatch((6.8, 4.2), (6.8, 3.6), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow3)
    
    # Flatten
    flatten_box = FancyBboxPatch((5.8, 3.2), 2, 0.35, boxstyle="round,pad=0.05", 
                                 edgecolor='black', facecolor='#F0F4C3', linewidth=2)
    ax.add_patch(flatten_box)
    ax.text(6.8, 3.37, 'Flatten → 9216', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Arrow to classifier
    arrow4 = FancyArrowPatch((6.8, 3.2), (6.8, 2.5), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow4)
    
    # Classifier
    classifier_box = FancyBboxPatch((2.8, 0.5), 8, 1.9, boxstyle="round,pad=0.15", 
                                    edgecolor='red', facecolor='#FFEBEE', linewidth=3)
    ax.add_patch(classifier_box)
    ax.text(6.8, 2.25, 'Classifier (AlexNet.classifier)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='red')
    
    # Classifier layers
    classifier_layers = [
        ('Dropout(0.5)', 'FC: 4096', 'ReLU'),
        ('Dropout(0.5)', 'FC: 4096', 'ReLU'),
        ('Modified Layer', 'FC: 3 (H, M, Z)', 'Output')
    ]
    
    y_class = 1.9
    for i, (drop, fc, act) in enumerate(classifier_layers):
        y_pos = y_class - i * 0.42
        
        # Layer components
        ax.text(3.5, y_pos, drop, ha='left', va='center', fontsize=8)
        ax.text(6, y_pos, fc, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(9, y_pos, act, ha='center', va='center', fontsize=8)
        
        if i < 2:
            ax.plot([3, 10.3], [y_pos - 0.18, y_pos - 0.18], 'k-', lw=0.5)
    
    # Arrow to final output
    arrow5 = FancyArrowPatch((6.8, 0.5), (6.8, -0.2), 
                            arrowstyle='->', lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow5)
    
    # Output
    output_box = FancyBboxPatch((5.5, -1), 2.6, 0.7, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6.8, -0.5, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6.8, -0.75, '3 class probabilities', ha='center', va='center', fontsize=8)
    
    # Side: Transfer Learning Info
    transfer_box = FancyBboxPatch((11.2, 6.5), 2.5, 2.5, boxstyle="round,pad=0.1", 
                                  edgecolor='green', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(transfer_box)
    ax.text(12.45, 8.7, 'Transfer Learning', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='green')
    ax.text(12.45, 8.3, '✓ Pre-trained on', ha='center', va='center', fontsize=7)
    ax.text(12.45, 8.05, '  ImageNet', ha='center', va='center', fontsize=7)
    ax.text(12.45, 7.7, '✓ Feature layers', ha='center', va='center', fontsize=7)
    ax.text(12.45, 7.45, '  frozen initially', ha='center', va='center', fontsize=7)
    ax.text(12.45, 7.1, '✓ Final layer', ha='center', va='center', fontsize=7)
    ax.text(12.45, 6.85, '  modified: 1000→3', ha='center', va='center', fontsize=7)
    
    # Side: Training Info
    train_box = FancyBboxPatch((11.2, 3.5), 2.5, 2.5, boxstyle="round,pad=0.1", 
                               edgecolor='orange', facecolor='#FFF3E0', linewidth=2)
    ax.add_patch(train_box)
    ax.text(12.45, 5.7, 'Training', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='orange')
    ax.text(12.45, 5.3, 'Optimizer:', ha='center', va='center', fontsize=7)
    ax.text(12.45, 5.05, 'Adam (lr=1e-4)', ha='center', va='center', fontsize=7)
    ax.text(12.45, 4.7, 'Loss Function:', ha='center', va='center', fontsize=7)
    ax.text(12.45, 4.45, 'Cross-Entropy', ha='center', va='center', fontsize=7)
    ax.text(12.45, 4.1, 'Weight Decay:', ha='center', va='center', fontsize=7)
    ax.text(12.45, 3.85, '1e-4', ha='center', va='center', fontsize=7)
    
    # Side: Model Stats
    stats_box = FancyBboxPatch((11.2, 0.5), 2.5, 2.5, boxstyle="round,pad=0.1", 
                               edgecolor='purple', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(stats_box)
    ax.text(12.45, 2.7, 'Model Info', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='purple')
    ax.text(12.45, 2.3, '• ~57M parameters', ha='left', va='center', fontsize=7)
    ax.text(12.45, 2.0, '• Input: 224×224×3', ha='left', va='center', fontsize=7)
    ax.text(12.45, 1.7, '• Output: 3 classes', ha='left', va='center', fontsize=7)
    ax.text(12.45, 1.4, '• Batch size: 16', ha='left', va='center', fontsize=7)
    ax.text(12.45, 1.1, '• Epochs: 50', ha='left', va='center', fontsize=7)
    ax.text(12.45, 0.8, '• Device: GPU/CPU', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Model architecture diagram saved as 'model_architecture.png'")
    plt.show()


def create_data_flow_diagram():
    """
    Create a data flow diagram showing how data moves through the system
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Data Flow Through the System', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Stage 1: Raw Data
    stage1_box = FancyBboxPatch((0.5, 5.5), 2, 1.2, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(1.5, 6.5, 'Stage 1', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.5, 6.15, 'Raw Scanned', ha='center', va='center', fontsize=8)
    ax.text(1.5, 5.9, 'Handwriting', ha='center', va='center', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(3, 6.1), xytext=(2.5, 6.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(2.75, 6.4, 'Load', ha='center', fontsize=7)
    
    # Stage 2: Preprocessing
    stage2_box = FancyBboxPatch((3, 5.3), 2.2, 1.6, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(4.1, 6.7, 'Stage 2', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.1, 6.4, 'Preprocess', ha='center', va='center', fontsize=8)
    ax.text(4.1, 6.05, '• Remove noise', ha='center', va='center', fontsize=7)
    ax.text(4.1, 5.8, '• Binarize', ha='center', va='center', fontsize=7)
    ax.text(4.1, 5.55, '• Normalize', ha='center', va='center', fontsize=7)
    
    # Arrow
    ax.annotate('', xy=(5.7, 6.1), xytext=(5.2, 6.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.45, 6.4, 'Clean', ha='center', fontsize=7)
    
    # Stage 3: Data Split
    stage3_box = FancyBboxPatch((5.7, 5.5), 2, 1.2, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(6.7, 6.5, 'Stage 3', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.7, 6.15, 'Split Data', ha='center', va='center', fontsize=8)
    ax.text(6.7, 5.85, '60/20/20', ha='center', va='center', fontsize=8)
    
    # Three arrows down for train/val/test
    ax.annotate('', xy=(5.9, 5), xytext=(5.9, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(5.5, 5.25, 'Train', ha='center', fontsize=7, color='green')
    
    ax.annotate('', xy=(6.7, 5), xytext=(6.7, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.text(6.3, 5.25, 'Val', ha='center', fontsize=7, color='orange')
    
    ax.annotate('', xy=(7.5, 5), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(7.9, 5.25, 'Test', ha='center', fontsize=7, color='blue')
    
    # Stage 4a: Augmentation (Training path)
    stage4a_box = FancyBboxPatch((5.2, 3.5), 1.5, 1.3, boxstyle="round,pad=0.1", 
                                 edgecolor='green', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(stage4a_box)
    ax.text(5.95, 4.5, 'Stage 4a', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(5.95, 4.2, 'Augment', ha='center', va='center', fontsize=7)
    ax.text(5.95, 3.95, '(Train only)', ha='center', va='center', fontsize=6)
    ax.text(5.95, 3.7, 'Transforms', ha='center', va='center', fontsize=7)
    
    # Stage 4b: No Augmentation (Val/Test path)
    stage4b_box = FancyBboxPatch((7, 3.5), 1.5, 1.3, boxstyle="round,pad=0.1", 
                                 edgecolor='blue', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(stage4b_box)
    ax.text(7.75, 4.5, 'Stage 4b', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(7.75, 4.2, 'Normalize', ha='center', va='center', fontsize=7)
    ax.text(7.75, 3.95, '(Val/Test)', ha='center', va='center', fontsize=6)
    ax.text(7.75, 3.7, 'Resize only', ha='center', va='center', fontsize=7)
    
    # Arrows converge to model
    ax.annotate('', xy=(8.7, 3.5), xytext=(6.7, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Stage 5: Model
    stage5_box = FancyBboxPatch((8.7, 3), 2.3, 2, boxstyle="round,pad=0.1", 
                                edgecolor='red', facecolor='#FFCCBC', linewidth=2)
    ax.add_patch(stage5_box)
    ax.text(9.85, 4.7, 'Stage 5', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(9.85, 4.4, 'AlexNet Model', ha='center', va='center', fontsize=8)
    ax.text(9.85, 4.1, 'Feature Extract', ha='center', va='center', fontsize=7)
    ax.text(9.85, 3.85, '↓', ha='center', va='center', fontsize=10)
    ax.text(9.85, 3.6, 'Classify', ha='center', va='center', fontsize=7)
    ax.text(9.85, 3.35, '(H, M, Z)', ha='center', va='center', fontsize=7)
    
    # Arrow to output
    ax.annotate('', xy=(11.5, 4), xytext=(11, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(11.25, 4.3, 'Predict', ha='center', fontsize=7)
    
    # Stage 6: Output
    stage6_box = FancyBboxPatch((11.5, 3.5), 2, 1, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='#A5D6A7', linewidth=2)
    ax.add_patch(stage6_box)
    ax.text(12.5, 4.3, 'Stage 6', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(12.5, 4, 'Probabilities', ha='center', va='center', fontsize=8)
    ax.text(12.5, 3.75, 'P(H), P(M), P(Z)', ha='center', va='center', fontsize=7)
    
    # Arrow down to threshold
    ax.annotate('', xy=(12.5, 3), xytext=(12.5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Stage 7: Unknown Detection
    stage7_box = FancyBboxPatch((11.2, 1.5), 2.6, 1.3, boxstyle="round,pad=0.1", 
                                edgecolor='purple', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(stage7_box)
    ax.text(12.5, 2.6, 'Stage 7', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(12.5, 2.3, 'Threshold Check', ha='center', va='center', fontsize=8)
    ax.text(12.5, 2, 'max(P) ≥ θ?', ha='center', va='center', fontsize=7)
    ax.text(12.5, 1.75, 'Yes: Known | No: Unknown', ha='center', va='center', fontsize=6)
    
    # Final arrows
    ax.annotate('', xy=(10.5, 1.9), xytext=(11.2, 1.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(10.3, 1.9, 'Known', ha='right', fontsize=7, color='green', fontweight='bold')
    
    ax.annotate('', xy=(14, 1.9), xytext=(13.8, 1.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(14.2, 1.9, 'Unknown', ha='left', fontsize=7, color='red', fontweight='bold')
    
    # Bottom: Data format info
    format_box = FancyBboxPatch((0.5, 0.2), 5, 1, boxstyle="round,pad=0.1", 
                                edgecolor='gray', facecolor='#F5F5F5', linewidth=1)
    ax.add_patch(format_box)
    ax.text(3, 1, 'Data Formats', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1, 0.7, 'Stage 1: Various (PNG/JPG)', ha='left', va='center', fontsize=7)
    ax.text(1, 0.5, 'Stage 2-3: NumPy arrays', ha='left', va='center', fontsize=7)
    ax.text(1, 0.3, 'Stage 4+: PyTorch tensors (224×224×3)', ha='left', va='center', fontsize=7)
    
    # Bottom right: Metrics
    metrics_box = FancyBboxPatch((8.5, 0.2), 5, 1, boxstyle="round,pad=0.1", 
                                 edgecolor='gray', facecolor='#F5F5F5', linewidth=1)
    ax.add_patch(metrics_box)
    ax.text(11, 1, 'Evaluation Metrics', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(9, 0.7, '• Accuracy', ha='left', va='center', fontsize=7)
    ax.text(9, 0.5, '• Precision/Recall/F1', ha='left', va='center', fontsize=7)
    ax.text(9, 0.3, '• Confusion Matrix', ha='left', va='center', fontsize=7)
    ax.text(11.5, 0.7, '• Loss (Cross-Entropy)', ha='left', va='center', fontsize=7)
    ax.text(11.5, 0.5, '• GradCAM Visualization', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Data flow diagram saved as 'data_flow_diagram.png'")
    plt.show()


def create_system_overview_diagram():
    """
    Create a high-level system overview diagram
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Writer Identification System - Overview', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Main components with circles
    circle_radius = 1.2
    
    # 1. Data Module
    circle1 = Circle((2, 7), circle_radius, edgecolor='blue', facecolor='#BBDEFB', linewidth=3)
    ax.add_patch(circle1)
    ax.text(2, 7.4, 'Data', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, 7, 'Module', ha='center', va='center', fontsize=10)
    ax.text(2, 6.6, 'H-, M-, Z-', ha='center', va='center', fontsize=8)
    
    # 2. Preprocessing Module
    circle2 = Circle((6, 7), circle_radius, edgecolor='orange', facecolor='#FFE0B2', linewidth=3)
    ax.add_patch(circle2)
    ax.text(6, 7.4, 'Preprocessing', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 7, 'Module', ha='center', va='center', fontsize=10)
    ax.text(6, 6.6, 'Clean & Norm', ha='center', va='center', fontsize=8)
    
    # 3. Augmentation Module
    circle3 = Circle((10, 7), circle_radius, edgecolor='green', facecolor='#C8E6C9', linewidth=3)
    ax.add_patch(circle3)
    ax.text(10, 7.4, 'Augmentation', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10, 7, 'Module', ha='center', va='center', fontsize=10)
    ax.text(10, 6.6, 'Train Boost', ha='center', va='center', fontsize=8)
    
    # Arrows between top modules
    ax.annotate('', xy=(4.8, 7), xytext=(3.2, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.annotate('', xy=(8.8, 7), xytext=(7.2, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # 4. Model Module (center, larger)
    circle4 = Circle((6, 4), 1.5, edgecolor='red', facecolor='#FFCDD2', linewidth=4)
    ax.add_patch(circle4)
    ax.text(6, 4.5, 'Model', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(6, 4, 'Module', ha='center', va='center', fontsize=11)
    ax.text(6, 3.6, 'AlexNet', ha='center', va='center', fontsize=9)
    ax.text(6, 3.3, 'Transfer Learn', ha='center', va='center', fontsize=8)
    
    # Arrows to model
    ax.annotate('', xy=(6, 5.5), xytext=(6, 5.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # 5. Evaluation Module
    circle5 = Circle((2, 1.5), circle_radius, edgecolor='purple', facecolor='#E1BEE7', linewidth=3)
    ax.add_patch(circle5)
    ax.text(2, 1.9, 'Evaluation', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, 1.5, 'Module', ha='center', va='center', fontsize=10)
    ax.text(2, 1.1, 'Metrics', ha='center', va='center', fontsize=8)
    
    # 6. Visualization Module
    circle6 = Circle((6, 1.5), circle_radius, edgecolor='teal', facecolor='#B2DFDB', linewidth=3)
    ax.add_patch(circle6)
    ax.text(6, 1.9, 'Visualization', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 1.5, 'Module', ha='center', va='center', fontsize=10)
    ax.text(6, 1.1, 'GradCAM', ha='center', va='center', fontsize=8)
    
    # 7. Detection Module
    circle7 = Circle((10, 1.5), circle_radius, edgecolor='brown', facecolor='#BCAAA4', linewidth=3)
    ax.add_patch(circle7)
    ax.text(10, 1.9, 'Unknown', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10, 1.5, 'Detection', ha='center', va='center', fontsize=10)
    ax.text(10, 1.1, 'Threshold', ha='center', va='center', fontsize=8)
    
    # Arrows from model to bottom modules
    ax.annotate('', xy=(2.8, 2.3), xytext=(5, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6, 2.7), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(9.2, 2.3), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # File associations
    files_box = FancyBboxPatch((0.3, 8.5), 11.4, 0.7, boxstyle="round,pad=0.1", 
                               edgecolor='gray', facecolor='#EEEEEE', linewidth=1)
    ax.add_patch(files_box)
    ax.text(6, 9, 'Python Files', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 8.7, 'preprocessing.py', ha='center', va='center', fontsize=7)
    ax.text(4, 8.7, 'augmentation.py', ha='center', va='center', fontsize=7)
    ax.text(6, 8.7, 'model.py', ha='center', va='center', fontsize=7)
    ax.text(7.7, 8.7, 'train.py', ha='center', va='center', fontsize=7)
    ax.text(9.2, 8.7, 'test_pipeline.py', ha='center', va='center', fontsize=7)
    ax.text(10.8, 8.7, 'gradcam.py', ha='center', va='center', fontsize=7)
    
    # Application box at bottom
    app_box = FancyBboxPatch((3, 0.1), 6, 0.7, boxstyle="round,pad=0.1", 
                             edgecolor='red', facecolor='#FFEBEE', linewidth=3)
    ax.add_patch(app_box)
    ax.text(6, 0.6, 'Application: Proxy Attendance Detection', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='red')
    ax.text(6, 0.3, 'Authenticate student signatures to prevent proxy attendance', 
            ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('system_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("System overview diagram saved as 'system_overview.png'")
    plt.show()


if __name__ == "__main__":
    print("Generating architecture and pipeline diagrams...")
    print("\n1. Creating overall pipeline diagram...")
    create_overall_pipeline_diagram()
    
    print("\n2. Creating model architecture diagram...")
    create_model_architecture_diagram()
    
    print("\n3. Creating data flow diagram...")
    create_data_flow_diagram()
    
    print("\n4. Creating system overview diagram...")
    create_system_overview_diagram()
    
    print("\n✅ All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - pipeline_diagram.png")
    print("  - model_architecture.png")
    print("  - data_flow_diagram.png")
    print("  - system_overview.png")
