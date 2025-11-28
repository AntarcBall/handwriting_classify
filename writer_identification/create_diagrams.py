import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from matplotlib import font_manager
import os


def create_pipeline_diagram():
    """Create a diagram showing the complete pipeline flow"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(8, 9.5, 'Offline Handwriting-Based Proxy Attendance Detection System', 
            fontsize=16, fontweight='bold', ha='center')
    
    # 1. Data Preprocessing Block
    rect1 = FancyBboxPatch((0.5, 7.5), 2, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightblue', 
                          edgecolor='blue', 
                          linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.5, 8.5, 'Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 8.2, 'Preprocessing', fontsize=10, ha='center')
    ax.text(1.5, 7.9, '(Noise Removal,', fontsize=9, ha='center')
    ax.text(1.5, 7.7, 'Binarization)', fontsize=9, ha='center')
    
    # 2. Data Augmentation Block
    rect2 = FancyBboxPatch((3.5, 7.5), 2, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', 
                          edgecolor='green', 
                          linewidth=2)
    ax.add_patch(rect2)
    ax.text(4.5, 8.5, 'Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.5, 8.2, 'Augmentation', fontsize=10, ha='center')
    ax.text(4.5, 7.9, '(Rotation,', fontsize=9, ha='center')
    ax.text(4.5, 7.7, 'Elastic Distortion)', fontsize=9, ha='center')
    
    # 3. AlexNet Model Block
    rect3 = FancyBboxPatch((6.5, 7.5), 3, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightyellow', 
                          edgecolor='orange', 
                          linewidth=2)
    ax.add_patch(rect3)
    ax.text(8, 8.5, 'AlexNet', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 8.2, 'Transfer Learning', fontsize=10, ha='center')
    ax.text(8, 7.9, '(3 Output Classes)', fontsize=9, ha='center')
    
    # 4. Confidence Thresholding Block
    rect4 = FancyBboxPatch((10.5, 7.5), 2.5, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', 
                          edgecolor='red', 
                          linewidth=2)
    ax.add_patch(rect4)
    ax.text(11.75, 8.5, 'Confidence', fontsize=12, fontweight='bold', ha='center')
    ax.text(11.75, 8.2, 'Thresholding', fontsize=10, ha='center')
    ax.text(11.75, 7.9, '(Unknown Writer', fontsize=9, ha='center')
    ax.text(11.75, 7.7, 'Detection)', fontsize=9, ha='center')
    
    # Arrows between blocks
    ax.annotate('', xy=(3.5, 8.25), xytext=(2.5, 8.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(6.5, 8.25), xytext=(5.5, 8.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(10.5, 8.25), xytext=(9.5, 8.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Input Data
    input_text = "Raw Handwriting Images\n(H, M, Z Signatures)"
    ax.text(0.5, 6.5, input_text, fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    # Output
    output_text = "Attendance Decision\n(Granted/Denied)"
    ax.text(13.5, 6.5, output_text, fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    # Arrow from input to first block
    ax.annotate('', xy=(0.5, 7.5), xytext=(0.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Arrow from last block to output
    ax.annotate('', xy=(13.5, 7.5), xytext=(13.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Grad-CAM Visualization
    rect5 = FancyBboxPatch((6.5, 5.5), 3, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightpink', 
                          edgecolor='purple', 
                          linewidth=2)
    ax.add_patch(rect5)
    ax.text(8, 6.5, 'Grad-CAM', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 6.2, 'Visualization', fontsize=10, ha='center')
    ax.text(8, 5.9, '(Model Attention)', fontsize=9, ha='center')
    
    # Connection from AlexNet to Grad-CAM
    connection = ConnectionPatch((8, 7.5), (8, 7), 
                                "data", "data", 
                                axesA=ax, axesB=ax,
                                arrowstyle="->", 
                                shrinkA=5, shrinkB=5,
                                mutation_scale=20, 
                                fc="purple", 
                                ec="purple",
                                lw=2)
    ax.add_patch(connection)
    
    # Dashed line to show visualization output
    ax.annotate('', xy=(8, 5.5), xytext=(8, 4.8),
                arrowprops=dict(arrowstyle='->', lw=1, color='purple', linestyle='--'))
    ax.text(8, 4.5, 'Attention Heatmap', fontsize=9, ha='center', color='purple')
    
    # Add legend
    legend_text = """
    Colors Legend:
    Blue: Data Preprocessing
    Green: Data Augmentation  
    Yellow: Neural Network
    Red: Decision Making
    Purple: Visualization
    """
    ax.text(12, 4, legend_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.axis('off')
    plt.tight_layout()
    
    output_path = 'pipeline_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Pipeline diagram saved as {output_path}")


def create_architecture_diagram():
    """Create a detailed architecture diagram of the AlexNet model"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    
    # Title
    ax.text(7, 11.5, 'AlexNet-based Architecture for Writer Identification', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Input layer
    input_box = FancyBboxPatch((1, 9), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='blue', 
                              linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 9.5, 'Input', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 9.2, '224×224×3', fontsize=9, ha='center')
    
    # Conv layers
    conv1_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='green', 
                              linewidth=1)
    ax.add_patch(conv1_box)
    ax.text(1.25, 8, 'Conv1', fontsize=8, ha='center')
    
    conv2_box = FancyBboxPatch((2.5, 7.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='green', 
                              linewidth=1)
    ax.add_patch(conv2_box)
    ax.text(3.25, 8, 'Conv2', fontsize=8, ha='center')
    
    conv3_box = FancyBboxPatch((4.5, 7.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='green', 
                              linewidth=1)
    ax.add_patch(conv3_box)
    ax.text(5.25, 8, 'Conv3', fontsize=8, ha='center')
    
    conv4_box = FancyBboxPatch((6.5, 7.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='green', 
                              linewidth=1)
    ax.add_patch(conv4_box)
    ax.text(7.25, 8, 'Conv4', fontsize=8, ha='center')
    
    conv5_box = FancyBboxPatch((8.5, 7.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='green', 
                              linewidth=1)
    ax.add_patch(conv5_box)
    ax.text(9.25, 8, 'Conv5', fontsize=8, ha='center')
    
    # Pooling layers
    pool1_box = FancyBboxPatch((0.5, 6), 1.5, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', 
                              edgecolor='orange', 
                              linewidth=1)
    ax.add_patch(pool1_box)
    ax.text(1.25, 6.4, 'Pool1', fontsize=7, ha='center')
    
    pool2_box = FancyBboxPatch((2.5, 6), 1.5, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', 
                              edgecolor='orange', 
                              linewidth=1)
    ax.add_patch(pool2_box)
    ax.text(3.25, 6.4, 'Pool2', fontsize=7, ha='center')
    
    pool3_box = FancyBboxPatch((8.5, 6), 1.5, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', 
                              edgecolor='orange', 
                              linewidth=1)
    ax.add_patch(pool3_box)
    ax.text(9.25, 6.4, 'Pool3', fontsize=7, ha='center')
    
    # FC layers
    fc1_box = FancyBboxPatch((3, 4), 2, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightcoral', 
                            edgecolor='red', 
                            linewidth=1)
    ax.add_patch(fc1_box)
    ax.text(4, 4.4, 'FC1 (4096)', fontsize=8, ha='center')
    
    fc2_box = FancyBboxPatch((6, 4), 2, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightcoral', 
                            edgecolor='red', 
                            linewidth=1)
    ax.add_patch(fc2_box)
    ax.text(7, 4.4, 'FC2 (4096)', fontsize=8, ha='center')
    
    # Output layer
    output_box = FancyBboxPatch((9, 4), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightpink', 
                               edgecolor='purple', 
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(10, 4.4, 'Output (3 classes)', fontsize=8, fontweight='bold', ha='center')
    
    # Arrows showing flow
    # Input to Conv1
    ax.annotate('', xy=(1.25, 9), xytext=(1.25, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Between Conv layers
    for i in range(5):
        x_pos = 1.25 + i * 2
        if i < 4:
            ax.annotate('', xy=(x_pos + 1.5, 8), xytext=(x_pos + 1.5, 8),
                        xycoords='data', arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Conv to Pool
    for i, (conv_x, pool_x) in enumerate([(1.25, 1.25), (3.25, 3.25), (9.25, 9.25)]):
        ax.annotate('', xy=(pool_x, 7.5), xytext=(conv_x, 8.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Pool to next Conv (where applicable)
    ax.annotate('', xy=(3.25, 7.5), xytext=(1.25, 6.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(5.25, 7.5), xytext=(3.25, 6.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Conv5 to FC layers
    ax.annotate('', xy=(5, 4), xytext=(9.25, 7.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Between FC layers
    ax.annotate('', xy=(5, 4.4), xytext=(5, 4.4),
                xycoords='data', arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # FC to Output
    ax.annotate('', xy=(9, 4.4), xytext=(6, 4.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Reshape operations
    reshape1 = FancyBboxPatch((5.5, 5.5), 2, 0.5, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightgray', 
                             edgecolor='gray', 
                             linewidth=1,
                             linestyle='--')
    ax.add_patch(reshape1)
    ax.text(6.5, 5.75, 'Reshape/Flatten', fontsize=7, ha='center')
    
    # Add text annotations
    ax.text(0.5, 10.5, 'Architecture Overview:', fontsize=10, fontweight='bold')
    ax.text(0.5, 10.2, '• 5 Conv + 3 Pool + 3 FC layers', fontsize=8)
    ax.text(0.5, 10.0, '• Transfer learning from ImageNet', fontsize=8)
    ax.text(0.5, 9.8, '• Modified last layer: 1000→3 classes', fontsize=8)
    ax.text(0.5, 9.6, '• For H, M, Z writer classification', fontsize=8)
    
    # Legend
    legend_elements = [
        'Green: Conv Layers',
        'Yellow: Pooling',
        'Red: Fully Connected',
        'Pink: Output Layer'
    ]
    
    for i, elem in enumerate(legend_elements):
        ax.text(11, 8 - i*0.4, elem, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray'))
    
    ax.axis('off')
    plt.tight_layout()
    
    output_path = 'architecture_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Architecture diagram saved as {output_path}")


def create_system_workflow_diagram():
    """Create a workflow diagram showing the system process"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(6, 9.5, 'System Workflow for Proxy Attendance Detection', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Step 1: Input
    step1 = FancyBboxPatch((5, 8), 2, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightblue', 
                          edgecolor='blue', 
                          linewidth=2)
    ax.add_patch(step1)
    ax.text(6, 8.5, 'Step 1: Input', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 8.2, 'Handwriting Image', fontsize=8, ha='center')
    
    # Step 2: Preprocessing
    step2 = FancyBboxPatch((5, 6.5), 2, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', 
                          edgecolor='green', 
                          linewidth=2)
    ax.add_patch(step2)
    ax.text(6, 7.0, 'Step 2: Preprocess', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 6.7, 'Noise removal, Binarization', fontsize=8, ha='center')
    
    # Step 3: Feature Extraction
    step3 = FancyBboxPatch((5, 5), 2, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightyellow', 
                          edgecolor='orange', 
                          linewidth=2)
    ax.add_patch(step3)
    ax.text(6, 5.5, 'Step 3: Feature Extraction', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 5.2, 'CNN-based feature learning', fontsize=8, ha='center')
    
    # Step 4: Classification
    step4 = FancyBboxPatch((3, 3.5), 2, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', 
                          edgecolor='red', 
                          linewidth=2)
    ax.add_patch(step4)
    ax.text(4, 4.0, 'Step 4a: Classification', fontsize=10, fontweight='bold', ha='center')
    ax.text(4, 3.7, 'Get H/M/Z probabilities', fontsize=8, ha='center')
    
    # Step 5: Threshold Check
    step5 = FancyBboxPatch((7, 3.5), 2, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightpink', 
                          edgecolor='purple', 
                          linewidth=2)
    ax.add_patch(step5)
    ax.text(8, 4.0, 'Step 4b: Threshold Check', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 3.7, 'Confidence > Threshold?', fontsize=8, ha='center')
    
    # Step 6a: Known Writer
    step6a = FancyBboxPatch((2, 2), 2, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightgreen', 
                           edgecolor='green', 
                           linewidth=2)
    ax.add_patch(step6a)
    ax.text(3, 2.4, 'Known Writer', fontsize=9, fontweight='bold', ha='center')
    ax.text(3, 2.1, 'Attendance GRANTED', fontsize=8, ha='center')
    
    # Step 6b: Unknown Writer
    step6b = FancyBboxPatch((8, 2), 2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='lightcoral',
                           edgecolor='red',
                           linewidth=2)
    ax.add_patch(step6b)
    ax.text(9, 2.4, 'Unknown Writer', fontsize=9, fontweight='bold', ha='center')
    ax.text(9, 2.1, 'Attendance DENIED', fontsize=8, ha='center')
    
    # Decision diamond
    diamond_x = [6, 6.5, 6, 5.5]
    diamond_y = [3.5, 3, 2.5, 3]
    diamond = patches.Polygon(list(zip(diamond_x, diamond_y)), 
                             facecolor='lightgray', 
                             edgecolor='black', 
                             linewidth=2)
    ax.add_patch(diamond)
    ax.text(6, 3.05, 'Decision', fontsize=8, fontweight='bold', ha='center')
    ax.text(6, 2.85, 'Point', fontsize=7, ha='center')
    
    # Arrows showing workflow
    ax.annotate('', xy=(6, 8), xytext=(6, 9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(6, 6.5), xytext=(6, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(6, 5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(4, 3.5), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(8, 3.5), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(3, 2.8), xytext=(5.5, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.annotate('', xy=(9, 2.8), xytext=(6.5, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add legend
    legend_text = """
    Blue: Input Processing
    Green: Data Operations
    Yellow: Feature Learning
    Red: Decision Making
    Pink: Confidence Check
    """
    ax.text(0.5, 5, legend_text, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Add decision logic text
    ax.text(1, 1, 'If max(probabilities) > threshold:\n  → Known writer (GRANTED)\nElse:\n  → Unknown writer (DENIED)', 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    ax.axis('off')
    plt.tight_layout()
    
    output_path = 'workflow_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Workflow diagram saved as {output_path}")


if __name__ == "__main__":
    print("Creating system diagrams...")
    
    create_pipeline_diagram()
    print()
    
    create_architecture_diagram()
    print()
    
    create_system_workflow_diagram()
    print()
    
    print("All diagrams have been created successfully!")
    print("- pipeline_diagram.png: Shows the complete system pipeline")
    print("- architecture_diagram.png: Shows the AlexNet model architecture") 
    print("- workflow_diagram.png: Shows the decision workflow for attendance")