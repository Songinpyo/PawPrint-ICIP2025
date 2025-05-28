import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import json
from tqdm import tqdm

from data.pawprint import PawprintDataset
from data.transforms import get_val_transforms
from models.backbone import PawprintModel

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    
    model = PawprintModel(
        num_classes=config['data']['num_classes'],
        backbone=config['model']['name'],
        pretrained=False,
        mode=config['model'].get('mode', 'full')
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and collect predictions"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_features = []
    
    for images, labels, _ in tqdm(loader, desc='Evaluating'):
        images = images.to(device)
        
        logits, features = model(images)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
        if features is not None:
            all_features.extend(features.cpu().numpy())
    
    return (np.array(all_preds), np.array(all_labels), 
            np.array(all_features) if all_features else None)

def analyze_results(preds, labels, dataset, save_dir):
    """Analyze and save evaluation results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names
    class_names = {idx: name for name, idx in dataset.class_to_idx.items()}
    class_name_list = [class_names[i] for i in range(len(class_names))]
    
    # Overall metrics
    report = classification_report(
        labels, preds, 
        target_names=class_name_list,
        output_dict=True,
        zero_division=0
    )
    
    # Save classification report
    with open(save_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Confusion matrix (raw counts)
    cm = confusion_matrix(labels, preds)
    
    # Normalize confusion matrix (convert to percentages)
    cm_percentage = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
    
    # Plot both raw counts and percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='BuGn',
        xticklabels=class_name_list,
        yticklabels=class_name_list,
        ax=ax1
    )
    ax1.set_title('Confusion Matrix - Counts')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.tick_params(axis='y', rotation=45)
    
    # Percentages
    sns.heatmap(
        cm_percentage,
        annot=True, 
        fmt='.1f',  # Show one decimal place
        cmap='BuGn',
        xticklabels=class_name_list,
        yticklabels=class_name_list,
        ax=ax2
    )
    ax2.set_title('Confusion Matrix - Percentages (%)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.tick_params(axis='x', rotation=45, ha='right')
    ax2.tick_params(axis='y', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    # Calculate per-class metrics
    class_metrics = {}
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        if np.sum(mask) > 0:
            class_metrics[class_names[class_idx]] = {
                'accuracy': np.mean(preds[mask] == labels[mask]),
                'support': int(np.sum(mask))
            }
    
    # Save detailed results
    results = {
        'overall_accuracy': report['accuracy'],
        'overall_macro_f1': report['macro avg']['f1-score'],
        'per_class_metrics': {
            class_names[i]: {
                'precision': report[class_names[i]]['precision'],
                'recall': report[class_names[i]]['recall'],
                'f1-score': report[class_names[i]]['f1-score'],
                'support': report[class_names[i]]['support']
            } for i in range(len(class_names))
        }
    }
    
    with open(save_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=4)

def analyze_metric_learning(features, labels, save_dir):
    """Analyze metric learning results"""
    if features is None:
        return
    
    # Compute pairwise distances
    distances = torch.cdist(
        torch.tensor(features), 
        torch.tensor(features)
    ).numpy()
    
    # Compute intra-class and inter-class distances
    labels = np.array(labels)
    intra_class_distances = []
    inter_class_distances = []
    
    for i in range(len(labels)):
        same_class = labels == labels[i]
        same_class[i] = False  # Exclude self
        
        if same_class.any():
            intra_class_distances.extend(distances[i, same_class].tolist())
        inter_class_distances.extend(distances[i, labels != labels[i]].tolist())
    
    # Plot distance distributions
    plt.figure(figsize=(10, 6))
    plt.hist(intra_class_distances, bins=50, alpha=0.5, label='Intra-class')
    plt.hist(inter_class_distances, bins=50, alpha=0.5, label='Inter-class')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Feature Distance Distribution')
    plt.legend()
    plt.savefig(save_dir / 'distance_distribution.png')
    plt.close()
    
    # Save distance statistics
    distance_stats = {
        'intra_class': {
            'mean': float(np.mean(intra_class_distances)),
            'std': float(np.std(intra_class_distances)),
            'min': float(np.min(intra_class_distances)),
            'max': float(np.max(intra_class_distances))
        },
        'inter_class': {
            'mean': float(np.mean(inter_class_distances)),
            'std': float(np.std(inter_class_distances)),
            'min': float(np.min(inter_class_distances)),
            'max': float(np.max(inter_class_distances))
        }
    }
    
    with open(save_dir / 'distance_statistics.json', 'w') as f:
        json.dump(distance_stats, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pawprint identification model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/PawPrint',
                      help='Path to data directory')
    parser.add_argument('--save-dir', type=str, default='./eval_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and config from checkpoint
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Create dataset and dataloader
    test_dataset = PawprintDataset(
        root_dir=args.data_dir,
        data_dir=config['data']['data_dir'],  # Use data_dir from config
        split='test',
        transform=get_val_transforms(config)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    try:
        # Evaluate model
        predictions, labels, features = evaluate(model, test_loader, device)
        
        # Analyze and save results
        save_dir = Path(args.save_dir) / Path(args.checkpoint).stem
        analyze_results(predictions, labels, test_dataset, save_dir)
        
        # Analyze metric learning results if features are available
        if features is not None:
            analyze_metric_learning(features, labels, save_dir)
        
        print(f"Evaluation results saved to {save_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    main() 