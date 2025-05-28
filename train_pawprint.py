import os
import wandb
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
from utils.config import load_config

from data.pawprint import PawprintDataset, PawprintTripletDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.backbone import PawprintModel
from data.sampler import get_balanced_sampler

def create_dataset_and_loader(config, split="train"):
    """Create dataset and dataloader based on mode"""
    transform = get_train_transforms(config) if split == "train" else get_val_transforms(config)
    if config['model']['mode'] == "metric" and config['model']['metric_learning']['weights']['triplet'] > 0:
        print("Using triplet dataset")
        # Create triplet dataset with weighted sampling consideration
        dataset = PawprintTripletDataset(
            root_dir=config['data']['root_dir'],
            data_dir=config['data']['data_dir'],
            split=split,
            transform=transform,
            mining_strategy=config['model']['metric_learning']['triplet']['mining_strategy'],
            num_triplets=1000,
            use_weighted_sampling=config['data'].get('use_weighted_sampling', False)
        )
        
        # Triplet dataset has its own sampling mechanism
        loader = DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=(split == "train"),
            num_workers=config['num_workers'],
            pin_memory=True
        )
    else:
        print("Using regular dataset")
        # Regular dataset with optional weighted sampling
        dataset = PawprintDataset(
            root_dir=config['data']['root_dir'],
            data_dir=config['data']['data_dir'],
            split=split,
            transform=transform
        )
        
        if split == "train" and config['data'].get('use_weighted_sampling', False):
            sampler = get_balanced_sampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = (split == "train")
        
        loader = DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    return dataset, loader

def train_epoch(model, loader, criterion, optimizer, device, ce_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_metric_loss = 0
    all_preds = []
    all_labels = []
    metric_metrics_sum = {}
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # Handle both regular and triplet batches
        if len(batch) == 3:  # Regular batch
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images)
        else:  # Triplet batch
            anchor, positive, negative, (anchor_labels, _, _) = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_labels = anchor_labels.to(device)
            
            # Get embeddings for all images
            anchor_logits, anchor_features = model(anchor)
            pos_logits, pos_features = model(positive)
            neg_logits, neg_features = model(negative)
            
            # Use anchor for classification
            logits, features = anchor_logits, anchor_features
            labels = anchor_labels
        
        # Compute weighted classification loss
        cls_loss = criterion(logits, labels) * ce_weight
        total_ce_loss += cls_loss.item()
        
        # Compute metric learning loss if enabled
        if model.use_metric_learning and features is not None:
            if len(batch) == 3:  # Regular batch
                metric_loss, metric_metrics = model.metric_learning(features, labels)
            else:  # Triplet batch
                metric_loss, metric_metrics = model.metric_learning(
                    features=torch.cat([anchor_features, pos_features, neg_features]),
                    labels=torch.cat([anchor_labels, anchor_labels, anchor_labels])
                )
            
            total_batch_loss = cls_loss + metric_loss
            total_metric_loss += metric_loss.item()
            
            # Accumulate metric learning metrics
            for k, v in metric_metrics.items():
                if isinstance(v, (float, int)):
                    metric_metrics_sum[k] = metric_metrics_sum.get(k, 0) + v
        else:
            total_batch_loss = cls_loss
        
        # Backward and optimize
        total_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        total_loss += total_batch_loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'total_loss': f'{total_batch_loss.item():.4f}',
            'ce_loss': f'{cls_loss.item():.4f}',
            'metric_loss': f'{metric_loss.item():.4f}' if model.use_metric_learning else 'N/A'
        })
    
    # Calculate average metrics
    num_batches = len(loader)
    metrics = {
        'train_total_loss': total_loss / num_batches,
        'train_ce_loss': total_ce_loss / num_batches,
        'train_accuracy': accuracy_score(all_labels, all_preds),
        'train_f1': f1_score(all_labels, all_preds, average='macro'),
        'train_precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'train_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    if model.use_metric_learning:
        metrics['train_metric_loss'] = total_metric_loss / num_batches
        for k, v in metric_metrics_sum.items():
            metrics[f'train_{k}'] = v / num_batches
    
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    env_preds = {}  # predictions per environment
    env_labels = {}  # labels per class
    class_preds = {}  # predictions per class
    class_labels = {}  # labels per class
    
    start_time = time.time()
    for batch in tqdm(loader, desc='Validation'):
        # Handle both regular and triplet batches
        if len(batch) == 3:  # Regular batch
            images, labels, envs = batch
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            
            # Track environment-wise performance
            for env, pred, label in zip(envs.numpy(), 
                                      torch.argmax(logits, dim=1).cpu().numpy(), 
                                      labels.cpu().numpy()):
                if env not in env_preds:
                    env_preds[env] = []
                    env_labels[env] = []
                env_preds[env].append(pred)
                env_labels[env].append(label)
        
        else:  # Triplet batch
            anchor, positive, negative, (anchor_labels, _, _) = batch
            anchor = anchor.to(device)
            anchor_labels = anchor_labels.to(device)
            
            # Only use anchor images for validation metrics
            logits, _ = model(anchor)
            labels = anchor_labels
            
            # No environment tracking for triplet batches
        
        # Common validation logic
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Track class-wise performance
        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            if label not in class_preds:
                class_preds[label] = []
                class_labels[label] = []
            class_preds[label].append(pred)
            class_labels[label].append(label)
    
    end_time = time.time()
    total_time = end_time - start_time
    num_samples = len(loader.dataset)
    time_per_sample = total_time / num_samples
    
    # Calculate overall metrics
    metrics = {
        'val_loss': total_loss / len(loader),
        'val_accuracy': accuracy_score(all_labels, all_preds),
        'val_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'val_precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'val_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'val_time_per_sample': time_per_sample
    }
    

    # Calculate environment-wise metrics if available
    if env_preds:
        env_names = {idx: name for name, idx in loader.dataset.env_to_idx.items()}
        for env in env_preds:
            env_name = env_names[env]
            metrics[f'val_accuracy_{env_name}'] = accuracy_score(
                env_labels[env], env_preds[env]
            )
            metrics[f'val_f1_{env_name}'] = f1_score(
                env_labels[env], env_preds[env], 
                average='macro', zero_division=0
            )
    
    # Calculate class-wise metrics
    class_names = {idx: name for name, idx in loader.dataset.class_to_idx.items()}
    for class_idx in class_preds:
        class_name = class_names[class_idx]
        metrics[f'val_accuracy_{class_name}'] = accuracy_score(
            class_labels[class_idx], class_preds[class_idx]
        )
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train pawprint identification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()

def create_model(config, device):
    """Create model based on config"""
    model = PawprintModel(
        num_classes=config['data']['num_classes'],
        backbone=config['model']['name'],
        pretrained=config['model']['pretrained'],
        mode=config['model']['mode'],
        metric_learning_config={
            'triplet_weight': config['model']['metric_learning']['weights']['triplet'],
            'center_weight': config['model']['metric_learning']['weights']['center'],
            'arcface_weight': config['model']['metric_learning']['weights']['arcface'],
            'triplet': config['model']['metric_learning']['triplet'],
            'center': config['model']['metric_learning']['center'],
            'arcface': config['model']['metric_learning']['arcface']
        } if config['model']['mode'] == "metric" else None
    ).to(device)
    return model

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Initialize wandb
    run = wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        config=config
    )
    
    # Setup device
    device = torch.device(config['device'])
    
    print(config)
    
    # Create datasets
    train_dataset, train_loader = create_dataset_and_loader(config, "train")
    
    val_dataset, val_loader = create_dataset_and_loader(config, "test")
    
    # Update config with actual number of classes
    config['data']['num_classes'] = train_dataset.num_classes
    
    # Create model based on the config
    model = create_model(config, device)
    
    # Log model architecture and trainable parameters
    wandb.watch(model, log='all')
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Get loss weights from config
    ce_weight = config['model'].get('loss_weights', {}).get('ce', 1.0)
    
    # Training loop
    best_val_f1 = 0
    best_val_acc = 0
    best_val_precision = 0
    best_val_recall = 0
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train with CE weight
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            ce_weight=ce_weight
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            best_val_acc = val_metrics['val_accuracy']
            best_val_precision = val_metrics['val_precision']
            best_val_recall = val_metrics['val_recall']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'best_val_acc': best_val_acc,
                'best_val_precision': best_val_precision,
                'best_val_recall': best_val_recall,
                'config': config
            }, f"checkpoints/{wandb.run.name}_best.pth")
    
            # Log all metrics for this epoch
        wandb.log({
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'best_val_precision': best_val_precision,
            'best_val_recall': best_val_recall,
            **train_metrics,
            **val_metrics
        })
    
    wandb.finish()

if __name__ == '__main__':
    main() 