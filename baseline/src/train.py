"""
Training and evaluation functions for Drowsiness Detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time

from .config import TrainConfig
from .model import create_model, count_parameters


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict:
    """
    Evaluate model on a dataset
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def compute_class_weights(dataloader: DataLoader) -> torch.Tensor:
    """
    Compute class weights based on class distribution in training data
    
    Args:
        dataloader: Training data loader
        
    Returns:
        Tensor of class weights [weight_class_0, weight_class_1]
    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels, minlength=2)
    
    # Compute weights: total / (num_classes * count_per_class)
    total = len(all_labels)
    weights = total / (2.0 * class_counts + 1e-6)
    
    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()
    
    print(f"\nClass distribution: Alert={class_counts[0]}, Drowsy={class_counts[1]}")
    print(f"Class weights: Alert={weights[0]:.3f}, Drowsy={weights[1]:.3f}")
    
    return torch.FloatTensor(weights)


def train_model(train_loader: DataLoader,
                test_loader: DataLoader,
                config: TrainConfig,
                use_class_weights: bool = True) -> Tuple[nn.Module, Dict]:
    """
    Full training loop
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        config: Training configuration
        use_class_weights: Whether to use class weights to handle imbalance
        
    Returns:
        Tuple of (trained_model, history)
    """
    # Device setup
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Loss with optional class weights
    if use_class_weights:
        class_weights = compute_class_weights(train_loader).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted loss function")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard loss function")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': []
    }
    
    best_f1 = 0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Training for {config.EPOCHS} epochs")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
        
        # Save best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation (Best Model)")
    print(f"{'='*60}")
    
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1-Score:  {final_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Alert  Drowsy")
    print(f"Actual Alert   {final_metrics['confusion_matrix'][0,0]:4d}   {final_metrics['confusion_matrix'][0,1]:4d}")
    print(f"       Drowsy  {final_metrics['confusion_matrix'][1,0]:4d}   {final_metrics['confusion_matrix'][1,1]:4d}")
    
    history['final_metrics'] = final_metrics
    
    return model, history


if __name__ == "__main__":
    from .data_loader import create_dataloaders
    
    config = TrainConfig()
    train_loader, test_loader = create_dataloaders(config)
    
    model, history = train_model(train_loader, test_loader, config)

