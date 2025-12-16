#!/usr/bin/env python3
"""
Drowsiness Detection - Baseline Model (Facial Landmarks Only)

This script runs the baseline model using only facial landmarks (2D).
It serves as a comparison point for multimodal approaches.

Usage:
    python run_baseline.py [--debug] [--epochs N] [--test-subject N]

Example:
    python run_baseline.py --debug --epochs 10  # Quick debug run
    python run_baseline.py --epochs 50          # Full training
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig, FullTrainConfig
from src.data_loader import create_dataloaders
from src.train import train_model
from src.utils import (
    plot_training_history, 
    plot_confusion_matrix, 
    print_metrics_summary,
    set_seed
)


def parse_args():
    parser = argparse.ArgumentParser(description='Drowsiness Detection Baseline Model')
    
    parser.add_argument('--debug', action='store_true', 
                        help='Use debug mode with fewer subjects')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--test-subject', type=int, default=None,
                        help='Subject ID to use for testing (Leave-One-Out)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='LSTM hidden size')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting (for headless environments)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save trained model')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Select configuration
    if args.debug:
        config = TrainConfig()
        print("Running in DEBUG mode (fewer subjects, quick training)")
    else:
        config = FullTrainConfig()
        print("Running in FULL mode")
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.test_subject is not None:
        config.TEST_SUBJECT = args.test_subject
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.hidden_size is not None:
        config.HIDDEN_SIZE = args.hidden_size
    
    # Set random seed
    set_seed(config.SEED)
    
    # Print configuration
    print("\n" + "="*60)
    print("DROWSINESS DETECTION - BASELINE MODEL (Landmarks Only)")
    print("="*60)
    print(f"Debug Mode:     {config.DEBUG_MODE}")
    print(f"Subjects:       {config.DEBUG_SUBJECTS if config.DEBUG_MODE else 'All 14'}")
    print(f"Test Subject:   {config.TEST_SUBJECT}")
    print(f"Epochs:         {config.EPOCHS}")
    print(f"Batch Size:     {config.BATCH_SIZE}")
    print(f"Learning Rate:  {config.LEARNING_RATE}")
    print(f"Hidden Size:    {config.HIDDEN_SIZE}")
    print(f"Window Size:    {config.WINDOW_SIZE}")
    print(f"Sample Rate:    1/{config.SAMPLE_RATE} frames")
    print("="*60)
    
    # Create data loaders
    print("\n[Step 1] Loading and preprocessing data...")
    train_loader, test_loader = create_dataloaders(config)
    
    if len(train_loader) == 0:
        print("ERROR: No training data loaded!")
        return
    
    if len(test_loader) == 0:
        print("WARNING: No test data loaded!")
    
    # Train model
    print("\n[Step 2] Training model...")
    model, history = train_model(train_loader, test_loader, config)
    
    # Print final metrics
    if 'final_metrics' in history:
        print_metrics_summary(history['final_metrics'])
    
    # Plot results
    if not args.no_plot:
        print("\n[Step 3] Plotting results...")
        try:
            plot_training_history(history, save_path='training_history.png')
            if 'final_metrics' in history:
                plot_confusion_matrix(
                    history['final_metrics']['confusion_matrix'],
                    save_path='confusion_matrix.png'
                )
        except Exception as e:
            print(f"Could not plot: {e}")
    
    # Save model
    if args.save_model:
        import torch
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history
        }, args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETE")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    main()

