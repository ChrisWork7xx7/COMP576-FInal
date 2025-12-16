"""
Data loading and preprocessing for DROZY dataset
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .config import (
    DATA_DIR, ANNOTATIONS_DIR, KSS_FILE,
    NUM_SUBJECTS, NUM_TESTS, MISSING_TESTS,
    KSS_THRESHOLD, LANDMARK_DIM_2D, TrainConfig
)


def load_kss_labels() -> np.ndarray:
    """
    Load KSS labels from KSS.txt
    
    Returns:
        np.ndarray: Shape (14, 3), KSS scores for each subject and test
    """
    kss = np.loadtxt(KSS_FILE, dtype=int)
    return kss


def get_valid_tests(config: TrainConfig) -> List[Tuple[int, int]]:
    """
    Get list of valid (subject_id, test_id) pairs
    
    Args:
        config: Training configuration
        
    Returns:
        List of (subject_id, test_id) tuples (1-indexed)
    """
    valid_tests = []
    
    subjects = config.DEBUG_SUBJECTS if config.DEBUG_MODE else list(range(1, NUM_SUBJECTS + 1))
    
    for subject in subjects:
        for test in range(1, NUM_TESTS + 1):
            if (subject, test) not in MISSING_TESTS:
                valid_tests.append((subject, test))
    
    return valid_tests


def load_landmarks(subject: int, test: int, use_3d: bool = False) -> np.ndarray:
    """
    Load facial landmarks for a specific test
    
    Args:
        subject: Subject ID (1-indexed)
        test: Test ID (1-indexed)
        use_3d: Whether to load 3D landmarks (default: 2D)
        
    Returns:
        np.ndarray: Shape (num_frames, landmark_dim)
    """
    suffix = "s3" if use_3d else "s2"
    filename = ANNOTATIONS_DIR / f"{subject}-{test}-{suffix}.txt"
    
    if not filename.exists():
        raise FileNotFoundError(f"Landmark file not found: {filename}")
    
    # Load landmarks - each line is space-separated values
    landmarks = np.loadtxt(filename)
    
    return landmarks


def create_windows(landmarks: np.ndarray, 
                   window_size: int, 
                   stride: int,
                   sample_rate: int = 1) -> np.ndarray:
    """
    Create sliding windows from landmark sequence
    
    Args:
        landmarks: Shape (num_frames, landmark_dim)
        window_size: Number of frames per window
        stride: Stride between windows
        sample_rate: Take every Nth frame before windowing
        
    Returns:
        np.ndarray: Shape (num_windows, window_size, landmark_dim)
    """
    # First, subsample the data
    landmarks = landmarks[::sample_rate]
    
    num_frames = len(landmarks)
    
    if num_frames < window_size:
        # Pad if too short
        pad_size = window_size - num_frames
        landmarks = np.pad(landmarks, ((0, pad_size), (0, 0)), mode='edge')
        num_frames = window_size
    
    # Create windows
    windows = []
    for start in range(0, num_frames - window_size + 1, stride):
        window = landmarks[start:start + window_size]
        windows.append(window)
    
    if len(windows) == 0:
        # If no windows created, use entire sequence as one window
        windows.append(landmarks[:window_size])
    
    return np.array(windows)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to zero mean and unit variance
    
    Args:
        landmarks: Shape (..., landmark_dim)
        
    Returns:
        Normalized landmarks
    """
    # Compute mean and std across all data
    mean = landmarks.mean(axis=(0, 1), keepdims=True)
    std = landmarks.std(axis=(0, 1), keepdims=True) + 1e-8
    
    return (landmarks - mean) / std


class DrowsinessDataset(Dataset):
    """
    PyTorch Dataset for drowsiness detection
    """
    
    def __init__(self, 
                 test_list: List[Tuple[int, int]],
                 kss_labels: np.ndarray,
                 config: TrainConfig,
                 normalize: bool = True):
        """
        Args:
            test_list: List of (subject_id, test_id) tuples
            kss_labels: KSS scores array
            config: Training configuration
            normalize: Whether to normalize landmarks
        """
        self.config = config
        self.samples = []  # List of (window, label)
        
        print(f"Loading data for {len(test_list)} tests...")
        
        all_windows = []
        all_labels = []
        
        for subject, test in test_list:
            try:
                # Load landmarks
                landmarks = load_landmarks(subject, test, use_3d=False)
                
                # Get KSS label (0-indexed in array)
                kss = kss_labels[subject - 1, test - 1]
                
                # Skip if KSS is 0 (invalid)
                if kss == 0:
                    print(f"  Skipping {subject}-{test}: KSS=0")
                    continue
                
                # Convert to binary label
                label = 0 if kss <= KSS_THRESHOLD else 1  # 0=Alert, 1=Drowsy
                
                # Create windows
                windows = create_windows(
                    landmarks,
                    window_size=config.WINDOW_SIZE,
                    stride=config.WINDOW_STRIDE,
                    sample_rate=config.SAMPLE_RATE
                )
                
                all_windows.append(windows)
                all_labels.extend([label] * len(windows))
                
                print(f"  Loaded {subject}-{test}: {len(windows)} windows, KSS={kss}, label={label}")
                
            except FileNotFoundError as e:
                print(f"  Skipping {subject}-{test}: {e}")
                continue
        
        # Concatenate all windows
        if all_windows:
            all_windows = np.concatenate(all_windows, axis=0)
            
            # Normalize if requested
            if normalize:
                all_windows = normalize_landmarks(all_windows)
            
            # Convert to tensors
            self.windows = torch.FloatTensor(all_windows)
            self.labels = torch.LongTensor(all_labels)
        else:
            self.windows = torch.FloatTensor([])
            self.labels = torch.LongTensor([])
        
        print(f"Total samples: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"Class distribution: Alert={sum(self.labels == 0)}, Drowsy={sum(self.labels == 1)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders using Leave-One-Subject-Out
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load KSS labels
    kss_labels = load_kss_labels()
    
    # Get valid tests
    valid_tests = get_valid_tests(config)
    
    # Split by subject (Leave-One-Subject-Out)
    train_tests = [(s, t) for s, t in valid_tests if s != config.TEST_SUBJECT]
    test_tests = [(s, t) for s, t in valid_tests if s == config.TEST_SUBJECT]
    
    print(f"\n{'='*50}")
    print(f"Data Split (Leave-One-Subject-Out)")
    print(f"{'='*50}")
    print(f"Train subjects: {sorted(set(s for s, t in train_tests))}")
    print(f"Test subject: {config.TEST_SUBJECT}")
    print(f"Train tests: {len(train_tests)}, Test tests: {len(test_tests)}")
    
    # Create datasets
    print(f"\n--- Loading Training Data ---")
    train_dataset = DrowsinessDataset(train_tests, kss_labels, config)
    
    print(f"\n--- Loading Test Data ---")
    test_dataset = DrowsinessDataset(test_tests, kss_labels, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    config = TrainConfig()
    train_loader, test_loader = create_dataloaders(config)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check one batch
    for X, y in train_loader:
        print(f"\nBatch shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Labels: {y}")
        break

