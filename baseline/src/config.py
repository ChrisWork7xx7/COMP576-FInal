"""
Configuration file for the Drowsiness Detection Baseline Model
"""
import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "DROZY"
ANNOTATIONS_DIR = DATA_DIR / "annotations-auto"
KSS_FILE = DATA_DIR / "KSS.txt"

# =============================================================================
# Data Configuration
# =============================================================================
# Number of subjects and tests
NUM_SUBJECTS = 14
NUM_TESTS = 3

# Missing tests (subject_id, test_id) - 1-indexed
MISSING_TESTS = [
    (7, 1),   # Test did not happen
    (9, 1),   # Backup lost
    (10, 2),  # Backup lost
    (12, 2),  # Backup lost
    (12, 3),  # Backup lost
    (13, 3),  # Backup lost
]

# KSS threshold for binary classification
# Alert: KSS <= 5, Drowsy: KSS >= 6
KSS_THRESHOLD = 5

# Landmark dimensions
LANDMARK_DIM_2D = 136  # 68 points * 2 (x, y)
LANDMARK_DIM_3D = 204  # 68 points * 3 (x, y, z)

# =============================================================================
# Training Configuration (Fast Debug Mode)
# =============================================================================
class TrainConfig:
    # Data sampling
    SAMPLE_RATE = 60   # Take 1 frame every N frames (60 ≈ 2 seconds at 30fps)
    WINDOW_SIZE = 30   # Number of frames per sample (sliding window)
    WINDOW_STRIDE = 10 # Stride for sliding window (more overlap = more samples)
    
    # Quick debug mode - use subset of subjects
    DEBUG_MODE = True
    DEBUG_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Use 11 subjects for testing
    
    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 8  # Smaller batch size for limited data
    LEARNING_RATE = 0.001
    
    # Model parameters
    HIDDEN_SIZE = 128  # Increased for more capacity
    NUM_LAYERS = 1
    DROPOUT = 0.3
    
    # Data split
    TEST_SUBJECT = 11  # Leave this subject out for testing
    
    # Device
    DEVICE = "cuda"  # Will fallback to CPU if CUDA not available
    
    # Random seed for reproducibility
    SEED = 42

# =============================================================================
# Full Training Configuration (for final experiments)
# =============================================================================
class FullTrainConfig(TrainConfig):
    DEBUG_MODE = False
    DEBUG_SUBJECTS = list(range(1, 15))  # All 14 subjects
    EPOCHS = 50
    SAMPLE_RATE = 100  # More data
    WINDOW_SIZE = 60   # Larger window

