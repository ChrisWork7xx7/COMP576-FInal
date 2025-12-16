"""
LSTM Model for Drowsiness Detection using Facial Landmarks
"""
import torch
import torch.nn as nn
from typing import Tuple

from .config import LANDMARK_DIM_2D, TrainConfig


class LandmarkLSTM(nn.Module):
    """
    LSTM-based model for drowsiness detection from facial landmarks
    
    Architecture:
        Input (batch, seq_len, 136) 
        -> LSTM -> (batch, hidden_size)
        -> FC -> (batch, 2)
    """
    
    def __init__(self, 
                 input_size: int = LANDMARK_DIM_2D,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            input_size: Input feature dimension (136 for 2D landmarks)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super(LandmarkLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # LSTM forward
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        # h_n[-1]: (batch, hidden_size)
        out = h_n[-1]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for potentially better performance
    """
    
    def __init__(self,
                 input_size: int = LANDMARK_DIM_2D,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super(BiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # FC layers (hidden_size * 2 because bidirectional)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate the final forward and backward hidden states
        # h_n shape: (num_layers * 2, batch, hidden_size)
        forward_hidden = h_n[-2]  # Last layer forward
        backward_hidden = h_n[-1]  # Last layer backward
        out = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # FC layers
        out = self.fc(out)
        
        return out


def create_model(config: TrainConfig, bidirectional: bool = False) -> nn.Module:
    """
    Factory function to create model
    
    Args:
        config: Training configuration
        bidirectional: Whether to use bidirectional LSTM
        
    Returns:
        Model instance
    """
    ModelClass = BiLSTM if bidirectional else LandmarkLSTM
    
    model = ModelClass(
        input_size=LANDMARK_DIM_2D,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=2,
        dropout=config.DROPOUT
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    config = TrainConfig()
    model = create_model(config)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = config.WINDOW_SIZE
    x = torch.randn(batch_size, seq_len, LANDMARK_DIM_2D)
    
    out = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")

