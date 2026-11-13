
import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """CNN component for extracting network traffic features."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # First convolutional block
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Second convolutional block
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.features(x)

class BiLSTMSequenceProcessor(nn.Module):
    """BiLSTM component for learning patterns from CNN features."""

    def __init__(self):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.bilstm(x)

        # Combine information across all 17 sequence positions
        x = x.mean(dim=1)

        x = self.dropout(x)

        return x

class CNNBiLSTM(nn.Module):
    """Hybrid CNN-BiLSTM model for network intrusion classification."""

    def __init__(self, num_classes=15):
        super().__init__()

        self.cnn = CNNFeatureExtractor()
        self.bilstm = BiLSTMSequenceProcessor()

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):

        # CNN feature extraction
        x = self.cnn(x)

        # Convert (B, 32, 17) to (B, 17, 32)
        x = x.transpose(1, 2)

        # BiLSTM sequence processing
        x = self.bilstm(x)

        # Final classification
        x = self.classifier(x)

        return x