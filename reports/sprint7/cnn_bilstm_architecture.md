# CNN-BiLSTM Hybrid Architecture

## 1. Overview

The CNN-BiLSTM hybrid model combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks for multiclass network intrusion detection using the CIC-IDS2017 dataset.

The CNN extracts local patterns from the input features. The BiLSTM processes the resulting feature sequence in both forward and backward directions. A classification layer predicts one of 15 traffic classes.

## 2. Input

- Input features: 70
- Input shape: (Batch Size, 1, 70)
- Number of output classes: 15

## 3. CNN Feature Extraction

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | 70 features | (B, 1, 70) |
| Conv1D | 1 to 16 channels, kernel 3 | (B, 16, 70) |
| ReLU | Activation | (B, 16, 70) |
| MaxPool1D | Pool size 2 | (B, 16, 35) |
| Conv1D | 16 to 32 channels, kernel 3 | (B, 32, 35) |
| ReLU | Activation | (B, 32, 35) |
| MaxPool1D | Pool size 2 | (B, 32, 17) |

The CNN extracts local feature patterns and reduces the sequence length.

## 4. CNN-to-BiLSTM Connection

The CNN output is transposed from (B, 32, 17) to (B, 17, 32).

This makes the 17 feature-map positions the sequence steps and the 32 channels the input features for each step.

## 5. BiLSTM Processing

- Input size: 32
- Hidden size: 32
- Number of layers: 1
- Direction: Bidirectional
- Output size per sequence step: 64

The forward and backward LSTM outputs are combined.

The resulting tensor has shape (B, 17, 64).

Mean pooling across the 17 sequence positions produces a 64-dimensional representation.

## 6. Classification Stage

| Layer | Configuration |
|---|---|
| Mean Pooling | 17 sequence positions to one vector |
| Dropout | 0.2 |
| Fully Connected | 64 to 15 |
| Output | 15 class logits |

CrossEntropyLoss is used during training. Softmax is not applied before this loss function.

## 7. Training Configuration

- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Loss function: CrossEntropyLoss
- Batch size: 8
- Epochs: 2
- Training samples: 6,004
- Total trainable parameters: 19,503

## 8. Data Flow

Network flow features
    -> Input reshaping
    -> CNN feature extraction
    -> Feature-map transposition
    -> BiLSTM sequence processing
    -> Mean pooling
    -> Dropout
    -> Fully connected classification
    -> 15-class prediction

## 9. Architectural Limitation

The CIC-IDS2017 input consists of 70 engineered features per network flow. These features do not represent a chronological sequence of packets or network flows.

Therefore, the BiLSTM processes positions in the CNN feature map rather than actual temporal network traffic. This is an important limitation when interpreting the sequential-learning capability of the hybrid model.

## 10. Model Size

Total trainable parameters: 19,503.

The architecture is relatively compact. However, training and evaluation memory requirements also depend on batch size, input data, intermediate activations, and the available system RAM.
