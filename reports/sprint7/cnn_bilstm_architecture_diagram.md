# CNN-BiLSTM Hybrid Architecture Diagram

```mermaid
flowchart TD
    A["Input: 70 Features (B, 1, 70)"]
    B["Conv1D: 1 to 16, Kernel 3"]
    C["ReLU + MaxPool1D"]
    D["Conv1D: 16 to 32, Kernel 3"]
    E["ReLU + MaxPool1D"]
    F["CNN Output: (B, 32, 17)"]
    G["Transpose: (B, 17, 32)"]
    H["Bidirectional LSTM: Hidden Size 32"]
    I["BiLSTM Output: (B, 17, 64)"]
    J["Mean Pooling: (B, 64)"]
    K["Dropout: 0.2"]
    L["Fully Connected: 64 to 15"]
    M["15-Class Prediction"]

    A --> B --> C --> D --> E --> F
    F --> G --> H --> I --> J --> K --> L --> M
