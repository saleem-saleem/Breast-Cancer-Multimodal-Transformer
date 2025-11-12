import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧩 1. Classification Head Definition
# ============================================================

class ClassificationHead(nn.Module):
    """
    Multi-class classification head.

    Args:
        input_dim (int): Dimension of input embedding (e.g., 256 from Transformer)
        num_classes (int): Number of recurrence types or cancer risk categories
        hidden_dim (int): Hidden layer dimension (for optional MLP)
        dropout (float): Dropout probability for regularization
        use_mlp (bool): If True, adds one hidden layer between input and output
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_mlp: bool = True
    ):
        super(ClassificationHead, self).__init__()

        self.use_mlp = use_mlp
        self.dropout = nn.Dropout(dropout)

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — fused patient embeddings
        Returns:
            logits: (B, num_classes)
        """
        x = self.dropout(x)
        if self.use_mlp:
            logits = self.mlp(x)
        else:
            logits = self.fc(x)
        return logits


# ============================================================
# ⚙️ 2. Classification Loss
# ============================================================

def classification_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights=None) -> torch.Tensor:
    """
    Computes the weighted cross-entropy loss for multi-class classification.

    Args:
        logits: (B, num_classes)
        labels: (B,) true class indices
        class_weights: optional tensor of shape (num_classes,) for imbalance correction
    Returns:
        loss (scalar tensor)
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion(logits, labels)


# ============================================================
# 📊 3. Prediction & Evaluation Utilities
# ============================================================

def predict_class(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns predicted class indices.
    """
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    return preds


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes classification accuracy.
    """
    preds = predict_class(logits)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def predict_proba(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts raw logits to class probabilities using softmax.
    """
    return F.softmax(logits, dim=1)


# ============================================================
# 🚀 4. Example Usage (Standalone Test)
# ============================================================

if __name__ == "__main__":
    import numpy as np

    # Dummy inputs
    batch_size = 8
    input_dim = 256
    num_classes = 4

    # Random input embeddings (from Transformer)
    embeddings = torch.randn(batch_size, input_dim)

    # Random labels
    labels = torch.tensor(np.random.randint(0, num_classes, size=batch_size))

    # Initialize model
    model = ClassificationHead(input_dim=input_dim, num_classes=num_classes, use_mlp=True)
    logits = model(embeddings)

    # Compute loss
    loss = classification_loss(logits, labels)
    print(f"Cross-Entropy Loss: {loss.item():.6f}")

    # Predictions
    preds = predict_class(logits)
    acc = compute_accuracy(logits, labels)
    probs = predict_proba(logits)

    print(f"Predictions: {preds.tolist()}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Probabilities (first sample): {probs[0].detach().numpy()}")
