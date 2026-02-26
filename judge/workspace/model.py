"""
Example submission: compliant with the load_model() interface in DESIGN.md.
- If /workspace/model.pt exists, load and return that model.
- Otherwise return an untrained small CNN (~10% accuracy; for pipeline demo only).
"""
import torch
import torch.nn as nn
from pathlib import Path

WORKSPACE = Path("/workspace")
if not WORKSPACE.exists():
    WORKSPACE = Path(__file__).resolve().parent
CHECKPOINT_PATH = WORKSPACE / "model.pt"


class SmallCNN(nn.Module):
    """MNIST classifier: (N, 1, 28, 28) -> (N, 10)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model() -> nn.Module:
    """No arguments; returns torch.nn.Module with input (N,1,28,28), output (N,10)."""
    model = SmallCNN()
    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    return model
