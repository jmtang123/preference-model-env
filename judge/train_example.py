#!/usr/bin/env python3
"""
Example training script: train SmallCNN on MNIST and save to workspace/model.pt.
After running, execute run_judge.py to get pass (accuracy >= 0.92).
Used to verify the Judge pipeline runs end-to-end inside a VM.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(__file__).resolve().parent / "data" / "mnist"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


class SmallCNN(nn.Module):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            opt.step()
        print("Epoch {}/3 done".format(epoch + 1))
    torch.save(model.state_dict(), WORKSPACE / "model.pt")
    print("Saved to {}. Run judge: python run_judge.py".format(WORKSPACE / "model.pt"))


if __name__ == "__main__":
    main()
