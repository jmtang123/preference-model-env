#!/usr/bin/env python3
"""
RL Environment Judge skeleton: load submitted model.py, evaluate on Judge's own MNIST
test set, output score. Runs inside a VM and does not use any self-reported metrics.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Config (aligned with DESIGN.md)
# -----------------------------------------------------------------------------
WORKSPACE_DIR = Path("/workspace")
if not WORKSPACE_DIR.exists():
    # Local or CI: use workspace under this directory
    WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"
MODEL_PY = WORKSPACE_DIR / "model.py"
PASS_THRESHOLD = 0.92
BATCH_SIZE = 256
MNIST_ROOT = Path("/data/mnist")
if not MNIST_ROOT.exists():
    MNIST_ROOT = Path(__file__).resolve().parent / "data" / "mnist"
MNIST_ROOT.mkdir(parents=True, exist_ok=True)


def load_judge_test_loader():
    """Load Judge's own MNIST test set; independent of the submission."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root=str(MNIST_ROOT),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


def load_submission_module():
    """Load module from /workspace/model.py and return the load_model callable."""
    if not MODEL_PY.exists():
        raise FileNotFoundError(f"Submission not found: {MODEL_PY}")
    spec = importlib.util.spec_from_file_location("model", MODEL_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODEL_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "load_model"):
        raise AttributeError("model.py must define load_model()")
    return module.load_model


def validate_module(module: nn.Module) -> None:
    """Verify the returned object is nn.Module and I/O shapes are correct."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"load_model() must return torch.nn.Module, got {type(module)}")
    x = torch.rand(1, 1, 28, 28)
    module.eval()
    with torch.no_grad():
        out = module(x)
    if out.shape != (1, 10):
        raise ValueError(f"Expected output shape (1, 10), got {out.shape}")


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on the Judge's test set."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def run_judge() -> dict:
    """
    Run the full scoring pipeline; return dict with score and pass.
    Anti-cheating: the only source of score is inference inside this function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"score": 0.0, "pass": False, "error": None}

    try:
        load_model_fn = load_submission_module()
        model = load_model_fn()
        validate_module(model)
        loader = load_judge_test_loader()
        accuracy = evaluate_accuracy(model, loader, device)
        result["score"] = round(accuracy, 4)
        result["pass"] = accuracy >= PASS_THRESHOLD
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    result = run_judge()
    print(json.dumps(result, indent=2))
    return 0 if result.get("pass", False) else 1


if __name__ == "__main__":
    sys.exit(main())
