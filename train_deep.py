"""
train_deep.py — Deep Metric Learning Fine-Tuning Script (MobileNetV3 + SimCLR / Contrastive Loss)

Approach B: Deep Learning Metric Learning for Palm Recognition
- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Projection Head: 128-Dimensional L2-normalized embedding output with BatchNorm
- Training Loss: NT-Xent (SimCLR) Contrastive Loss with Cosine Annealing LR Scheduler
- ONNX Runtime Export (`mobilenet_v3_palm.onnx`) for edge hardware deployment
"""

import os
import argparse
from typing import Tuple, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models


class SimCLRPalmNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        try:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            backbone = models.mobilenet_v3_small(pretrained=True)

        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projector(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR Contrastive Loss)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)

        pos_i = torch.diag(sim_matrix, batch_size)
        pos_j = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([pos_i, pos_j], dim=0)

        log_prob = positives - torch.logsumexp(sim_matrix, dim=1)
        return -log_prob.mean()


def generate_realistic_palm_pair(rng, size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two augmented views of a single palm ROI for SimCLR training."""
    skin_tone = rng.integers(140, 210, size=(3,), dtype=np.uint8)
    base = np.full((size, size, 3), skin_tone, dtype=np.uint8)

    num_lines = rng.integers(3, 7)
    for _ in range(num_lines):
        pt1 = tuple(rng.integers(20, size - 20, size=2))
        pt2 = tuple(rng.integers(20, size - 20, size=2))
        color = (int(skin_tone[0] * 0.4), int(skin_tone[1] * 0.4), int(skin_tone[2] * 0.4))
        thickness = int(rng.integers(2, 5))
        cv2.line(base, pt1, pt2, color, thickness)

    def augment(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        angle = float(rng.uniform(-15, 15))
        scale = float(rng.uniform(0.9, 1.1))
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        if rng.random() > 0.5:
            aug = cv2.GaussianBlur(aug, (5, 5), 0)

        noise = rng.normal(0, 4, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return aug

    view_i = augment(base)
    view_j = augment(base)
    return view_i, view_j


def preprocess_batch(imgs: np.ndarray) -> torch.Tensor:
    """Normalizes BGR uint8 array to ImageNet PyTorch tensor (B, 3, H, W)."""
    rgb = imgs[..., ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    norm = (rgb - mean) / std
    t = torch.from_numpy(norm.transpose(0, 3, 1, 2))
    return t


def train_deep_metric_model(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_pth: str = "mobilenet_v3_palm.pth",
    output_onnx: str = "mobilenet_v3_palm.onnx"
):
    print(f"[*] Initializing MobileNetV3-Small Deep Metric Learning Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLRPalmNet(embedding_dim=128).to(device)

    if os.path.exists(output_pth):
        try:
            model.load_state_dict(torch.load(output_pth, map_location=device), strict=False)
            print(f"[*] Loaded existing weights from {output_pth} for fine-tuning!")
        except Exception as e:
            print(f"[!] Info: Initializing fresh weights ({e})")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = NTXentLoss(temperature=0.07)

    rng = np.random.default_rng(42)
    print(f"[*] Fine-tuning MobileNetV3 for {epochs} epochs on device: {device}...")

    model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_steps = 10

        for step in range(num_steps):
            batch_i_list = []
            batch_j_list = []

            for _ in range(batch_size):
                img_i, img_j = generate_realistic_palm_pair(rng, size=224)
                batch_i_list.append(img_i)
                batch_j_list.append(img_j)

            tensor_i = preprocess_batch(np.array(batch_i_list)).to(device)
            tensor_j = preprocess_batch(np.array(batch_j_list)).to(device)

            optimizer.zero_grad()
            z_i = model(tensor_i)
            z_j = model(tensor_j)

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / num_steps
        current_lr = scheduler.get_last_lr()[0]
        print(f"    Epoch [{epoch:02d}/{epochs:02d}] — SimCLR Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_pth)

    print(f"[✓] Saved fine-tuned PyTorch model checkpoint to: {output_pth} (Best Loss: {best_loss:.4f})")

    # Export ONNX Runtime model
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )
        print(f"[✓] Exported fine-tuned ONNX Runtime model to: {output_onnx} (Raspberry Pi deployment ready)")
    except Exception as e:
        print(f"[!] ONNX export note ({e}). PyTorch model checkpoint saved as {output_pth}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 Deep Metric Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_deep_metric_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

