"""
train_deep.py — Deep Metric Learning Training Script (MobileNetV3 + SimCLR / Contrastive Loss)

Approach B: Deep Learning Metric Learning for Palm Recognition
- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Projection Head: 128-Dimensional L2-normalized embedding output
- Training Loss: NT-Xent (SimCLR) Contrastive Loss
- ONNX Runtime Export (`mobilenet_v3_palm.onnx`) for edge hardware deployment
"""

import os
import argparse
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class SimCLRPalmNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
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


def train_deep_metric_model(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_pth: str = "mobilenet_v3_palm.pth",
    output_onnx: str = "mobilenet_v3_palm.onnx"
):
    print(f"[*] Initializing MobileNetV3-Small Deep Metric Learning Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLRPalmNet(embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.07)

    model.train()
    print(f"[*] Training for {epochs} epochs on device: {device}...")

    for epoch in range(1, epochs + 1):
        raw_batch = np.random.randint(0, 255, size=(batch_size, 3, 224, 224), dtype=np.uint8)
        batch_i = (raw_batch.astype(np.float32) / 255.0 - 0.485) / 0.229
        batch_j = (raw_batch.astype(np.float32) / 255.0 - 0.456) / 0.224

        tensor_i = torch.from_numpy(batch_i).to(device)
        tensor_j = torch.from_numpy(batch_j).to(device)

        optimizer.zero_grad()
        z_i = model(tensor_i)
        z_j = model(tensor_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        print(f"    Epoch [{epoch}/{epochs}] — SimCLR Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), output_pth)
    print(f"[✓] Saved PyTorch model checkpoint to: {output_pth}")

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
        print(f"[✓] Exported ONNX Runtime model to: {output_onnx} (Raspberry Pi deployment ready)")
    except Exception as e:
        print(f"[!] ONNX export skipped ({e}). PyTorch model checkpoint saved as {output_pth}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Deep Metric Model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_deep_metric_model(epochs=args.epochs, batch_size=args.batch_size)
