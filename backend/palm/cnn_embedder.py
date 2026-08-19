"""
Palm embedding extractor (Track B - CNN PyTorch/ONNX MobileNetV3).
Optional deep learning embedder for higher accuracy benchmark comparisons.
"""

import os
from typing import Optional

import cv2
import numpy as np


class PalmEmbedderCNN:
    def __init__(self, model_path: str = "mobilenet_v3_palm.pth", embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.use_onnx = model_path.endswith(".onnx")
        self.session = None
        self.torch_model = None

        if os.path.exists(model_path):
            if self.use_onnx:
                try:
                    import onnxruntime as ort
                    self.session = ort.InferenceSession(model_path)
                    print(f"[*] Loaded ONNX Palm Embedder from {model_path}")
                except Exception as e:
                    print(f"[!] Failed to load ONNX model {model_path}: {e}")
            else:
                try:
                    import torch
                    import torchvision.models as models

                    class SimCLRPalmNet(torch.nn.Module):
                        def __init__(self, embedding_dim: int = 128):
                            super().__init__()
                            try:
                                backbone = models.mobilenet_v3_small(weights=None)
                            except Exception:
                                backbone = models.mobilenet_v3_small(pretrained=False)
                            in_features = backbone.classifier[0].in_features
                            backbone.classifier = torch.nn.Identity()
                            self.backbone = backbone
                            self.projector = torch.nn.Sequential(
                                torch.nn.Linear(in_features, 256),
                                torch.nn.BatchNorm1d(256),
                                torch.nn.Hardswish(),
                                torch.nn.Linear(256, embedding_dim)
                            )

                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            features = self.backbone(x)
                            embeddings = self.projector(features)
                            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    net = SimCLRPalmNet(embedding_dim=self.embedding_dim)
                    checkpoint = torch.load(model_path, map_location="cpu")
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        net.load_state_dict(checkpoint["state_dict"], strict=False)
                    elif isinstance(checkpoint, dict):
                        net.load_state_dict(checkpoint, strict=False)
                    net.eval()
                    self.torch_model = net
                    print(f"[*] Loaded PyTorch CNN SimCLR Palm Embedder from {model_path}")
                except Exception as e:
                    print(f"[!] Failed to load PyTorch model {model_path}: {e}")

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Extracts CNN feature embedding from 224x224 BGR image."""
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = (resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor_inp = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {input_name: tensor_inp})[0][0]
        elif self.torch_model is not None:
            import torch
            with torch.no_grad():
                inp = torch.from_numpy(tensor_inp)
                out = self.torch_model(inp)[0].numpy()
        else:
            # Fallback mock embedding
            rng = np.random.default_rng(hash(aligned_bgr.tobytes()) % 2**32)
            out = rng.normal(0, 1.0, self.embedding_dim).astype(np.float32)

        norm = np.linalg.norm(out)
        return (out / norm).astype(np.float32) if norm > 0 else out.astype(np.float32)
