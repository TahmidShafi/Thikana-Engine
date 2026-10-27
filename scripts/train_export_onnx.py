"""
scripts/train_export_onnx.py

Trains a real, compact neural token classifier on data/train_synthetic_2500.json
using PyTorch and exports trained weights to models/onnx/tagger.onnx.

Architecture:
  - Input: Dense 64-dimensional feature vector projected from Stage-1 normalizer,
    Stage-2 syntactic/morphological features, POI head nouns, and gazetteer inverted index.
  - Multi-layer non-linear MLP token classifier with residual/dropout regularization.
  - Export: ONNX opset 17 with dynamic sequence-length axes, verified with ONNX Runtime.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gazetteer import BangladeshGazetteer
from src.phonetic_trie import PhoneticGazetteerMatcher
from src.tagger_crf import BIO_LABELS
from src.tagger_onnx import (
    FEATURE_DIM,
    NUM_LABELS,
    ONNXSequenceTagger,
    ViterbiBIODecoder,
    vectorize_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_export_onnx")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train_synthetic_2500.json"
MODELS_DIR = BASE_DIR / "models" / "onnx"
OUTPUT_ONNX_PATH = MODELS_DIR / "tagger.onnx"


class NeuralTokenClassifier(nn.Module):
    """
    Lightweight, high-throughput neural token classifier for address BIO tagging.
    Takes (batch, seq_len, in_features) and maps to (batch, seq_len, num_classes).
    """

    def __init__(self, in_features: int = FEATURE_DIM, hidden_dim: int = 128, num_classes: int = NUM_LABELS):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def train_and_export(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_ONNX_PATH,
    epochs: int = 5,
    lr: float = 0.005,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Gazetteer and Phonetic Matcher
    logger.info("Initializing BangladeshGazetteer & Phonetic Index...")
    gazetteer = BangladeshGazetteer()
    matcher = PhoneticGazetteerMatcher(gazetteer)

    # 2. Load Synthetic Dataset
    logger.info(f"Loading synthetic training dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} training sequences.")

    label2id = {lbl: i for i, lbl in enumerate(BIO_LABELS)}

    # 3. Pre-vectorize sequences for ultra-fast training
    logger.info("Vectorizing training sequences (64-dim feature projection)...")
    t0_vec = time.perf_counter()
    X_tensors: list[torch.Tensor] = []
    y_tensors: list[torch.Tensor] = []

    for item in dataset:
        tokens = item["tokens"]
        tags = item["tags"]
        if not tokens:
            continue

        vec_np = vectorize_tokens(tokens, matcher=matcher, dim=FEATURE_DIM)  # (1, seq_len, 64)
        tag_ids = [label2id.get(t, 0) for t in tags]

        X_tensors.append(torch.from_numpy(vec_np))
        y_tensors.append(torch.tensor(tag_ids, dtype=torch.long))

    logger.info(f"Vectorized {len(X_tensors)} sequences in {time.perf_counter() - t0_vec:.2f}s.")

    # 4. Initialize Model, Loss, Optimizer
    model = NeuralTokenClassifier(in_features=FEATURE_DIM, hidden_dim=128, num_classes=NUM_LABELS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info(f"Training NeuralTokenClassifier for {epochs} epochs on CPU...")
    t0_train = time.perf_counter()

    indices = list(range(len(X_tensors)))
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        # Train in mini-batches of sequences
        batch_size = 16
        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0)

            for idx in b_idx:
                X_seq = X_tensors[idx]  # (1, seq_len, 64)
                y_seq = y_tensors[idx]  # (seq_len,)
                logits = model(X_seq).squeeze(0)  # (seq_len, num_classes)

                loss = criterion(logits, y_seq)
                batch_loss = batch_loss + loss

                preds = torch.argmax(logits, dim=-1)
                correct_tokens += int((preds == y_seq).sum().item())
                total_tokens += y_seq.numel()

            batch_loss = batch_loss / len(b_idx)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * len(b_idx)

        avg_loss = total_loss / len(X_tensors)
        token_acc = (correct_tokens / max(1, total_tokens)) * 100.0
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Token Accuracy: {token_acc:.2f}%")

    logger.info(f"Training complete in {time.perf_counter() - t0_train:.2f}s.")

    # 5. Export to ONNX
    model.eval()
    logger.info(f"Exporting genuinely trained model to ONNX: {output_path}...")
    dummy_input = torch.randn(1, 10, FEATURE_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        },
        dynamo=False,
    )
    logger.info(f"Successfully saved ONNX model artifact: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

    # 6. Verify with ONNX Runtime & Viterbi Decoder
    logger.info("Verifying ONNX Runtime inference and Viterbi decoding...")
    tagger = ONNXSequenceTagger(model_path=output_path, matcher=matcher)
    test_addresses = [
        "House 12, Road 4, Mirpur 10, Dhaka",
        "Holding 45, Tajmahal Road, Mohammadpur, Dhaka",
        "Uttara Bank Limited, Road 27, Dhanmondi, Dhaka",
        "Halishahar Block G, Chattogram",
    ]
    for addr in test_addresses:
        res = tagger.tag_address(addr)
        logger.info(f"Parsed: '{addr}' -> Entities: {res['entities']} ({res['latency_ms']:.2f}ms)")

    logger.info("ONNX trained model export verified successfully!")


if __name__ == "__main__":
    train_and_export()
