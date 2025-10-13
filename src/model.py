from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import get_default_config, Config
from .utils import ensure_dir, set_seed


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, dims: Tuple[int, int, int], output_dim: int):
        super().__init__()
        t_dim, i_dim, u_dim = dims
        self.gate = nn.Sequential(
            nn.Linear(t_dim + i_dim + u_dim, u_dim),
            nn.Sigmoid(),
        )
        self.ff = nn.Linear(t_dim + i_dim + u_dim, output_dim)

    def forward(self, t: torch.Tensor, i: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, i, u], dim=1)
        g = self.gate(x)
        x = torch.cat([t, i, u * g], dim=1)
        return self.ff(x)


class MultimodalProjector(nn.Module):
    def __init__(self, cfg: Config, text_dim: int, image_dim: int, user_dim: int):
        super().__init__()
        self.cfg = cfg
        if cfg.fusion == "concat_mlp":
            self.fusion = FusionMLP(text_dim + image_dim + user_dim, cfg.fusion_hidden_dim, cfg.output_embedding_dim)
        else:
            self.fusion = GatedFusion((text_dim, image_dim, user_dim), cfg.output_embedding_dim)
        self.item_projector = nn.Linear(text_dim + image_dim, cfg.output_embedding_dim)
        self.user_projector = nn.Linear(user_dim, cfg.output_embedding_dim)

    def forward_item(self, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, i], dim=1)
        return nn.functional.normalize(self.item_projector(x), dim=1)

    def forward_user(self, u: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.user_projector(u), dim=1)

    def forward(self, t: torch.Tensor, i: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.fusion(torch.cat([t, i, u], dim=1)), dim=1)


def load_artifacts(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    art = os.path.join(data_dir, "artifacts")
    t = np.load(os.path.join(art, "item_text_embeddings.npy"))
    v = np.load(os.path.join(art, "item_image_embeddings.npy"))
    u = np.load(os.path.join(art, "user_features.npy"))

    inter = np.loadtxt(os.path.join(data_dir, "samples", "interactions.csv"), delimiter=",", skiprows=1, dtype=float)
    # columns: user_id, item_id, clicks, rating, purchased
    return t, v, u, inter


def train_projection(cfg: Config, data_dir: str, artifacts_dir: str) -> None:
    set_seed(cfg.random_seed)
    ensure_dir(artifacts_dir)

    t, v, u, inter = load_artifacts(data_dir)

    text_dim = t.shape[1]
    image_dim = v.shape[1]
    user_dim = u.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalProjector(cfg, text_dim, image_dim, user_dim).to(device)

    # Prepare training triples (user, positive item) with pairwise ranking against random negatives
    inter = inter.astype(int)
    user_ids = inter[:, 0]
    item_ids = inter[:, 1]

    user_feat = torch.tensor(u, dtype=torch.float32, device=device)
    text_emb = torch.tensor(t, dtype=torch.float32, device=device)
    img_emb = torch.tensor(v, dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    bce = nn.BCEWithLogitsLoss()

    num_samples = len(user_ids)
    for epoch in range(cfg.num_epochs):
        perm = np.random.permutation(num_samples)
        total_loss = 0.0
        for start in range(0, num_samples, cfg.train_batch_size):
            idx = perm[start : start + cfg.train_batch_size]
            u_batch = torch.tensor(user_ids[idx], dtype=torch.long, device=device)
            i_pos = torch.tensor(item_ids[idx], dtype=torch.long, device=device)
            i_neg = torch.randint(0, t.shape[0], (len(idx),), device=device)

            u_vec = model.forward_user(user_feat[u_batch])
            i_pos_vec = model.forward_item(text_emb[i_pos], img_emb[i_pos])
            i_neg_vec = model.forward_item(text_emb[i_neg], img_emb[i_neg])

            pos_scores = torch.sum(u_vec * i_pos_vec, dim=1)
            neg_scores = torch.sum(u_vec * i_neg_vec, dim=1)

            logits = pos_scores - neg_scores
            labels = torch.ones_like(logits)
            loss = bce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - loss: {avg_loss:.4f}")

    # Save projected item embeddings and user embeddings
    with torch.no_grad():
        item_proj = model.forward_item(text_emb, img_emb).cpu().numpy()
        user_proj = model.forward_user(user_feat).cpu().numpy()

    np.save(os.path.join(artifacts_dir, "item_embeddings.npy"), item_proj)
    np.save(os.path.join(artifacts_dir, "user_embeddings.npy"), user_proj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="data/artifacts")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.train:
        train_projection(cfg, args.data_dir, args.artifacts_dir)


if __name__ == "__main__":
    main()
