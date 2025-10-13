from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer

from .config import get_default_config, Config
from .utils import set_seed, ensure_dir


def generate_synthetic_data(cfg: Config, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.random_seed)
    ensure_dir(os.path.join(out_dir, "samples"))
    ensure_dir(os.path.join(out_dir, "images"))

    # Users
    user_ids = np.arange(cfg.num_users)
    user_segments = np.random.choice(["A", "B", "C"], size=cfg.num_users)
    user_age = np.random.randint(18, 70, size=cfg.num_users)
    user_df = pd.DataFrame({
        "user_id": user_ids,
        "segment": user_segments,
        "age": user_age,
    })

    # Items
    item_ids = np.arange(cfg.num_items)
    titles = [f"Item {i}" for i in item_ids]
    descriptions = [f"This is a great product number {i} with features." for i in item_ids]

    # Generate placeholder images
    image_paths = []
    for i in tqdm(item_ids, desc="Generating images"):
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img_path = os.path.join(out_dir, "images", f"item_{i}.jpg")
        img.save(img_path)
        image_paths.append(img_path)

    item_df = pd.DataFrame({
        "item_id": item_ids,
        "title": titles,
        "description": descriptions,
        "image_path": image_paths,
    })

    # Interactions
    interactions = []
    for _ in tqdm(range(cfg.num_interactions), desc="Generating interactions"):
        u = int(np.random.choice(user_ids))
        i = int(np.random.choice(item_ids))
        clicks = np.random.poisson(0.3)
        rating = np.clip(np.random.normal(3.5, 1.0), 1, 5)
        purchased = int(np.random.rand() < 0.1)
        interactions.append((u, i, clicks, rating, purchased))
    inter_df = pd.DataFrame(interactions, columns=["user_id", "item_id", "clicks", "rating", "purchased"])

    # Save CSVs
    user_df.to_csv(os.path.join(out_dir, "samples", "users.csv"), index=False)
    item_df.to_csv(os.path.join(out_dir, "samples", "items.csv"), index=False)
    inter_df.to_csv(os.path.join(out_dir, "samples", "interactions.csv"), index=False)

    return user_df, item_df, inter_df


def build_text_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_image_encoder(backbone: str) -> Tuple[torch.nn.Module, transforms.Compose, int]:
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = 1280
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # remove final classification layer
    if hasattr(net, "fc"):
        net.fc = torch.nn.Identity()
    elif hasattr(net, "classifier"):
        if isinstance(net.classifier, torch.nn.Sequential):
            net.classifier[-1] = torch.nn.Identity()
        else:
            net.classifier = torch.nn.Identity()

    net.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return net, preprocess, feat_dim


def compute_embeddings(cfg: Config, data_dir: str) -> Dict[str, np.ndarray]:
    users = pd.read_csv(os.path.join(data_dir, "samples", "users.csv"))
    items = pd.read_csv(os.path.join(data_dir, "samples", "items.csv"))

    # Text embeddings
    text_model = build_text_encoder(cfg.text_backbone)
    item_texts = (items["title"].astype(str) + ". " + items["description"].astype(str)).tolist()
    text_emb = text_model.encode(item_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)

    # Image embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model, img_preprocess, img_dim = build_image_encoder(cfg.image_backbone)
    img_model.to(device)
    all_img = []
    with torch.inference_mode():
        for p in tqdm(items["image_path"].tolist(), desc="Image embeddings"):
            img = Image.open(p).convert("RGB")
            t = img_preprocess(img).unsqueeze(0).to(device)
            feat = img_model(t).detach().cpu().numpy()[0]
            feat = feat / (np.linalg.norm(feat) + 1e-12)
            all_img.append(feat)
    img_emb = np.stack(all_img, axis=0)

    # User features (simple one-hot of segment + age bucket)
    segment_onehot = pd.get_dummies(users["segment"], prefix="seg").values
    age_bucket = np.clip(((users["age"].values - 18) // 10), 0, 5)
    age_onehot = np.eye(6)[age_bucket]
    user_feat = np.concatenate([segment_onehot, age_onehot], axis=1).astype(np.float32)

    # Save numpy arrays
    out = os.path.join(data_dir, "artifacts")
    ensure_dir(out)
    np.save(os.path.join(out, "item_text_embeddings.npy"), text_emb)
    np.save(os.path.join(out, "item_image_embeddings.npy"), img_emb)
    np.save(os.path.join(out, "user_features.npy"), user_feat)

    return {
        "item_text_embeddings": text_emb,
        "item_image_embeddings": img_emb,
        "user_features": user_feat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_synthetic", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    cfg = get_default_config()
    ensure_dir(args.output_dir)

    if args.generate_synthetic:
        generate_synthetic_data(cfg, args.output_dir)

    compute_embeddings(cfg, args.output_dir)


if __name__ == "__main__":
    main()
