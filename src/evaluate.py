from __future__ import annotations

import argparse
import os
from typing import List, Set

import numpy as np
import pandas as pd

from .utils import cosine_similarity, precision_recall_at_k, ndcg_at_k, ensure_dir, save_plot_topk


def load_data(data_dir: str, artifacts_dir: str):
    users = pd.read_csv(os.path.join(data_dir, "samples", "users.csv"))
    items = pd.read_csv(os.path.join(data_dir, "samples", "items.csv"))
    inter = pd.read_csv(os.path.join(data_dir, "samples", "interactions.csv"))
    user_emb = np.load(os.path.join(artifacts_dir, "user_embeddings.npy"))
    item_emb = np.load(os.path.join(artifacts_dir, "item_embeddings.npy"))
    return users, items, inter, user_emb, item_emb


def build_ground_truth(inter: pd.DataFrame, num_items: int) -> List[Set[int]]:
    num_users = inter["user_id"].max() + 1
    gt: List[Set[int]] = [set() for _ in range(num_users)]
    for _, row in inter.iterrows():
        if int(row["purchased"]) == 1 or row["rating"] >= 4.0 or row["clicks"] >= 1:
            gt[int(row["user_id"])].add(int(row["item_id"]))
    return gt


def evaluate(data_dir: str, artifacts_dir: str, topk: int) -> None:
    users, items, inter, user_emb, item_emb = load_data(data_dir, artifacts_dir)

    scores = cosine_similarity(user_emb, item_emb)

    gt = build_ground_truth(inter, num_items=item_emb.shape[0])

    p, r = precision_recall_at_k(scores, gt, k=topk)
    n = ndcg_at_k(scores, gt, k=topk)

    print({"precision@k": round(p, 4), "recall@k": round(r, 4), "ndcg@k": round(n, 4)})

    # Visualize a couple of users
    plots_dir = os.path.join(artifacts_dir, "plots")
    ensure_dir(plots_dir)

    for u in [0, min(1, scores.shape[0]-1)]:
        top_idx = np.argsort(-scores[u])[:topk]
        top_scores = scores[u][top_idx].tolist()
        save_plot_topk(u, top_idx.tolist(), top_scores, os.path.join(plots_dir, f"user_{u}_top{topk}.png"))

    # Example query: print top-5 items for user 0
    u = 0
    top_idx = np.argsort(-scores[u])[:5]
    print("Example recommendations for user 0:")
    print(items.iloc[top_idx][["item_id", "title"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="data/artifacts")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    evaluate(args.data_dir, args.artifacts_dir, args.topk)


if __name__ == "__main__":
    main()
