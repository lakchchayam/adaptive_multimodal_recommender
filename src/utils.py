from __future__ import annotations

import os
import random
from typing import Iterable, List, Tuple, Dict

import numpy as np
import torch
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def precision_recall_at_k(
    scores: np.ndarray,
    ground_truth: List[set],
    k: int,
) -> Tuple[float, float]:
    # scores: [num_users, num_items]
    num_users = scores.shape[0]
    precisions = []
    recalls = []
    for u in range(num_users):
        topk = np.argsort(-scores[u])[:k]
        hits = len([i for i in topk if i in ground_truth[u]])
        precisions.append(hits / k)
        denom = max(len(ground_truth[u]), 1)
        recalls.append(hits / denom)
    return float(np.mean(precisions)), float(np.mean(recalls))


def ndcg_at_k(scores: np.ndarray, ground_truth: List[set], k: int) -> float:
    # binary relevance
    y_true = []
    y_score = []
    for u in range(scores.shape[0]):
        rel = np.zeros(scores.shape[1])
        rel[list(ground_truth[u])] = 1.0
        y_true.append(rel[None, :])
        y_score.append(scores[u][None, :])
    # sklearn ndcg_score expects shape (1, n_items) per user; average manually
    ndcgs = []
    for u in range(scores.shape[0]):
        ndcgs.append(ndcg_score(y_true[u], y_score[u], k=k))
    return float(np.mean(ndcgs))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot_topk(
    user_id: int,
    item_ids: List[int],
    scores: List[float],
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 4))
    sns.barplot(x=[str(i) for i in item_ids], y=scores)
    plt.title(f"Top-{len(item_ids)} Items for User {user_id}")
    plt.xlabel("Item ID")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
