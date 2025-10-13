from __future__ import annotations

from typing import Optional

import numpy as np
import shap
import torch
import torch.nn as nn


def shap_explain_projection(
    model: nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    max_samples: int = 100,
) -> shap._explanation.Explanation:
    def predict(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            t = torch.tensor(x, dtype=torch.float32)
            y = model(t)
            return y.detach().cpu().numpy()

    masker = shap.maskers.Independent(background[: max_samples])
    explainer = shap.Explainer(predict, masker)
    return explainer(samples[: max_samples])
