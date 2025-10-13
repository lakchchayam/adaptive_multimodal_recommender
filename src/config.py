from dataclasses import dataclass
from typing import Literal


TextBackbone = Literal["all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-uncased"]
ImageBackbone = Literal["resnet18", "resnet50", "efficientnet_b0"]
FusionType = Literal["concat_mlp", "gated"]


@dataclass
class Config:
    random_seed: int = 42
    num_users: int = 200
    num_items: int = 500
    num_interactions: int = 4000

    data_dir: str = "data"
    artifacts_dir: str = "data/artifacts"

    text_backbone: TextBackbone = "all-MiniLM-L6-v2"
    image_backbone: ImageBackbone = "resnet18"
    fusion: FusionType = "concat_mlp"

    text_embedding_dim: int = 384  # for MiniLM
    image_embedding_dim: int = 512  # resnet18 penultimate layer
    user_feature_dim: int = 16

    fusion_hidden_dim: int = 256
    output_embedding_dim: int = 256

    train_batch_size: int = 64
    eval_batch_size: int = 128
    num_epochs: int = 2
    learning_rate: float = 1e-3


def get_default_config() -> Config:
    return Config()
