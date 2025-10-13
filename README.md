# Adaptive Multi-Modal Recommendation Engine

A production-ready, modular Python project for building recommendations using text, images, and user interaction data. It supports synthetic data generation, modern Transformer and CNN encoders, multimodal fusion, offline evaluation (Precision@k, Recall@k, NDCG), visualizations, and optional explainability.

## Features
- Multimodal inputs: text, image, and user interaction features
- Pluggable encoders: BERT/DistilBERT via SentenceTransformers; ResNet/EfficientNet via torchvision
- Fusion strategies: concatenation + MLP, gated fusion
- Similarity search: cosine similarity; hybrid with collaborative filtering-signals
- Offline metrics: Precision@k, Recall@k, NDCG
- Visualizations: recommendation charts, top-k per user
- Synthetic dataset generation for quick demos
- Optional explainability (attention scores/SHAP for the fusion MLP)

## Project Structure
```
.
├── data/
│   ├── images/                 # synthetic images generated at runtime (cached)
│   └── samples/                # synthetic CSVs generated at runtime (cached)
├── notebooks/
│   └── demo.ipynb              # end-to-end demo
├── scripts/
│   └── init_github.sh          # initialize and push to GitHub
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluate.py
├── .gitignore
├── Dockerfile                  # optional
├── requirements.txt
└── README.md
```

## Quickstart

### 1) Environment
- Python 3.10+
- Recommended: create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Generate synthetic data and run training/evaluation
```bash
python -m src.preprocessing --generate_synthetic --output_dir data
python -m src.model --train --data_dir data --artifacts_dir data/artifacts
python -m src.evaluate --data_dir data --artifacts_dir data/artifacts --topk 10
```

### 3) Explore in notebook
```bash
jupyter lab
# open notebooks/demo.ipynb
```

### 4) Initialize and push to GitHub (optional)
```bash
bash scripts/init_github.sh <your_github_repo_url>
```

## Configuration
Key hyperparameters and model choices live in `src/config.py`. You can swap encoders (e.g., SentenceTransformer model name, torchvision backbone) and change fusion/hidden sizes.

## Data
- Synthetic data: generated via `src.preprocessing` if no real data is provided
- Format:
  - `samples/users.csv`: user_id, demographics/segments, aggregate stats
  - `samples/items.csv`: item_id, title, description, image_path
  - `samples/interactions.csv`: user_id, item_id, clicks, rating, purchased
  - `images/`: image files referenced by `items.csv`

## Evaluation
- Precision@k, Recall@k, and NDCG computed per-user and averaged
- Visualizations saved to `data/artifacts/plots`

## Explainability (optional)
- Attention weights in the fusion layer
- SHAP for the final MLP (if enabled)

## Docker (optional)
Build and run:
```bash
docker build -t multimodal-recsys:latest .
docker run --rm -it -v "$PWD":/app multimodal-recsys:latest bash
```

## License
MIT
