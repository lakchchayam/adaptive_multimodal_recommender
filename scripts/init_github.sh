#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <github_repo_url>"
  exit 1
fi

REPO_URL="$1"

# Initialize git if not already
if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "Initial commit: Adaptive Multi-Modal Recommendation Engine"

# Set default branch to main
if git show-ref --verify --quiet refs/heads/main; then
  git branch -M main
else
  git checkout -b main || git branch -M main
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

git push -u origin main
