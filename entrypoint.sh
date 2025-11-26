#!/usr/bin/env bash
set -euo pipefail

# Default PORT already set in Dockerfile, but allow override
PORT=${PORT:-7860}

# If model artifacts are missing, run training script (this uses vehicles_dataset_5000.csv)
if [ ! -f /app/dynamic_pricing_model.pkl ] || [ ! -f /app/vehicle_encoder.pkl ]; then
  echo "Model artifacts not found — training model. This may take a few minutes..."
  python /app/model/model.py
else
  echo "Model artifacts found, skipping training."
fi

# Start Uvicorn
exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
