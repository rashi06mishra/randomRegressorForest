# Minimal Dockerfile for Dynamic Pricing API
# Assumes `dynamic_pricing_model.pkl` and `vehicle_encoder.pkl` are present in the repo
# Small image: does not install system build deps. Use this when pickles are committed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

# Ensure entrypoint is executable (entrypoint starts uvicorn)
RUN chmod +x /app/entrypoint.sh || true

EXPOSE ${PORT}

CMD ["/app/entrypoint.sh"]
