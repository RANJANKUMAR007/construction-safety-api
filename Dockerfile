# Base image: slim Python 3.11 to match the local dev environment (python 3.11.9)
FROM python:3.11-slim

WORKDIR /app

# opencv (used internally by ultralytics) needs these system libraries even in
# "headless" setups - without them, the container crashes on import with a
# confusing libGL error that has nothing to do with your actual code.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installed with the CPU-only PyTorch index explicitly. Without this, pip can
# pull a CUDA-enabled torch build (multiple GB, and useless anyway since this
# container has no GPU) instead of the much smaller CPU-only wheel.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Only copy the app code and model weights - not venv, notebooks, or test scripts.
# See .dockerignore for the full exclusion list.
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
