FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the batch inference script
COPY init.sql .
COPY upload_predictions.py .
COPY batch_inference.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the batch inference script
CMD ["python", "batch_inference.py"] 