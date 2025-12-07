# TikTok RAG App - Railway Deployment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create data directory for SQLite database
RUN mkdir -p /app/data

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for data directory
ENV DATA_DIR=/app/data

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8080"]
