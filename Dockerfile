# Dockerfile para deploy gratuito
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY simple_fastapi.py .
COPY GRADIO/ ./GRADIO/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "simple_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]