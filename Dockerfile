# Use the official Python image with Python 3.12 (replace with the correct tag if available)
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopencv-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    # Handle specific versions or constraints
    && pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the Flask app
COPY . .

# Expose port 80
EXPOSE 80

# Command to run the application
CMD ["python", "app.py"]
