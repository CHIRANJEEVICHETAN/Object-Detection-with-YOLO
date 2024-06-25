# Use the official Python image with Python 3.12 (replace with the correct tag if available)
FROM python:3.12

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

# Create and activate a non-root user
RUN adduser --disabled-password --gecos '' --uid 10001 appuser

# Give ownership of the /app directory to the appuser
RUN chown -R appuser:appuser /app

# Set the user ID to 10001
USER 10001

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install gunicorn

# Copy the Flask app
COPY . .

# Expose port 80
EXPOSE 80

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
