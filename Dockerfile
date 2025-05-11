# Use slim official Python 3.13 base image
FROM python:3.13.0-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default launch
CMD ["python", "app.py"]
