FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements first (leverages Docker caching)
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y awscli && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Run the application
CMD ["python3", "app.py"]
