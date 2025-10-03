# Use a fuller Python base image (fewer slow compiles)
FROM python:3.9-bullseye

WORKDIR /app

# Copy only requirements first
COPY requirements.txt ./

# Upgrade pip and install deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose Railway port (uses env var PORT)
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
