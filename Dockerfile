# Use a fuller Python base image (has more system deps preinstalled)
FROM python:3.9-bullseye

# Set working directory
WORKDIR /app

# Copy only requirements first (to leverage Docker caching)
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Railway injects PORT as an env variable, so use it
EXPOSE 8000

# Run FastAPI with uvicorn, bind to $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
