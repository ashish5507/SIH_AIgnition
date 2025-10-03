# Use the standard, reliable Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install PyTorch (CPU version only) separately for speed and reliability
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the smaller dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the correct port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]