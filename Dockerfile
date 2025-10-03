# Use the standard, reliable Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# STEP 1: Install the large PyTorch library (CPU version only) from its special URL
# This is much faster and smaller than the default.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install the rest of the smaller, regular dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the correct port for the platform (7860 is common, 80 also works)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]