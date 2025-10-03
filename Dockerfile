# Use an official, slim, CPU-based PyTorch image that includes Python 3.10
FROM pytorch/pytorch:2.3.0-cpu-py3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the remaining dependencies
# This step will be much faster now
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell the server that the app runs on port 7860
EXPOSE 7860

# The command to start your FastAPI app using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]