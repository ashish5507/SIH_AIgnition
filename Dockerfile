# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes your api.py, index.html, and the src/, models/, data/ folders
COPY . .

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Define the command to run your application's backend server on the correct port
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
