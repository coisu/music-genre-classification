# Use Python 3.8-slim base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Copy the necessary files
COPY requirements.txt requirements.txt
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script to train the model if needed
RUN python train_if_needed.py

# Run the Flask app
CMD ["python", "app.py"]



# FROM python:3.8-slim

# # Set working directory
# WORKDIR /app

# # Install system packages
# RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# # Copy necessary files
# COPY requirements.txt requirements.txt
# COPY app.py app.py
# COPY templates/ templates/
# COPY fma_metadata/ fma_metadata/
# COPY fma_small/ fma_small/

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Create uploads directory
# RUN mkdir uploads

# ENV TF_CPP_MIN_LOG_LEVEL=2
# ENV TF_ENABLE_ONEDNN_OPTS=0

# # Run Flask app
# CMD ["python", "app.py"]
