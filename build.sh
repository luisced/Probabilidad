#!/bin/bash

# Define the image name and container name
IMAGE_NAME="my-streamlit-app"
CONTAINER_NAME="my-streamlit-container"

# Build the Docker image
echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully."
else
    echo "Docker image build failed."
    exit 1
fi

# Run the Docker container with a specified name
echo "Running the Docker container..."
docker run --name $CONTAINER_NAME -p 8501:8501 $IMAGE_NAME

# Check if the container started successfully
if [ $? -eq 0 ]; then
    echo "Docker container is running. Access the app at http://localhost:8501"
else
    echo "Failed to start the Docker container."
    exit 1
fi
