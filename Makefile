# Variables
IMAGE_NAME = music-genre-classifier
CONTAINER_NAME = music-genre-container
PORT = 5000
CLASS_WEIGHTS_FILE = class_weights.txt
PREPROCESSED_DATA_FILE = preprocessed_data_fma_medium.npz
MODEL_DIR = music_genre_classifier_saved_model
FMA_MEDIUM = fma_medium
FMA_METADATA = fma_metadata

# Build, run, and display logs
all: build run logs

# Build the Docker image
build:
	@echo "Building the Docker image $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) .
	@echo "Docker image $(IMAGE_NAME) built successfully."

# Run the Docker container (detached mode)
run:
	@echo "Checking if container $(CONTAINER_NAME) already exists..."
	@if [ "$$(docker ps -a -f name=$(CONTAINER_NAME) --format '{{.Names}}')" = "$(CONTAINER_NAME)" ]; then \
		echo "Container $(CONTAINER_NAME) already exists. Removing the container..."; \
		docker rm -f $(CONTAINER_NAME); \
	fi

	@echo "Running the Docker container $(CONTAINER_NAME)..."
	# Start the container and run in detached mode, but don't start the app yet
	docker run -d -p $(PORT):5000 --name $(CONTAINER_NAME) $(IMAGE_NAME) tail -f /dev/null
	@echo "Docker container $(CONTAINER_NAME) is running without starting the app."

	# Conditional copying of preprocessed data and the model directory
	@if [ -f $(PREPROCESSED_DATA_FILE) ]; then \
		echo "Copying $(PREPROCESSED_DATA_FILE) into the container..."; \
		docker cp $(PREPROCESSED_DATA_FILE) $(CONTAINER_NAME):/app/$(PREPROCESSED_DATA_FILE); \
		echo "$(PREPROCESSED_DATA_FILE) copied successfully."; \
	fi

	# Conditional copying of music_genre_classifier_saved_model if it exists
	@if [ -d $(MODEL_DIR) ]; then \
		echo "Copying $(MODEL_DIR) into the container..."; \
		docker cp $(MODEL_DIR) $(CONTAINER_NAME):/app/$(MODEL_DIR); \
		echo "$(MODEL_DIR) copied successfully."; \
	fi

	# Conditional copying of fma_medium and fma_metadata directories if they exist AND if MODEL_DIR and PREPROCESSED_DATA_FILE do not exist
	@if [ ! -d "$(MODEL_DIR)" ] && [ ! -f "$(PREPROCESSED_DATA_FILE)" ]; then \
		echo "Neither $(MODEL_DIR) nor $(PREPROCESSED_DATA_FILE) exist. Preparing to copy $(FMA_MEDIUM) and $(FMA_METADATA)..."; \
		if [ -d "$(FMA_MEDIUM)" ]; then \
			echo "Copying $(FMA_MEDIUM) into the container..."; \
			docker cp "$(FMA_MEDIUM)" $(CONTAINER_NAME):/app/$(FMA_MEDIUM); \
			echo "$(FMA_MEDIUM) copied successfully."; \
		else \
			echo "$(FMA_MEDIUM) does not exist, skipping."; \
		fi; \
		if [ -d "$(FMA_METADATA)" ]; then \
			echo "Copying $(FMA_METADATA) into the container..."; \
			docker cp "$(FMA_METADATA)" $(CONTAINER_NAME):/app/$(FMA_METADATA); \
			echo "$(FMA_METADATA) copied successfully."; \
		else \
			echo "$(FMA_METADATA) does not exist, skipping."; \
		fi; \
	fi

	@echo "Starting the app after all files have been copied..."
	# Now that files have been copied, we start the app
	docker exec $(CONTAINER_NAME) /bin/sh -c "python3 test.py"
	@echo "App started."

# Show logs from the running container
logs:
	@echo "Displaying logs from container $(CONTAINER_NAME)..."
	docker logs -f $(CONTAINER_NAME)

# Stop the running container
stop:
	@echo "Stopping container $(CONTAINER_NAME)..."
	-@docker stop $(CONTAINER_NAME)
	@echo "Container $(CONTAINER_NAME) stopped."

# Remove the stopped container
clean:
	@echo "Removing container $(CONTAINER_NAME)..."
	-@docker rm $(CONTAINER_NAME)
	@echo "Container $(CONTAINER_NAME) removed."

# Rebuild the Docker image and run the container
re: stop clean build run

# Clean all Docker resources (containers, images, class weights, and model files)
modelclean: stop clean
	@echo "Removing Docker image and files..."
	-@docker rmi $(IMAGE_NAME)
	-rm -f $(CLASS_WEIGHTS_FILE)
	-rm -rf $(MODEL_DIR)
	-rm -f $(PREPROCESSED_DATA_FILE)
	@echo "Docker image $(IMAGE_NAME) and trained model files removed."

.PHONY: build run stop clean re logs fclean
