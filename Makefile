# Variables
IMAGE_NAME = music-genre-classifier
CONTAINER_NAME = music-genre-container
PORT = 5000
CLASS_WEIGHTS_FILE = class_weights.txt
MODEL_DIR = music_genre_classifier_saved_model
KERAS_MODEL = music_genre_classifier.keras
H5_MODEL = music_genre_classifier.h5

# Build, run, and display logs
all: build run logs

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container (detached mode)
run:
	@if [ $$(docker ps -a -f name=$(CONTAINER_NAME) --format '{{.Names}}') = $(CONTAINER_NAME) ]; then \
		docker rm -f $(CONTAINER_NAME); \
	fi
	docker run -d -p $(PORT):5000 --name $(CONTAINER_NAME) -v $(shell pwd)/uploads:/app/uploads -v $(shell pwd)/fma_small:/app/fma_small $(IMAGE_NAME)

# Show logs from the running container
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop the running container
stop:
	-@docker stop $(CONTAINER_NAME)

# Remove the stopped container
clean:
	-@docker rm $(CONTAINER_NAME)

# Rebuild the Docker image and run the container
re: stop clean build run

# Clean all Docker resources (containers, images, class weights, and model files)
fclean: stop clean
	@echo "Removing Docker image and files..."
	-@docker rmi $(IMAGE_NAME)
	-rm -f $(CLASS_WEIGHTS_FILE)
	-rm -rf $(MODEL_DIR)
	-rm -f $(KERAS_MODEL)
	-rm -f $(H5_MODEL)

.PHONY: build run stop clean re logs fclean
