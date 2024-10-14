# Variables
IMAGE_NAME = music-genre-classifier
CONTAINER_NAME = music-genre-container
PORT = 5000

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
	docker stop $(CONTAINER_NAME)

# Remove the stopped container
clean:
	docker rm $(CONTAINER_NAME)

# Rebuild the Docker image and run the container
re: stop clean build run

# Clean all Docker resources (containers and images)
fclean: stop clean
	docker rmi $(IMAGE_NAME)

.PHONY: build run stop clean re logs fclean