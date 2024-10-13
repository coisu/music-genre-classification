# music-genre-classification
study for machine learning



 docker build -t music-genre-classifier .
 docker run -d -p 5000:5000 -v ~/music-genre-classification/fma_small:/app/fma_small music-genre-classifier
