# music-genre-classification
study for machine learning

```
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
wget unzip fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
```
```
docker build -t music-genre-classifier .
docker run -d -p 5000:5000 -v ~/music-genre-classification/fma_small:/app/fma_small music-genre-classifier
```
