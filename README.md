[Ongoing project] : trying to apply class weights to address the class imbalance issue 


# music-genre-classification
study for machine learning

# related env, tools, frameworks
linux Docker tensorflow flask Makefile

# 

```
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
```
```
docker build -t music-genre-classifier .
docker run -d -p 5000:5000 -v ~/music-genre-classification/fma_small:/app/fma_small music-genre-classifier
```
