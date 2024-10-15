_[Ongoing project] : trying to apply class weights to address the class imbalance issue for higher accuracy_
_+ apply openai api for better result user friendly info_


# music-genre-classification
study for machine learning


## Required Environment
> ### Essential Software:
* **Docker**:  Required to run the project in a containerized environment.
* **Make**:  Used to automate build and execution tasks via the `Makefile`.
* **Python 3.8** or **higher**:  The project has been tested on Python 3.8 or above.
> ### Required Libraries:
  Libraries listed in `requirements.txt` are needed. These will be installed automatically within the Docker container.
> ### Dataset:
  The project normally contains `music_genre_classifier_saved_model` file as default pre-trained model but when it is not exist,
  The `fma_medium` dataset is used by default. If storage space is limited, the `fma_small` dataset can be used instead.


## Usage
```
git clone https://github.com/coisu/music-genre-classification.git
cd music-genre-classification
```
> before do 'make' check the Pre-trained model exist.
```
ls music_genre_classifier_saved_model
```
> if it is not exist, run
```
wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip
unzip fma_medium.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
```
> if you have storage isuue, use 'fma_small.zip' instead 'fma_medium.zip'
> then
```
make
```




# ENV
> **.env.gpg**

encrypt environment file
```
make encrypt-mama
```
decrypt env file
```
make decrypt-mama
```
delete decrypted env file
```
make clean-env
```



```
docker build -t music-genre-classifier .
docker run -d -p 5000:5000 -v ~/music-genre-classification/fma_small:/app/fma_small music-genre-classifier
```
