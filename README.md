# Eluvio-Data-Science-ML-Challenge

This repository houses my solution for Eluvio Data Science ML Challenge.

### Data Preparing

First, add hyperparameters for performing tokenization and preparing the standardized data representation

```python
vocab_size = 10000
embedding_dim = 16
max_length = 30
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```

Second, checking if there is null values.

Third, dropping unnecessary labels and using LabelEncoder() from sklearn.preprocessing to change the values of 'over_18' values from True and False to 1 and 0.

Forth, splitting train(80%), validation(10%) and test(10%).

Final, performing the tokenization and sequence padding for turn each 'title' into a sequence of integers.

```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(x_train)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
```

### Model Architecture
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 30, 16)            160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 64)                1088      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 161,153
Trainable params: 161,153
Non-trainable params: 0
_________________________________________________________________
```
### Training Results

![eluivo](https://user-images.githubusercontent.com/61292363/114931912-8c5d8700-9e3f-11eb-94be-875e1b28a148.png)

![Unknown1](https://user-images.githubusercontent.com/61292363/114931994-a8612880-9e3f-11eb-9d3f-ecba9d83ba26.png)

