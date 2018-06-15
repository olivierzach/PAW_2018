# sentiment_glove.py
# demo of sequential model sentiment analysis
# using pre-built GloVe word embeddings with LSTM
# dummy hard-coded data

import numpy as np
np.random.seed(1)

import keras as K
import keras.preprocessing.text as kpt
import keras.preprocessing.sequence as kps

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  reviews = [
    'great movie',
    # 'Great Movie!', -- same
    'nice film',
    'good story and plot',
    'not bad at all',
    'excellent film',

    'not good period',
    'poor movie',
    'weak story',
    'bad',
    'sad excuse']

  print("\nMicro-reviews are: ")
  print(reviews)
  
  labels = np.array([1,1,1,1,1, 0,0,0,0,0], dtype=np.int32)

  t = kpt.Tokenizer(num_words=None, lower=True, split=" ")
  t.fit_on_texts(reviews)
  vocab_size = len(t.word_index) + 1

  print("\nEncoding and cleaning reviews")
  encoded_revs = t.texts_to_sequences(reviews)

  max_length = 4
  padded_revs = kps.pad_sequences(encoded_revs,
    maxlen=max_length, padding='pre')
  print("\nEncoded and padded reviews: ")
  print(padded_revs)

  print("Loading GloVe embedding data ")
  embedding_dict = dict()
  file_path = ".\\Data\\glove.6B.100d.txt"

  f = open(file_path, "r", encoding="utf8")
  for line in f:
    values = line.split()
    word = values[0]
    word_vec = np.asarray(values[1:], dtype=np.float32)
    embedding_dict[word] = word_vec
  f.close()
  print('Loaded %s GloVe word vectors.' % len(embedding_dict))

  vec = embedding_dict['the']
  print("vec for word \'the\' = ")
  np.set_printoptions(precision=4, suppress=True)
  print(vec)

  embedding_matrix = np.zeros((vocab_size, 100))
  for word, i in t.word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

  print("\nCreating LSTM prediction model")
  init = K.initializers.glorot_uniform(seed=1)
  model = K.models.Sequential()
  e = K.layers.Embedding(vocab_size, 100,
    weights=[embedding_matrix], input_length=4,
    embeddings_initializer=init, trainable=False)
  model.add(e)
  model.add(K.layers.Flatten())
  model.add(K.layers.Dense(1, activation='sigmoid',
    kernel_initializer=init))

  model.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=['acc'])

  # print("\nmodel summary: \n")
  # print(model.summary())

  print("Starting training ")
  model.fit(padded_revs, labels, epochs=50, verbose=0)
  print("Training complete")

  loss, accuracy = model.evaluate(padded_revs, labels,
    verbose=0)
  print('\nTrained model accuracy: %0.2f%%' % (accuracy*100))

  inpt = np.array([[0, 0, 8, 1]], dtype=np.float32)  # 'nice movie'
  print("\nPredicting sentiment for micro-review \'nice movie\' ")
  print(inpt)
  sentiment = model.predict(inpt)
  print("Prediction (0 = negative, 1 = positive): ")
  print(sentiment)

if __name__=="__main__":
  main()

