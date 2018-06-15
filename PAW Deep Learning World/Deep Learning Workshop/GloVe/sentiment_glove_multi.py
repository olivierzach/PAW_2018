# sentiment_glove_multi.py
# demo of sequential model sentiment analysis
# using pre-built GloVe word embeddings with LSTM
# dummy hard-coded data
# multi-class: neg (1 0 0), neutral (0 1 0), pos (0 0 1)

import numpy as np
np.random.seed(1)

import keras as K
import keras.preprocessing.text as kpt
import keras.preprocessing.sequence as kps

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

def main():
  print("\nSentiment analysis using GloVe \n")

  reviews = [
    'not a good film period',
    'poor movie and bad acting',
    'weak story',
    'bad',
    'sad excuse of a movie',

    'so-so plot, so-so movie',
    'mediocre at best',
    'not good, not bad', 
    'i\'ve seen worse',
    'neutral opinion on this film',

    'this is a great movie',
    'nice film in every way',
    'good story and plot',
    'not bad at all',
    'excellent film' ]

  labels = np.array([[1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0],
                     [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0],
                     [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]],
                     dtype=np.int32)      

  t = kpt.Tokenizer(num_words=None, lower=True, split=" ")
  t.fit_on_texts(reviews)
  vocab_size = len(t.word_index) + 1

  for (k,v) in t.word_index.items():
    print(str(k) + " " + str(v))

  encoded_revs = t.texts_to_sequences(reviews)

  max_length = 5
  padded_revs = kps.pad_sequences(encoded_revs,
    maxlen=max_length, padding='pre')
  print("\nEncoded and padded reviews: ")
  print(padded_revs)

  embedding_dict = dict()
  file_path = ".\\Data\\glove.6B.100d.txt"

  f = open(file_path, "r", encoding="utf8")
  for line in f:
    values = line.split()
    word = values[0]
    word_vec = np.asarray(values[1:], dtype=np.float32)
    embedding_dict[word] = word_vec
  f.close()
  print('\nLoaded %s GloVe word vectors.' % len(embedding_dict))

  # vec = embedding_dict['the']
  # print("vec for word \'the\' = ")
  # np.set_printoptions(precision=4, suppress=True)
  # print(vec)
  # print("")

  embedding_matrix = np.zeros((vocab_size, 100))
  for word, i in t.word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

  init = K.initializers.glorot_uniform(seed=1)
  model = K.models.Sequential()
  e = K.layers.Embedding(vocab_size, 100,
    weights=[embedding_matrix], input_length=max_length,
    embeddings_initializer=init, trainable=False)
  model.add(e)
  model.add(K.layers.Flatten())
  model.add(K.layers.Dense(3, activation='softmax',
    kernel_initializer=init))

  model.compile(optimizer='adam', loss='categorical_crossentropy',
    metrics=['acc'])

  # print("\nmodel summary: \n")
  # print(model.summary())

  print("Starting training ")
  model.fit(padded_revs, labels, epochs=50, verbose=0)
  print("Training complete")

  loss, accuracy = model.evaluate(padded_revs, labels,
    verbose=0)
  print('\nTrained model accuracy: %0.2f%%' % (accuracy*100))

  inpt = np.array([[0, 1, 6, 4, 3]], dtype=np.float32)  # 
  print("\nPredicting sentiment for phrase \'not a bad movie\' ")
  print(inpt)
  sentiment = model.predict(inpt)
  print("Prediction (1,0,0) = negative, (0,1,0) = neutral, (0,0,1) = positive): ")
  print(sentiment)

if __name__=="__main__":
  main()

