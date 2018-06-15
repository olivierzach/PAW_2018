# iris_nn.py
# Iris classification
# Python 3.5.2 Keras 2.1.5 TensorFlow 1.7.0

import numpy as np
import keras as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  print("Iris dataset using Keras/TensorFlow ")

  print("Loading Iris train/test data into memory \n")
  train_file = ".\\Data\\iris_train_data.txt"
  test_file = ".\\Data\\iris_test_data.txt"

  train_x = np.loadtxt(train_file, usecols=[0,1,2,3],
    delimiter=",",  skiprows=0, dtype=np.float32)
  train_y = np.loadtxt(train_file, usecols=[4,5,6],
    delimiter=",", skiprows=0, dtype=np.float32)

  test_x = np.loadtxt(test_file, usecols=[0,1,2,3],
    delimiter=",",  skiprows=0, dtype=np.float32)
  test_y = np.loadtxt(test_file, usecols=[4,5,6],
    delimiter=",", skiprows=0, dtype=np.float32)

  np.random.seed(4)
  model = K.models.Sequential()

  #Set random initial values for weights
  #Need to set random seed within this function in order to make it reproducible
  #This random seed is separate from the np.random.seed that was set earlier
  init = K.initializers.RandomUniform(minval=-0.01,
    maxval=0.01, seed=1)

#SGD is synonymous to backpropagation (not exactly but is for these purposes)
#Define customer stochastic gradient descent with non-standard values
#lr = learning rate (controls how much weights get adjusted each time)
#momentum = checks if error is getting smaller and then keeps going until model is no longer getting better
#Adam is considered the best general purpose optimizer

  #opt = K.optimizers.SGD(lr=0.05, momentum=0.50)

  #Input layer is implicitly defined by first explicit layer (input_dim = 4)
  model.add(K.layers.Dense(units=5, input_dim=4, activation='tanh', kernel_initializer=init)) #hidden layer
  model.add(K.layers.Dense(units=3, activation='softmax', kernel_initializer=init)) #output
  
  #Convert Keras code to Tensorflow
  #Loss is synonymous with error function
  #Compare computed outputs with actual outputs
  #Will go back and adjust weights to minimize error
  #optimizer = 'sgd' will use stochastic gradient descent with standard parameters
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  #batch_size = 1 (online training, analyze 1 row of data, update)
  #batch_size = 120 (whole training set, batch training)
  #batch size does not matter too much for simple neural networks
  #batch size does matter for LSTM
  
  print("Starting training \n")
  h = model.fit(train_x, train_y, batch_size=1, epochs=12, verbose=1)  # 1 = very chatty
  print("\nTraining finished \n")

  eval = model.evaluate(test_x, test_y, verbose=0)
  print("Test data: loss = %0.6f  accuracy = %0.2f%% \n" \
    % (eval[0], eval[1]*100) )

  mp = ".\\Models\\iris_model.h5"
  model.save(mp)

  np.set_printoptions(precision=4)
  unknown = np.array([[6.1, 3.1, 5.1, 1.1]],
    dtype=np.float32)
  predicted = model.predict(unknown)
  print("Using model to predict species for features: ")
  print(unknown)
  print("\nPredicted species is: ")
  print(predicted)

if __name__=="__main__":
  main()
