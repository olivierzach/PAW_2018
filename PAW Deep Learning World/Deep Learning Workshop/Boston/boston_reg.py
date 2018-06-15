# boston_reg.py
# regression on the Boston Housing dataset
# Keras 2.1.5 over TensorFlow 1.7.0

import numpy as np
import keras as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MyLogger(K.callbacks.Callback):
  def __init__(self, n, data_x, data_y, pct_close):
    self.n = n   # print loss & acc every n epochs
    self.data_x = data_x
    self.data_y = data_y
    self.pct_close = pct_close

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss = logs.get('loss')  # loss on curr batch
      total_acc = my_accuracy(self.model, self.data_x,
        self.data_y, self.pct_close)
      print("epoch = %4d  curr batch loss (mse) = %0.6f \
 overall acc = %0.2f%%" % (epoch, curr_loss, total_acc * 100))

def my_accuracy(model, data_x, data_y, pct_close):
  correct = 0; wrong = 0
  n = len(data_x)
  for i in range(n):
    predicted = model.predict(np.array([data_x[i]],
      dtype=np.float32) )  # [[ x ]]
    actual = data_y[i]
    if np.abs(predicted[0][0] - actual) < \
      np.abs(pct_close * actual):
      correct += 1
    else:
      wrong += 1
  return (correct * 1.0) / (correct + wrong)

# def my_mse(model, data_x, data_y):
#   n = len(data_x)
#   sum_sq_err = 0.0
#   for i in range(n):
#     predicted = model.predict(np.array([data_x[i]],
#        dtype=np.float32) )  # [[ x ]]
#     actual = data_y[i]
#     sum_sq_err += (predicted[0][0] - actual) *
#       (predicted[0][0] - actual)
#   return sum_sq_err / n 

def main():
  print("\nBoston Houses dataset regression example \n")
  np.random.seed(1)

  kv = K.__version__
  print("Using Keras: ", kv, "\n")

  # 506 items min-max, median value / 10
  data_file = ".\\Data\\boston_mm_tab.txt"  
  all_data = np.loadtxt(data_file, delimiter="\t",
    skiprows=0, dtype=np.float32) 
  n = len(all_data)  # number rows
  indices = np.arange(n)  # an array [0, 1, . . n-1]
  np.random.shuffle(indices)     # by ref
  ntr = int(0.90 * n)  # number training items
  data_x = all_data[indices,:-1]  
  data_y = all_data[indices,-1] 
  train_x = data_x[0:ntr,:] 
  train_y = data_y[0:ntr] 
  test_x = data_x[ntr:n,:]
  test_y = data_y[ntr:n]
  
  print("Creating 13-(10-10)-1 DNN with Adam, Glorot uniform")
  cust_adam = K.optimizers.Adam(lr=0.15, beta_1=0.95, beta_2=0.9)
  init = K.initializers.glorot_uniform(seed=1)

  model = K.models.Sequential()
  model.add(K.layers.Dense(units=10, input_dim=13,
    activation='tanh', kernel_initializer=init))
  model.add(K.layers.Dense(units=10,
    activation='tanh', kernel_initializer=init)) 
  model.add(K.layers.Dense(units=1, activation=None)) 
  model.compile(loss='mean_squared_error',
    optimizer=cust_adam, metrics=['mse'])

  max_epochs = 10 
  my_logger = MyLogger(int(max_epochs/10),
    train_x, train_y, 0.15) 
  print("Starting training")
  h = model.fit(train_x, train_y, batch_size=1,
    epochs=max_epochs, verbose=0, callbacks=[my_logger])
  print("Trainijng complete")

  mp = ".\\Models\\boston_model.h5"
  model.save(mp)

  acc = my_accuracy(model, test_x, test_y, 0.15) 
  print("\nModel accuracy on test data  = %0.4f " % acc)

  # [0] = loss (mse), [1] = compile-metrics = 'mse' again
  eval_results = model.evaluate(train_x, train_y, verbose=0)  
  print("Model loss (mse) on train data = %0.6f" % eval_results[0])

  eval_results = model.evaluate(test_x, test_y, verbose=0)
  print("Model loss (mse) on test data = %0.6f" % eval_results[0])

  # loss_list = h.history['loss']  # loss of LAST BATCH every epoch
  # print(loss_list[len(loss_list)-1])

  # agrees with model.evaluate() result but not history['loss']
  # my_err = my_mse(model, train_x, train_y)  
  # print("My MSE = %0.6f \n" % my_err)

if __name__=="__main__":
  main()
