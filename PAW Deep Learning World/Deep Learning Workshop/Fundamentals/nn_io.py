# nn_io.py
# just plain Python 3.x and NumPy
# Katherine Ng

import numpy as np

def my_print(arr, decs, shapes):
  # print serialized-as-arrary matrices
  fmt = "% 0." + str(decs) + "f " 
  k = 0;
  for t in shapes:
    for i in range(t[0]):
      for j in range(t[1]):
        print(fmt % arr[k], end="")
        k += 1
      print("")
    print("")

class NeuralNetwork:

  def __init__(self, num_input, num_hidden, num_output, seed):
    self.ni = num_input
    self.nh = num_hidden
    self.no = num_output
	
    self.i_nodes = np.zeros(shape=[self.ni], dtype=np.float32)
    self.h_nodes = np.zeros(shape=[self.nh], dtype=np.float32)
    self.o_nodes = np.zeros(shape=[self.no], dtype=np.float32)
	
    self.ih_weights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
    self.ho_weights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)
	
    self.h_biases = np.zeros(shape=[self.nh], dtype=np.float32)
    self.o_biases = np.zeros(shape=[self.no], dtype=np.float32)
	
    self.rnd = np.random.RandomState(seed) 
    self.init_weights()
 	
  def set_weights(self, weights):
    # ih_wts -> h_biases -> ho_wts -> o_biases
    if len(weights) != self.total_weights(self.ni, self.nh, self.no):
      print("Warning: len(weights) error in set_weights()")	

    idx = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.ih_weights[i,j] = weights[idx]
        idx += 1
		
    for j in range(self.nh):
      self.h_biases[j] = weights[idx]
      idx += 1

    for j in range(self.nh):
      for k in range(self.no):
        self.ho_weights[j,k] = weights[idx]
        idx += 1
	  
    for k in range(self.no):
      self.o_biases[k] = weights[idx]
      idx += 1
	  
  def get_weights(self):
    tw = self.total_weights(self.ni, self.nh, self.no)
    result = np.zeros(shape=[tw], dtype=np.float32)
    idx = 0  # points into result
    
    for i in range(self.ni):
      for j in range(self.nh):
        result[idx] = self.ih_weights[i,j]
        idx += 1
		
    for j in range(self.nh):
      result[idx] = self.h_biases[j]
      idx += 1

    for j in range(self.nh):
      for k in range(self.no):
        result[idx] = self.ho_weights[j,k]
        idx += 1
	  
    for k in range(self.no):
      result[idx] = self.o_biases[k]
      idx += 1
	  
    return result
 	
  def init_weights(self):
    num_wts = self.total_weights(self.ni, self.nh, self.no)
    lo = -0.01; hi = 0.01
    wts = self.rnd.uniform(lo, hi, num_wts)
    self.set_weights(wts)

  def compute_outputs(self, x_values):
    h_sums = np.zeros(shape=[self.nh], dtype=np.float32)
    o_sums = np.zeros(shape=[self.no], dtype=np.float32)

    for i in range(self.ni):
      self.i_nodes[i] = x_values[i] 

    for j in range(self.nh):
      for i in range(self.ni):
        h_sums[j] += self.i_nodes[i] * self.ih_weights[i,j]

    for j in range(self.nh):
      h_sums[j] += self.h_biases[j]
	  
    for j in range(self.nh): 
      #self.h_nodes[j] = self.my_tanh(h_sums[j])
      self.h_nodes[j] = self.my_logsig(h_sums[j])

    for k in range(self.no):
      for j in range(self.nh):
        o_sums[k] += self.h_nodes[j] * self.ho_weights[j,k]

    for k in range(self.no):
      o_sums[k] += self.o_biases[k]
 
    soft_out = self.softmax(o_sums)
    for k in range(self.no):
      self.o_nodes[k] = soft_out[k]
	  
    result = np.zeros(shape=self.no, dtype=np.float32)
    for k in range(self.no):
      result[k] = self.o_nodes[k]
	  
    return result

  @staticmethod
  def my_tanh(x):
    if x < -20.0:
      return -1.0
    elif x > 20.0:
      return 1.0
    else:
      return np.tanh(x)

  @staticmethod
  def my_logsig(x):
    if x < -20.0:
      return 0.0
    elif x > 20.0:
      return 1.0
    else:
      return 1.0 / (1.0 + np.exp(-x))  

  @staticmethod	  
  def softmax(o_sums):
    # naive - not for production - overflow
    result = np.zeros(shape=[len(o_sums)], dtype=np.float32)
    sum = 0.0
    for k in range(len(o_sums)):
      sum += np.exp(o_sums[k])
    for k in range(len(result)):
      result[k] =  np.exp(o_sums[k]) / sum
    return result
	
  @staticmethod
  def total_weights(n_input, n_hidden, n_output):
   tw = (n_input * n_hidden) + n_hidden + (n_hidden * n_output) + n_output
   return tw

# end class NeuralNetwork

def main():
  print("\nBegin NN IO demo ")
  print("Katherine Ng")
  
  num_input = 4
  num_hidden = 5
  num_output = 3
  print("\nCreating a %d-%d-%d neural network " %
    (num_input, num_hidden, num_output) )
  nn = NeuralNetwork(num_input, num_hidden, num_output, seed=3)

  wts = np.array([
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15,
    0.16, 0.17, 0.18, 0.19, 0.20,

    0.21, 0.22, 0.23, 0.24, 0.25,

    0.26, 0.27, 0.28,
    0.29, 0.30, 0.31,
    0.32, 0.33, 0.34,
    0.35, 0.36, 0.37,
    0.38, 0.39, 0.40,

    0.41, 0.42, 0.43], dtype=np.float32)

  np.set_printoptions(precision=4, suppress=True)
  # print("Setting weights and biases: \n")
  # print(wts)
  my_print(wts, 4, [(4,5),(1,5),(5,3),(1,3)])
  nn.set_weights(wts)
  
  inpts = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
  np.set_printoptions(precision=1)  
  print("Input = ")
  print(inpts)

  probs = nn.compute_outputs(inpts)
  np.set_printoptions(precision=4)
  print("\nOutput = ")
  print(probs)
 
if __name__ == "__main__":
  main()

# end script

