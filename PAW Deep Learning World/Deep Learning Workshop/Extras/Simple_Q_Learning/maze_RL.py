# maze_RL.py
# solve a 3x4 maze using Q learning
# lots of hard-coding badness here

import numpy as np

def my_print(Q, dec):
  # complete hack for this problem only
  fmt = "% 6.2f "
  rows = len(Q); cols = len(Q[0])
  print("       0      1      2      3      4      5\
      6      7      8      9      10     11")
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: print(" ", end="")
    for j in range(cols):
      print(fmt % Q[i,j], end="")
    print("")
  print("")

def get_poss_next_states(s, f):
  # given a state s and a feasibility matrix f
  # get list of possible next states
  poss_next_states = []
  for j in range(12):
    if f[s,j] == 1: poss_next_states.append(j)
  return poss_next_states

def get_rnd_next_state(s, f):
  # given a state s, pick a feasible next state
  poss_next_states = get_poss_next_states(s, f)
  next_state = \
    poss_next_states[np.random.randint(0,len(poss_next_states))]
  return next_state 

def train(f, R, Q, gamma, max_epochs):
  # find the Q matrix
  for i in range(0,max_epochs):
    curr_s = np.random.randint(0,12)  # random start state

    while(True):
      next_s = get_rnd_next_state(curr_s, f)
      poss_next_next_states = get_poss_next_states(next_s, f)

      max_Q = -9999.99
      for j in range(len(poss_next_next_states)):
        nn_s = poss_next_next_states[j]
        q = Q[next_s,nn_s]
        if q > max_Q:
          max_Q = q
   
      Q[curr_s, next_s] = R[curr_s, next_s] + (gamma * max_Q)

      curr_s = next_s
      if curr_s == 11: break

def walk(start, Q):
  # go to goal (11) from start using Q
  curr = start
  print(str(curr) + "->", end="")
  while curr != 11:
    next = np.argmax(Q[curr])
    print(str(next) + "->", end="")
    curr = next
  print("done")
  
def main():
  np.random.seed(1)
  # s = np.array([0,1,2,3,4,5,6,7,8,9,10,11], dtype=np.int)

  print("\nSetting up maze")

  f = np.zeros(shape=[12,12], dtype=np.int)  # feasible actions
  f[0,1] = 1; f[0,4] = 1
  f[1,0] = 1; f[1,5] = 1
  f[2,3] = 1; f[2,6] = 1
  f[3,2] = 1; f[3,7] = 1
  f[4,0] = 1; f[4,8] = 1
  f[5,1] = 1; f[5,6] = 1; f[5,9] = 1
  f[6,2] = 1; f[6,5] = 1; f[6,7] = 1
  f[7,3] = 1; f[7,6] = 1; f[7,11] = 1
  f[8,4] = 1; f[8,9] = 1
  f[9,5] = 1; f[9,8] = 1; f[9,10] = 1
  f[10,9] = 1
  f[11,11] = 1

  R = np.zeros(shape=[12,12], dtype=np.float32)  # Reward
  R[0,1] = -0.1; R[0,4] = -0.1
  R[1,0] = -0.1; R[1,5] = -0.1
  R[2,3] = -0.1; R[2,6] = -0.1
  R[3,2] = -0.1; R[3,7] = -0.1
  R[4,0] = -0.1; R[4,8] = -0.1
  R[5,1] = -0.1; R[5,6] = -0.1; R[5,9] = -0.1
  R[6,2] = -0.1; R[6,5] = -0.1; R[6,7] = -0.1
  R[7,3] = -0.1; R[7,6] = -0.1; R[7,11] = -0.1
  R[8,4] = -0.1; R[8,9] = -0.1
  R[9,5] = -0.1; R[9,8] = -0.1; R[9,10] = -0.1
  R[10,9] = -0.1
  R[7,11] = 10.0

  Q = np.zeros(shape=[12,12], dtype=np.float32)  # Quality
  
  print("Analyzing maze with RL Q learning")
  gamma = 0.5
  max_epochs = 1000
  train(f, R, Q, gamma, max_epochs)
  print("Done ")
  
  print("The Q matrix is: \n ")
  my_print(Q, 2)

  print("Using Q to go from 8 to goal (11)")
  start = 8
  walk(start, Q)

if __name__ == "__main__":
  main()

