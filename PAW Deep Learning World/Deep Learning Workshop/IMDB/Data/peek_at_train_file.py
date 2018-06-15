# peek_at_train_file.py
# display first n lines of a (big) training file

f = open(".\\imdb_train_50w.txt", "r")

ctr = 0
num_lines_to_show = 2
for line in f:
  ctr += 1
  if ctr > num_lines_to_show: break
  print(line)

f.close()

