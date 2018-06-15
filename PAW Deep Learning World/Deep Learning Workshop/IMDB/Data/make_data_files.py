# make_data_files.py
#
# input: source Stanford 50,000 data files reviews
# output: one combined train file, one combined test file
# output files are in index version, using the Keras dataset
# format where 0 = padding, 1 = 'start', 2 = OOV, 3 = unused
# 4 = most frequent word ('the'), 5 = next most frequent, etc.
# i'm skipping the start=1 because it makes no sense.
# these data files will be loaded into memory then feed
# a built-in Embedding layer (rather than custom embeddings)

import os

# allow the Windws cmd shell to deal with wacky characters
import sys
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)

# ---------------------------------------------------------------

def get_reviews(dir_path, num_reviews, punc_str):
  punc_table = {ord(char): None for char in punc_str}  # dictionary 
  reviews = []  # list-of-lists of words
  ctr = 1
  for file in os.listdir(dir_path):
    if ctr > num_reviews: break
    curr_file = os.path.join(dir_path, file)
    f = open(curr_file, "r", encoding="utf8")   # each review has only one line . . . 
    for line in f:
      line = line.strip()
      if len(line) > 0:  # number characters
        # print(line)  # to show non-ASCII == errors
        line = line.translate(punc_table)  # remove punc
        line = line.lower()  # lower case
        line = " ".join(line.split())  # remove consecutive WS
        word_list = line.split(" ")  # one review is a list of words
        reviews.append(word_list)    # 
    f.close()  # close curr file
    ctr += 1
  return reviews

# ---------------------------------------------------------------

def make_vocab(all_reviews):
  word_freq_dict = {}   # key = word, value = frequency

  for i in range(len(all_reviews)):
    reviews = all_reviews[i]
    for review in reviews:
      for word in review:
        if word in word_freq_dict:
          word_freq_dict[word] += 1
        else:
          word_freq_dict[word] = 1

  kv_list = []  # list of word-freq tuples so can sort
  for (k,v) in word_freq_dict.items():
    kv_list.append((k,v))

  # list of tuples where index is 0-based rank, val is (word,freq)
  sorted_kv_list = sorted(kv_list, key=lambda x: x[1], reverse=True)  # sort by freq

  vocab_dict = {}  # key = word, value = 1-based rank ('the' = 1, 'a' = 2, etc.)
  for i in range(len(sorted_kv_list)):
    w = sorted_kv_list[i][0]  # word is at [0]
    vocab_dict[w] = i+1       # 1-based as in Keras dataset

  return vocab_dict

# ---------------------------------------------------------------

def generate_file(reviews_lists, outpt_file, w_or_a, vocab_dict, max_review_len, label_char):
  fout = open(outpt_file, w_or_a, encoding="utf8")    # write first time, append later
  offset = 3  # Keras offset: 'the' = 1 (most frequent) -> 1+3 = 4
      
  for i in range(len(reviews_lists)):  # walk thru each review-list
    curr_review = reviews_lists[i]
    n_words = len(curr_review)     
    if n_words > max_review_len:
      continue  # next i, continue without writing anything
    n_pad = max_review_len - n_words   # number of 0s to prepend
    for j in range(n_pad):
      fout.write("0 ")
    for word in curr_review:      
      if word not in vocab_dict:  # a word in test set might not have been in traning set
        fout.write("2 ")   # 2 is the special out-of-vocab index        
      else:
        idx = vocab_dict[word] + offset
        fout.write("%d " % idx)
    fout.write(label_char + "\n")  # like '0' or '1', or 'N' or 'P"
        
  fout.close()

# ---------------------------------------------------------------          

def main():
  remove_chars = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"   # leave ' for words like it's

  # you'll probably have to change the file path here
  print("\nLoading all reviews into memory - be patient ")
  pos_train_reviews = get_reviews(".\\IMDB_Data\\train\\pos", 12500, remove_chars)
  neg_train_reviews = get_reviews(".\\IMDB_Data\\train\\neg", 12500, remove_chars)
  pos_test_reviews = get_reviews(".\\IMDB_Data\\test\\pos", 12500, remove_chars)
  neg_test_reviews = get_reviews(".\\IMDB_Data\\test\\neg", 12500, remove_chars)

  mp = max(len(l) for l in pos_train_reviews)  # 2469
  mn = max(len(l) for l in neg_train_reviews)  # 1520
  mm = max(mp, mn)  # longest (in words) review in training set  # 2469
  # print(mp, mn)

  print("\nAnalyzing reviews and making vocabulary ")
  vocab_dict = make_vocab([pos_train_reviews, neg_train_reviews])  # key = word, value = word rank
  v_len = len(vocab_dict)  # need this value, plus 4, for Embedding Layer: 129888+4 = 129892
  print("\nVocab size = %d -- use this +4 for Embedding nw " % v_len)

  max_review_len = 50   # use None for all reviews (any len)
  if max_review_len == None or max_review_len > mm:
    max_review_len = mm

  print("\nGenerating training file with len %d words or less " % max_review_len)

  generate_file(pos_train_reviews, ".\\imdb_train_50w.txt", "w", vocab_dict, max_review_len, "1")
  generate_file(neg_train_reviews, ".\\imdb_train_50w.txt", "a", vocab_dict, max_review_len, "0")

  print("\nGenerating test file with len %d words or less " % max_review_len)

  generate_file(pos_test_reviews, ".\\imdb_test_50w.txt", "w", vocab_dict, max_review_len, "1")
  generate_file(neg_test_reviews, ".\\imdb_test_50w.txt", "a", vocab_dict, max_review_len, "0")

  # inspect a generated file
  # vocab_dict was used indirectly (offset)

  # print("Displaying encoded training file: \n")
  # f = open(".\\imdb_train_50w.txt", "r", encoding="utf8")
  # for line in f: 
  #   print(line, end="")
  # f.close()

  # print("\nDisplaying decoded training file: \n") 

  # index_to_word = {}
  # index_to_word[0] = "<PAD>"
  # index_to_word[1] = "<ST>"
  # index_to_word[2] = "<OOV>"
  # for (k,v) in vocab_dict.items():
  #   index_to_word[v+3] = k

  # f = open(".\\imdb_train_50w.txt", "r", encoding="utf8")
  # for line in f:
  #   line = line.strip()
  #   indexes = line.split(" ")
  #   for i in range(len(indexes)-1):  # last is '0' or '1'
  #     idx = (int)(indexes[i])
  #     w = index_to_word[idx]
  #     print("%s " % w, end="")
  #   print("%s \n" % indexes[len(indexes)-1])
  # f.close()

  # generate encodings for a new review 
  # offset = 3
  # review = "i wish i could say i liked this movie but i can't"
  # words = review.split()
  # for i in range(max_review_len - len(words)):
  #   print("0 ", end="")
  # for w in words:
  #   idx = vocab_dict[w] + offset
  #   print(str(idx) + " ", end="")
  
if __name__ == "__main__":
  main()
