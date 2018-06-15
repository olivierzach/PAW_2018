# converter_keras.py
# raw binary MNIST to Keras text file
#
# locate the zipped or g-zipped data and extract
# note the path to the four binary files:
# t10k-images.idx3-ubyte.bin, t10k-labels.idx1-ubyte.bin
# train-images.idx3-ubyte.bin, train-labels.idx1-ubyte.bin

# target format:
# 5 ** 0 0 152 27 .. 0
# 3 ** 0 0 38 122 .. 0
# single label val at [0] and 784 vals at [2-785] = [2-786)
# dummy ** seperator at [1] 

def generate(img_bin_file, lbl_bin_file,
            result_file, n_images):

  img_bf = open(img_bin_file, "rb")    # binary image pixels
  lbl_bf = open(lbl_bin_file, "rb")    # binary labels
  res_tf = open(result_file, "w")      # result file

  img_bf.read(16)   # discard image header info
  lbl_bf.read(8)    # discard label header info

  for i in range(n_images):   # number images requested 
    # digit label first
    lbl = ord(lbl_bf.read(1))  # get label like '3' (one byte) 
    res_tf.write(str(lbl)) 

    res_tf.write(" ** ")  # arbitrary seperator char for readibility

    # now do the image pixels
    for j in range(784):  # get 784 vals for each image file
      val = ord(img_bf.read(1))
      res_tf.write(str(val))
      if j != 783: res_tf.write(" ")  # avoid trailing space 
    res_tf.write("\n")  # next image

  img_bf.close(); lbl_bf.close();  # close the binary files
  res_tf.close()                   # close the result text file

# ================================================================

def main():
  # change the paths to point to the directory holding the unzipped files
  generate(".\\MNIST_Data\\MNIST_Data\\train-images.idx3-ubyte.bin",
          ".\\MNIST_Data\\MNIST_Data\\train-labels.idx1-ubyte.bin",
          ".\\mnist_train_keras_1000.txt",
          n_images = 1000)  # first n images

  generate(".\\MNIST_Data\\MNIST_Data\\t10k-images.idx3-ubyte.bin",
          ".\\MNIST_Data\\MNIST_Data\\t10k-labels.idx1-ubyte.bin",
          ".\\mnist_test_keras_100.txt",
          n_images = 100)  # first n images

if __name__ == "__main__":
  main()
