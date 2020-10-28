import tensorflow as tf
import cv2
import os
import sys

def main(N=50, outfolder="images"):

  (_, _), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
  for itr in range(N):
    x, y = test_x[itr], test_y[itr][0]
    path = os.path.join(outfolder,f"img_{itr}_label_{y}.jpg")
    cv2.imwrite(path, x[:,:,::-1])

if __name__ == "__main__":
  if len(sys.argv) > 1:
    try:
      main(int(sys.argv[1]))
      exit()
    except:
      print("have some error, try default version (N=50)")
      pass
  main()
