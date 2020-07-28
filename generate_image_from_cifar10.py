import tensorflow as tf
import cv2
import os

def main(N=50, outfolder="images"):

  (_, _), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
  for itr in range(N):
    x, y = test_x[itr], test_y[itr][0]
    path = os.path.join(outfolder,f"img_{itr}_label_{y}.jpg")
    cv2.imwrite(path, x[:,:,::-1])

if __name__ == "__main__":
  main()
