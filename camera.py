# https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-to-a-pil-image
from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image
import numpy as np

def get_one_image_raspberrypi(delay=2, preview=False):
  # Create the in-memory stream
  stream = BytesIO()
  camera = PiCamera()
  camera.resolution = (256, 256)
  if preview:
    camera.start_preview()

  sleep(delay)
  camera.capture(stream, format='jpeg')
  # "Rewind" the stream to the beginning so we can read its content
  stream.seek(0)
  image = Image.open(stream)
  return image

if __name__ == "__main__":
  image = get_one_image_raspberrypi(preview=True)
  image.save("foo.jpg")
