import os
import numpy as np
import tflite_runtime.interpreter as tflite
import platform
from PIL import Image
from camera import get_one_image_raspberrypi

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

model_file = "./models/Cifar10_CNN_quant_edgetpu.tflite"
image_file = "./images/img_0_label_3.jpg"
label_file = "./models/Cifar10_label.txt"

interpreter = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,{})
        ]
      )

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# image = np.array(Image.open(image_file).convert("RGB")).astype(np.float32)
PIL_image = get_one_image_raspberrypi(delay=5,preview=True).convert("RGB").resize(input_details[0]['shape'][-3:-1])
image = np.array(PIL_image).astype(np.float32)
image = np.expand_dims(image,0)

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

label_map = dict() 
with open(label_file) as f:
  lines = f.read().split('\n')
  for line in lines:
    tokens = line.split(" ")
    if len(tokens) != 2:
      continue
    label_map[int(tokens[0])] = tokens[1]

num_of_classes = len(label_map)

rank = sorted(range(num_of_classes), key=lambda itr:output_data[itr], reverse=True)

print("predictions:")
for ith_rank in range(num_of_classes):
  print(f"{label_map[rank[ith_rank]]:>10}{rank[ith_rank]:>10}{output_data[rank[ith_rank]]:>10.3f}")

folder = "output"
try:
  os.mkdir(folder)
except:
  pass
out_name = folder + os.sep + f"{label_map[rank[0]]}_{rank[0]}_{output_data[rank[0]]:>.2f}.jpg"
print("save file to", out_name, "...")
PIL_image.save(out_name)
