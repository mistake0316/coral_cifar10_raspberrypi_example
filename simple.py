import numpy as np
import tflite_runtime.interpreter as tflite
import platform
from PIL import Image


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

model_file = "./models/Cifar10_CNN_quant_edgetpu.tflite"
image_file = "./images/img_0_label_3.jpg"
label_file = "./models/Cifar10_label.txt"

image = np.array(Image.open(image_file).convert("RGB")).astype(np.float32)
image = np.expand_dims(image,0)
interpreter = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,{})
        ]
      )

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

