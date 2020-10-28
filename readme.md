# A Simple Example for Coral TPU USB Accelerator on raspberrypi

0. Follow the install instruction from https://coral.ai/docs/accelerator/get-started/
1. Run Ipynb file, then download the `*_edgetpu.tflite` to your device. [link to this Ipynb File](https://colab.research.google.com/github/mistake0316/coral_cifar10_raspberrypi_example/blob/master/Build_Simple_Cifar10_Model.ipynb)
2. Move `*_edgetpu.tflite` to ./models
3. Select one line from following lines
```bash
   # if you have setup raspberrypi's camera
   python3 simple.py # get image from your pi camera
   # else 
   python3 simple.py ./images/img_0_label_3.jpg # you can replace the image here(should have size 32x32)
```
4. You can view the reslut and screen shot in output folder
