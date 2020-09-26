## PSPNet_Cityscapes

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=master)](https://travis-ci.org/PaddlePaddle/PaddleSeg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)


### 1.Introduction
#### Paddlepaddle implementation of PSPNet, trained on the Cityscapes dataset.The mIOU is 70.54.

### 2.Directory structure and file description
```
├── infer.py #  Core code,to predict the segmentation model
├── freeze_model # The maching learning model training by PSPNet
├── test_img # The testing data
└── README.md # Documentation
```

### 3.Install dependency packages
#### Before you run the program,you should install `PaddlePaddle` ,`gflags`,`opencv` and so on.The required packages are as follows:


```python
!pip install python-gflags
!pip install pyyaml
!pip install opencv-python
!pip install futures
!pip install numpy
#pip install paddlepaddle
```

### 4.Prediction
#### Input the following command at the terminal to predict:
#### `!python infer.py --conf=./freeze_model/deploy.yaml --input_dir=./test_img`
#### Parameter description


| Parameter| Necessary | Description |
| -------- | -------- | -------- |
| conf     | Yes     | File path of model configuration  |
| input_dir     | Yes     | Directory of images to be predicted |

#### *If the hardware supports (such as Tesla V100 GPU, etc.), this program supports using NVIDIA tensorrt for fp32 and fp16 precision for inference performance acceleration.You can use parameters such as `--trt_mode=fp16` or `--trt_mode=fp32`



```python
!python infer.py --conf=./freeze_model/deploy.yaml --input_dir=./test_img
```


#### After running this program, the program will scan the input_ All pictures in specified format in dir directory which are generated, and create the results.

### 5. Visualization


```python
%matplotlib inline
import matplotlib.pyplot as plt
img1 = plt.imread('test_img/a2.jpg')
img2 = plt.imread('test_img/a2_jpg_result.png')
img3 = plt.imread('test_img/a3.jpg')
img4 = plt.imread('test_img/a3_jpg_result.png')
img5 = plt.imread('test_img/a4.jpg')
img6 = plt.imread('test_img/a4_jpg_result.png')
fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(img1)
ax.set_title('Origin')
ax.axis('off')
ax = fig.add_subplot(222)
ax.imshow(img2)
ax.set_title('PSPNet')
ax.axis('off')
ax = fig.add_subplot(223)
ax.imshow(img3)
ax.axis('off')
ax = fig.add_subplot(224)
ax.imshow(img4)
ax.axis('off')
plt.show()
```

###  6.Calculating color blocks


```python
from PIL import Image
import numpy as np
from collections import Counter
import pandas as pd

image = Image.open('test_img/a3_jpg_result.png') #the predicted image file
width = image.width
height = image.height
image_list = []
for x in range(height):
    for y in range(width):
        pixel = image.getpixel((y, x))
        image_list.append(pixel)
sky,building,terrain,vegetation = 0,0,0,0
total_pix = len(image_list)

# RBG
for i in range(total_pix):
    if (image_list[i]==(0,128,64)):
        sky += 1
    if (image_list[i]==(0,128,0)):
        building += 1
    if (image_list[i]==(0,0,192)):
        terrain += 1
    if (image_list[i]==(0,0,64)):
        vegetation += 1
sky,building,terrain,vegetation= sky/total_pix*100,building/total_pix*100,terrain/total_pix*100,vegetation/total_pix*100
print('Sky:{}'.format(sky))
print('Building:{}'.format(building))
print('Terrain:{}'.format(terrain))
print('Vegetation:{}'.format(vegetation))
```

