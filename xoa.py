from collections import Counter
import cv2
import time
from capture import DETECT
import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
import glob
import os

#Khai báo
loai = '1'
flag_all_done = 0
list_container_codes = dict()
img_size = 640
conf_thres = 0.25
iou_thres = 0.45
device = ''
update = True
model_total = dict()
# Load model
model_total =  dict()
t1_loadmodel = time.time()
set_logging()
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model nhan dien container
weights = './weights/container.pt'
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names
# Load model nhan dien owner
weights1 = './weights/owner.pt'
model1 = attempt_load(weights1, map_location=device)  # load FP32 model
stride1 = int(model1.stride.max())  # model stride
names1 = model1.module.names if hasattr(model1, 'module') else model1.names
# Load model nhan dien serial
weights2 = './weights/serial.pt'
model2 = attempt_load(weights2, map_location=device)  # load FP32 model
stride2 = int(model2.stride.max())  # model stride
names2 = model2.module.names if hasattr(model2, 'module') else model2.names
# Load model nhan dien ISO
weights3 = './weights/iso.pt'
model3 = attempt_load(weights3, map_location=device)  # load FP32 model
stride3 = int(model3.stride.max())  # model stride
names3 = model3.module.names if hasattr(model3, 'module') else model3.names
# Load model nhan dien check
weights4 = './weights/check.pt'
model4 = attempt_load(weights4, map_location=device)  # load FP32 model
stride4 = int(model4.stride.max())  # model stride
names4 = model4.module.names if hasattr(model4, 'module') else model4.names
if half:
    model.half()
    model1.half()
    model2.half()
    model3.half()
    model4.half()# to FP16
t2_loadmodel = time.time()
model_total['detect'] = (model, stride, names)
model_total['owner'] = (model1, stride1, names1)
model_total['serial'] = (model2, stride2, names2)
model_total['iso'] = (model3, stride4, names3)
model_total['digit'] = (model4, stride4, names4)
print('Loaded model in %f s'%(t2_loadmodel-t1_loadmodel))
detect = DETECT(model_total, device, half)

def ContainerMain(name='Container2',source=0, out='test5.jpg'):
    global flag_all_done
    i = 0
    cap = cv2.VideoCapture(source)
    result = []
    t0 = time.time()
    # while (time.time() - t0  < 6):
    frame = cv2.imread('4.jpg')
    if frame is not None:
        i=i+1
        print('frame is not None '+str(i)+ ' ' +str(time.time() - t0))
        img2, string_result = detect.catchframe(frame)
        cv2.imwrite(out, img2)
        result.append(string_result)
    if len(result)!=0:
        dic = dict(Counter(result))
        max_key = max(dic, key=dic.get)
        print("============" + name +"==================")
        print(max_key)
        flag_all_done = flag_all_done + 1
        print('*')
        id = int(name[-1])
        print('**')
        list_container_codes[id] = max_key
        
            
ContainerMain()