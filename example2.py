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
from capture import DETECT
import time
import threading
import re
import cv2
from pathlib import Path
import argparse

from glob 						import glob
from os.path 					import splitext, basename
from src.keras_utils import load_model
from lpdetection import detect_license
from ocr import detect_ocr_1, detect_ocr_2
import numpy as np
from src.drawing_utils import draw_losangle
import darknet.python.darknet as dn
from skimage.filters import threshold_local
import os

check_requirements(exclude=('pycocotools', 'thop'))
#Khai báo
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




def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./images/1.jpg')

    return arg.parse_args()
cv2.startWindowThread()

wpod_net_path = "weights/lp-detector/wpod-net_update1.h5"
ocr_weights = bytes('data/ocr/ocr-net.backup', encoding="utf-8")
ocr_netcfg  = bytes('data/ocr/ocr-net.cfg', encoding="utf-8")
ocr_dataset = bytes('data/ocr/ocr-net.data', encoding="utf-8")

ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)
wpod_net = load_model(wpod_net_path)
cap = cv2.VideoCapture('rtsp://admin:MQ123456@192.168.6.128:554')

args = get_arguments()
# License Plate Camera
# Container Camera

print("====== START ======")

def ContainerMain(source, out):

	cap = cv2.VideoCapture(source)
	
	while(True):
		ret, frame = cap.read()
		if frame is not None:
			img2 = detect.catchframe(frame)
			cv2.imwrite(out, img2)
			# cv2.imshow('CON',img2)
			# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	break

def LicensePlateMain():

	
	while(True):
		ret, frame = cap.read()
		# start
		start = time.time()

		# load model
		lp_image,llp, scale = detect_license(frame, wpod_net)
		if lp_image is not None:
			# lp_image = lp_image[:,5:-5]
			pts = llp[0].pts * (np.array([frame.shape[1],frame.shape[0]]).reshape(2,1))
			draw_losangle(frame, pts, (0,  0,255), 3)
			# cv2.imshow('License Plate', frame)
			# if cv2.waitKey(25) & 0xFF == ord('q'):
			#     exit(0)
			print("!!!!!!!!!!!!!!!!scale:"+str(scale))

			if (scale > 1.5):
		    		print("1 line plate")
		    		lp_str1 = detect_ocr_1(ocr_net, ocr_meta, lp_image)
			else:
		    		print("2 line plate")
		    		lp_str1 = detect_ocr_2(ocr_net, ocr_meta, lp_image)
			if (len(lp_str1) > 6):

				#check license plate format
				pattern = '^\d\d[A-Z]\d+'
				result = re.findall(pattern, lp_str1) 
				#print(result)
				if (len(result) > 0):
					print("lp_str1:" + result[0])

					# end
					end = time.time()
					#lp_str1 = ""
					print('Model process on %.2f s' % (end - start))
					font = cv2.FONT_HERSHEY_PLAIN
					if lp_str1:
							
							pts = llp[0].pts * (np.array([frame.shape[1],frame.shape[0]]).reshape(2,1))
							draw_losangle(frame, pts, (0,  0,255), 3)
							cv2.putText(frame, lp_str1, (int(pts[0][0]),int(pts[1][0])-10), font, 1, (0, 255, 0),1 , cv2.LINE_AA)

					else:
				    		pass

			# show image
			#cv2.imwrite('./out/' + base_n, image)

		#else:
			#pass
		cv2.imshow('LP',frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


class ContainerThread (threading.Thread):
   def __init__(self, name,  source , out):
      threading.Thread.__init__(self)
      self.source = source
      self.name = name
      self.out = out
   def run(self):
      print ("Starting " + self.name)
      ContainerMain(self.source, self.out)
      print ("Exiting " + self.name)

# Create new threads
thread1= ContainerThread("ContainerThread1", 'rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0', "out1.jpg")
thread2 = ContainerThread("ContainerThread1", 'rtsp://admin:MQ123456@192.168.6.128:554', "out2.jpg")

# Start new Threads
thread1.start()
thread2.start()

while 1:
	LicensePlateMain()



