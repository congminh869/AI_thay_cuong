from recognition import E2E
import cv2
from pathlib import Path
import argparse
import time
from glob 						import glob
from os.path 					import splitext, basename
from src.keras_utils import load_model
from lpdetection import detect_license
from ocr import detect_ocr_1, detect_ocr_2
import numpy as np
from src.drawing_utils import draw_losangle
import darknet.python.darknet as dn
from skimage.filters import threshold_local
import torch
import torch.backends.cudnn as cudnn
from numpy import random




def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./images/1001020210401003012460_001845907101_full.jpg')

    return arg.parse_args()
cv2.startWindowThread()

wpod_net_path = "weights/lp-detector/wpod-net_update1.h5"
ocr_weights = bytes('data/ocr/ocr-net.weights', encoding="utf-8")
ocr_netcfg  = bytes('data/ocr/ocr-net.cfg', encoding="utf-8")
ocr_dataset = bytes('data/ocr/ocr-net.data', encoding="utf-8")

ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)
wpod_net = load_model(wpod_net_path)

args = get_arguments()

cap = cv2.VideoCapture('rtsp://admin:MQ123456@192.168.6.128:554')
while(True):
	ret, frame = cap.read()
	#img_path = Path(args.image_path)
	# read image
	start1 = time.time()
	#frame = cv2.imread(str(img_path))

	# load model
	#model = E2E()
	lp_image,llp, scale = detect_license(frame, wpod_net)
	check = True
	if lp_image is not None:
		#img_code = CONTAINER()
		#frame = detect_obj(model, stride, names, model1, stride1, names1, img_detect = img_code, iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
		#cv2.imwrite('%d.jpg'%t, frame)
		#end
		lp_image = lp_image[:,5:-5]
		pts = llp[0].pts * (np.array([frame.shape[1],frame.shape[0]]).reshape(2,1))
		draw_losangle(frame, pts, (0,  0,255), 3)
		#cv2.imshow('License Plate', frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
		    exit(0)
		# recognize license plate
		#lp_str = model.predict(lp_image)
		#print("lp_str:" + lp_str)
		# V = cv2.split(cv2.cvtColor(lp_image, cv2.COLOR_BGR2HSV))[2]

		# # adaptive threshold
		# T = threshold_local(V, 35, offset=10, method="gaussian")
		# thresh = (V > T).astype("uint8") * 255
		# backtorgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
		# cv2.imwrite("thresh.png", backtorgb)
		#print(scale)

		if (scale > 0):
	    		print("1 line plate")
	    		lp_str1 = detect_ocr_1(ocr_net, ocr_meta, lp_image)
		else:
	    		print("2 line plate")
	    		lp_str1 = detect_ocr_2(ocr_net, ocr_meta, lp_image)
		if (len(lp_str1) < 8):
			print("Not correct LP")
			continue

		print("lp_str1:" + lp_str1)

		# end
		end = time.time()

		print('Model process on %.2f s' % (end - start1))
		font = cv2.FONT_HERSHEY_PLAIN
		if lp_str1:
	    		pts = llp[0].pts * (np.array([frame.shape[1],frame.shape[0]]).reshape(2,1))
	    		draw_losangle(frame, pts, (0,  0,255), 3)
	    		cv2.putText(frame, lp_str1, (int(pts[0][0]),int(pts[1][0])-10), font, 1, (0, 255, 0),1 , cv2.LINE_AA)

		else:
	    		pass

		# show image
		#cv2.imwrite('./out/' + base_n, image)

	else:
		continue
		# cv2.imshow('frame',frame)
		# if cv2.waitKey(25) & 0xFF == ord('q'):
		# 	break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

