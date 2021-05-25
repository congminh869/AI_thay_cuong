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
import datetime
import pycurl
from urllib.parse import urlencode
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
import sys	
from collections import Counter
from io import BytesIO
import json

check_requirements(exclude=('pycocotools', 'thop'))

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

def sendAPI_MQ(loai,lp, con1, con2, img_path):
    img_path_ret = ''

    url_img = 'http://crm.mqsolutions.vn:8880/api/image'
    url_event = 'http://crm.mqsolutions.vn:8880/api/transfer_history'
    
    #Send image       
    curl = pycurl.Curl()
    curl.setopt(pycurl.POST, 1)
    curl.setopt(curl.URL, url_img)
    buffer = BytesIO()
    filename = lp + ".jpg"
    curl.setopt(pycurl.WRITEFUNCTION, buffer.write)
    print("filename = ", filename)
    print("img_path = ", img_path)
    curl.setopt(curl.HTTPPOST,[("filename", filename),("file", (curl.FORM_FILE, img_path))])
    curl.perform()
    status_code = curl.getinfo(pycurl.RESPONSE_CODE)
    if status_code == 200:
        resp = buffer.getvalue().decode('utf8')
        json_data = json.loads(resp)
        if json_data['file_path'] is not None:
            img_path_ret = json_data['file_path']
    else:
        return -1
        
    #Send event
    data = {
            "license_plate": "",
            "container_ids": ["", ""], 
            "images": ["", ""], 
            "status": False
       }
    data['license_plate'] = lp
    data['container_ids'] = [con1, con2]
    data['images'] = [img_path_ret, ""]
    if loai == '1':
        data['status'] = False
    elif loai == '2':
        data['status'] = True
        
    data_json = json.dumps(data)
    
    curl.setopt(curl.URL, url_event) 
    buffer = BytesIO()
    curl.setopt(pycurl.HTTPHEADER, ['Accept: application/json',
                                'Content-Type: application/json'])
    curl.setopt(pycurl.WRITEFUNCTION, buffer.write)
    curl.setopt(pycurl.POSTFIELDS, data_json)       
    curl.perform()
    status_code = curl.getinfo(pycurl.RESPONSE_CODE)
    if status_code == 200:
        return 1
    else:
        return 0


def getInfoContainer(data):
	"""Get information of container from Cloud Server, input is @<dict>
	{'loai' : '<loai>',
	'maKhoBai' : '<maKhoBai>',
	'ngayGetin' : '<date>',
	'soContainer' : '<code_container>'}

	return @<int>(Status code), @<string>(Json data)
	"""
	API_URL = 'https://gps.cs.etc.vn:15443/etcaccr-ecargo-api/swagger-resources/request-object'
	url_data = urlencode(data)
	url = API_URL + "?" +  url_data

	curl = pycurl.Curl()
	curl.setopt(curl.SSL_VERIFYPEER, 0)
	curl.setopt(pycurl.URL, url)
	curl.setopt(pycurl.HTTPHEADER, ['Accept: application/json',
	                                'Content-Type: application/json'])

	buffer = BytesIO()

	# prepare and send. See also: pycurl.READFUNCTION to pass function instead
	curl.setopt(pycurl.WRITEFUNCTION, buffer.write)
	curl.perform()

	status_code = curl.getinfo(pycurl.RESPONSE_CODE)

	return status_code, buffer.getvalue().decode('utf8')

def checkInfoContainer(loai, num, con1, con2 = ''):
	x = datetime.datetime.now()
	date = str(x.year) + '-' + str(x.month) + '-' + str(x.day)
	ret = 0
	if con1 == '':
		ret = -1
		return ret
	if num == 1 :
		data = {'loai' : '1',
		    'maKhoBai' : 'VNCXP',
		    'ngayGetin' : '',
		    'soContainer' : ''}
		data['loai'] = loai
		data['ngayGetin'] = date
		data['soContainer'] = con1

		status_code, req = getInfoContainer(data)
		if status_code == 200 :
			json_data = json.loads(req)
			if json_data["dsToKhai"] is None:
				ret = 0
			else:
				ret = 1

	elif num == 2 :
		data1 = {'loai' : '1', 'maKhoBai' : 'VNCXP', 'ngayGetin' : '', 'soContainer' : ''}
		data1['loai'] = loai
		data1['ngayGetin'] = date
		data1['soContainer'] = con1

		data2 = {'loai' : '1', 'maKhoBai' : 'VNCXP', 'ngayGetin' : '', 'soContainer' : ''}
		data2['loai'] = loai
		data2['ngayGetin'] = date
		data2['soContainer'] = con2

		status_code, req = getInfoContainer(data1)
		if status_code == 200 :
			json_data = json.loads(req)
			if json_data["dsToKhai"] is not None:
				ret = ret +1

		status_code, req = getInfoContainer(data2)
		if status_code == 200 :
			json_data = json.loads(req)
			if json_data["dsToKhai"] is not None:
				ret = ret +1
	return ret			


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
mutex = threading.Lock()
mutex_alldone = threading.Lock()

def ContainerMain(name,source, out, event):
	global flag_all_done
	cap = cv2.VideoCapture(source)
	while True:
		event.wait()
		#event.clear()
		result = []
		t0 = time.time()
		while (time.time() - t0  < 6):
			frame = cv2.imread('4.jpg')
			#ret, frame = cap.read()
			if frame is not None:
				#try:
				mutex.acquire()
				img2, string_result = detect.catchframe(frame)
				#cv2.imwrite(out, img2)
				result.append(string_result)
				mutex.release()
				#except:
					#print("Oops!", sys.exc_info()[0], "occurred.")
					#print("Next entry.")
					#continue

		if len(result)!=0:
			dic = dict(Counter(result))
			max_key = max(dic, key=dic.get)
			#print("============" + name +"==================")
			#print(max_key)
			mutex_alldone.acquire()
			flag_all_done = flag_all_done + 1
			id = int(name[-1])
			list_container_codes[id] = max_key
			mutex_alldone.release()
			#print("=========================================")
			#cv2.imshow('CON',img2)
			#if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	break

def LicensePlateMain(event):
	global flag_all_done
	global loai
	t0 = time.time()
	old_lp_str1 = ''
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
			#print("!!!!!!!!!!!!!!!!scale:"+str(scale))

			#print("==============================================================")
			num_con = 1
			code_con_1 = ''
			code_con_2 = ''

			if flag_all_done == 4:
				list_code = ['','','','']
				num_check_40 = 0
				num_check_20 = 0
				id_get = 0
				for id, code in list_container_codes.items():
					regex1 = re.findall('2\d[A-Z]\d+', code)
					regex2 = re.findall('4\d[A-Z]\d+', code)
					if len(regex2) != 0:
						id_get = id - 1
						num_check_40 = num_check_40 + 1
					elif len(regex1) != 0:
						id_get = id - 1
						num_check_20 = num_check_20 + 1
					print("ContainerThread %d"%id, " --- ", code)
					if code.find("*") >= 0:
						list_container_codes[id] = ''
					if id == 1:
						list_code[0] = code
					elif id == 2:
						list_code[1] = code
					elif id == 3:
						list_code[2] = code
					elif id == 4:
						list_code[3] = code

				#Case container is 40ft
				if num_check_40 >= 2 :
					num_con = 1
					code_con_1 = list_code[id_get]
				else: 
					if num_check_20 == 1 : #Case 1 container is 20ft
						num_con = 1
						code_con_1 = list_code[id_get]
					if num_check_20 == 2 : 
						if list_code[0] == list_code[1]: #Case 1 container is 20ft ahead
							num_con = 1
							code_con_1 = list_code[0]
						elif list_code[2] == list_code[3]: #Case 1 container is 20ft behind
							num_con = 1
							code_con_1 = list_code[2]
						elif list_code[1] == list_code[3] or list_code[0] == list_code[3] or list_code[0] == list_code[2] or list_code[1] == list_code[2]: #Case 2 container is 20ft
							num_con = 2
							if list_code[0] != '':
								code_con_1 = list_code[0]
							else:
								code_con_1 = list_code[1]

							if list_code[2] != '':
								code_con_2 = list_code[2]
							else:
								code_con_2 = list_code[3]
					if num_check_20 >= 3 :  #Case 2 container is 20ft
						num_con = 2
						if list_code[1] == list_code[0]:
							code_con_1 = list_code[0]
							if list_code[2] != '':
								code_con_2 = list_code[2]
							else:
								code_con_2 = list_code[3]

						if list_code[2] == list_code[3]:
							code_con_2 = list_code[2]
							if list_code[0] != '':
								code_con_1 = list_code[0]
							else:
								code_con_1 = list_code[1]

				#### CHECK INFO CONTAINER #####
				ret_mq = sendAPI_MQ(loai, old_lp_str1, code_con_1, code_con_2, "./out/lp.jpg")
				if ret_mq == 1:
					print("Sent Event MQServer!")
				else:
					print("Fail Sent Event MQServer!")
				ret = checkInfoContainer(loai, num_con, code_con_1, code_con_2)
				if ret != -1 :
					if ret < num_con:
						print("==============================")
						print("CONTAINER KHONG DUOC XAC THUC")
						print("==============================")
					elif ret == num_con:
						print("==============================")
						print("CONTAINER DA DUOC XAC THUC")
						print("==============================")
			#print("==============================================================")

			if (scale > 1.5):
				#print("1 line plate")
				lp_str1 = detect_ocr_1(ocr_net, ocr_meta, lp_image)
			else:
				#print("2 line plate")
				lp_str1 = detect_ocr_2(ocr_net, ocr_meta, lp_image)

			if lp_str1 != old_lp_str1:
				old_lp_str1 = lp_str1
				flag_all_done = 0
			else:
				mutex_alldone.acquire()
				#print(flag_all_done, " ------ " , time.time() - t0)

				if flag_all_done == 4:
					if (time.time() - t0)  < 30:
						mutex_alldone.release()
						time.sleep(1)
						continue
					else:
						flag_all_done = 0
						mutex_alldone.release()
				else:
					mutex_alldone.release()


			if (len(lp_str1) > 6):

				#check license plate format
				pattern = '^\d\d[A-Z]\d+'
				result = re.findall(pattern, lp_str1) 
				#print(result)
				if (len(result) > 0):
					#print("lp_str1:" + result[0])

					# end
					end = time.time()
					#lp_str1 = ""
					#print('Model process on %.2f s' % (end - start))
					font = cv2.FONT_HERSHEY_PLAIN
					if lp_str1:
							t0 = time.time()
							event.set()
							event.clear()
							pts = llp[0].pts * (np.array([frame.shape[1],frame.shape[0]]).reshape(2,1))
							draw_losangle(frame, pts, (0,  0,255), 3)
							cv2.putText(frame, lp_str1, (int(pts[0][0]),int(pts[1][0])-10), font, 1, (0, 255, 0),1 , cv2.LINE_AA)
							time_save = datetime.datetime.now()
							date_save = 'H' + str(time_save.hour) + '_D'+ str(time_save.day) + '_M' + str(time_save.month) + '_Y' + str(time_save.year)
							base_n = lp_str1 +'_'+ date_save +'.jpg'
							#cv2.imwrite('./out/' + base_n, frame)
							cv2.imwrite('./out/lp.jpg', frame)

					else:
				    		pass

			# show image
		#else:
			#pass
		cv2.imshow('LP',frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


class ContainerThread (threading.Thread):
   def __init__(self, name,  source , out, event):
      threading.Thread.__init__(self)
      self.source = source
      self.name = name
      self.out = out
      self.event = event
   def run(self):
      print ("Starting " + self.name)
      ContainerMain(self.name, self.source, self.out, self.event)
      print ("Exiting " + self.name)
# Create new threads
event = threading.Event()
thread = threading.Thread(target= LicensePlateMain, args=(event, ))
thread1= ContainerThread("ContainerThread1", 'rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0', "out1.jpg", event)
thread2 = ContainerThread("ContainerThread2", 'rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0', "out2.jpg",event)
thread3 = ContainerThread("ContainerThread3", 'rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0', "out3.jpg",event)
thread4 = ContainerThread("ContainerThread4",'rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0', "out4.jpg",event)
# Start new Threads
thread.start()
thread1.start()
thread2.start()
thread3.start()
thread4.start()