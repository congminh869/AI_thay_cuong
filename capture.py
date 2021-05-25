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




class DETECT():
    def __init__(self, model_total, device, half):
        self.model_detect, self.stride_detect, self.names_detect = model_total['detect']
        self.model_owner, self.stride_owner, self.names_owner = model_total['owner']
        self.model_serial, self.stride_serial, self.names_serial = model_total['serial']
        self.model_iso, self.stride_iso, self.names_iso = model_total['iso']
        self.model_digit, self.stride_digit, self.names_digit = model_total['digit']
        self.device = device
        self.half = half
        self.string_final = ''
    def catchframe(self, frame):
        if frame is None:
            Print("Image is empty")
            return None

        t2 = time.time()
        frame, string_result = DETECT.detect_obj(self, self.model_detect, self.stride_detect, 
            self.names_detect, img_detect =frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
        t3 = time.time()
        #print('++++++++++++++++++++++++Processing in %.3f'%(t3-t2))
        return frame, string_result
        
    def rmsame(lst):
        for inter in range(len(lst)):
            check_local = []
            for inter1 in range(len(lst)-inter-1):
                if abs(lst[inter][0] - lst[inter1][0]) <2 and abs(lst[inter][1] - lst[inter1][1]) <2:
                    if lst[inter][3]>lst[inter1][3]:
                        lst.remove(lst[inter1])
                    elif lst[inter][3]<lst[inter1][3]:
                        lst.remove(lst[inter])
        return lst
    def detect_obj(self, model, stride, names , img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
        # print(model)
        print('stride = ',stride)
        print('name', names)
        imgsz = img_size #640
        high, weight = img_detect.shape[:2]
        print("high: ",high)
        print('weight',weight)
        #####################################
        classify = False
        agnostic_nms = False
        augment = False
        # Set Dataloader
            #vid_path, vid_writer = None, None
        # Get names and colors

        # Run inference
        t0 = time.time()
        #processing images
        '''
        Tiền xử lí ảnh
        '''
        #from utils.datasets import letterbox
        print('img_detect')
        print(img_detect.shape) #(720, 480, 3)
        print(img_detect)
        im0 = letterbox(img_detect, 640, stride= 32)[0] #(640, 448, 3) convert image to array
        print('im0 = ',im0.shape)
        im0 = im0[:, :, ::-1].transpose(2, 0, 1)
        im0 = np.ascontiguousarray(im0)
        im0 = torch.from_numpy(im0).to(self.device)
        print('im0 = torch.from_numpy(im0).to(self.device)')
        print(im0)
        print(im0.shape)
        im0 = im0.half() if self.half else im0.float()
        print('im0 = im0.half() if self.half else im0.float()')
        print(im0)
        print(im0.shape)
        im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        print('im0 /= 255.0')
        print(im0)
        print(im0.shape)
        if im0.ndimension() == 3:
            im0 = im0.unsqueeze(0)
            print('im0 = im0.unsqueeze(0)')
            print(im0)
            print(im0.shape)
        # Inference
        t1 = time.time()
        pred = model(im0, augment= False)[0]
        print('pred = model(im0, augment= False)[0]')
        print(pred.shape)
        print(pred)
        # Apply NMS
        classes = None
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
        print('pred = non_max_suppression(pred)')
        # print(pred.shape)
        print(pred)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, im0, img_ocr)


        gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
        print('gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]')
        print(gn.shape)#torch.Size([4])
        print(gn)#tensor([480, 720, 480, 720])
        result = dict()
        string = []
        container = dict()
        print('container =',container)
        print('result = ',result)
        print(names)
        for key in names: #['owner', 'serial', 'check', 'ISO']
            container[key] = ''
            result[key] = '*'
        print('container[key] = ',container)
        print('result[key] = ',result)
        print('len(pred[0]) = ',len(pred[0]))
        if len(pred[0]): #4
            pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
            print('pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()')
            print(pred[0][:, :4])
            for c in pred[0][:, -1].unique():
                n = (pred[0][:, -1] == c).sum()  # detections per class
            count = 0
            for box in pred[0]:
                c1 = (int(box[0]), int(box[1]))
                c2 = (int(box[2]), int(box[3]))
                x1, y1 = c1
                x2, y2 = c2
                acc = round(float(box[4])*100,2)
                cls = int(box[5])
                label = names[cls]
                img_crop = img_detect[y1:y2, x1:x2]
                xyxy = [x1, y1, x2, y2]
                image_detect = cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 1)
                result[label] = [xyxy, img_crop]
            # cv2.imshow('window_name', image_detect)
            # cv2.waitKey(0) & 0xff == ord('q')

            
            if container['owner'] != None and result['owner'] != '*':
                container['owner'] = DETECT.detect_ocr(self, self.model_owner, self.stride_owner, 'owner', result['owner'][0], self.names_owner,
                                                img_ocr = result['owner'][1], iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
            if container['serial'] != None and result['serial'] != '*':
                container['serial'] = DETECT.detect_ocr(self, self.model_serial, self.stride_serial, 'serial', result['serial'][0], self.names_serial,
                                                img_ocr = result['serial'][1], iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
            if container['ISO'] != None and result['ISO'] != '*':
                container['ISO'] = DETECT.detect_ocr(self, self.model_iso, self.stride_iso, 'ISO', result['ISO'][0], self.names_iso,
                                                img_ocr = result['ISO'][1], iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
            if container['check'] != None and result['check'] != '*':
                container['check'] = DETECT.detect_ocr(self, self.model_digit, self.stride_digit, 'check', result['check'][0], self.names_digit,
                                                img_ocr = result['check'][1], iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
            self.string_final = container['owner'] +' '+ container['serial'] +' '+ container['check'] +' '+ container['ISO'] 
            #image_detect = cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 1)
            #cv2.putText(image_detect, label, c1, cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
            #print(str(label)+'      '+''.join(string))
            #print(string_result)
            cv2.putText(img_detect, self.string_final , (0,high-3), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
        return img_detect, self.string_final
    def detect_ocr(self, model, stride, labels, xyxy, names1, img_ocr = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
        h_1 = int(xyxy[3])-int(xyxy[1])
        w_1 = int(xyxy[2])-int(xyxy[0])
        labels_1 = labels
        imgsz_1 = img_size
        classify_1 = False
        agnostic_nms_1 = False
        augment_1 = False
        names1_1 = names1
        # Set Dataloader
        #vid_path, vid_writer = None, None
        # Get names and colors

        # Run inference
        t0_1 = time.time()
        #processing images
        '''
        Tiền xử lí ảnh
        '''
        im0_1 = letterbox(img_ocr, 640, 32)[0]
        im0_1 = im0_1[:, :, ::-1].transpose(2, 0, 1)
        im0_1 = np.ascontiguousarray(im0_1)
    
        #####################################
        im0_1 = torch.from_numpy(im0_1).to(self.device)
        im0_1 = im0_1.half() if self.half else im0_1.float()
        im0_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im0_1.ndimension() == 3:
            im0_1 = im0_1.unsqueeze(0)
        # Inference
        t1_1 = time.time()
        pred_1 = model(im0_1, augment= augment_1)[0]
        # Apply NMS
        classes = None
        pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms_1)
    
        # Apply Classifier
        if classify_1:
            pred_1 = apply_classifier(pred_1, modelc, im0_1, img_ocr)
        gn_1 = torch.tensor(img_ocr.shape)[[1, 0, 1, 0]]# normalization gain whwh
        result_1 = []
        string_1 = []
        if len(pred_1[0]):
            pred_1[0][:, :4] = scale_coords(im0_1.shape[2:], pred_1[0][:, :4], img_ocr.shape).round()
            for c_1 in pred_1[0][:, -1].unique():
                n_1 = (pred_1[0][:, -1] == c_1).sum()  # detections per class
            count_1 = 0
            for box_1 in pred_1[0]:
                c1_1 = (int(box_1[0]), int(box_1[1]))
                c2_1 = (int(box_1[2]), int(box_1[3]))
                acc_1 = round(float(box_1[4])*100,2)
                cls_1 = int(box_1[5])
                label_1 = names1[cls_1]
                #image_ocr = cv2.rectangle(img_ocr, c1, c2, (255, 0, 0), 1)
                if acc_1 >0.7:
                    result_1.append([int(box_1[0]), int(box_1[1]), label_1, acc_1])
                count_1 += 1
            result_1=DETECT.rmsame(result_1)
            if h_1 < w_1 :
                for m in range(len(result_1)):
                    for n in range(len(result_1)-m-1):
                        if result_1[m][0] > result_1[n+m+1][0]:
                            middle = result_1[m]
                            result_1[m] = result_1[n+m+1]
                            result_1[n+m+1] = middle
                [string_1.append(lb[2]) for lb in result_1]
            else:
                for m in range(len(result_1)):
                    for n in range(len(result_1)-m-1):
                        if result_1[m][1] > result_1[n+m+1][1]:
                            middle = result_1[m]
                            result_1[m] = result_1[n+m+1]
                            result_1[n+m+1] = middle
                [string_1.append(lb[2]) for lb in result_1]
            t2_1 = time.time()
            #print('OCR in %f s'%(t2_1-t1_1))
            #print(str(labels_1)+'      '+''.join(string_1))
        return ''.join(string_1)
# if __name__ == '__main__':
#     check_requirements(exclude=('pycocotools', 'thop'))
#     #Khai báo
#     img_size = 640
#     conf_thres = 0.25
#     iou_thres = 0.45
#     device = ''
#     update = True
#     model_total = dict()
#     # Load model
#     model_total =  dict()
#     t1_loadmodel = time.time()
#     set_logging()
#     device = select_device(device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#     # Load model nhan dien container
#     weights = './weights/container.pt'
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     names = model.module.names if hasattr(model, 'module') else model.names
#     # Load model nhan dien owner
#     weights1 = './weights/owner.pt'
#     model1 = attempt_load(weights1, map_location=device)  # load FP32 model
#     stride1 = int(model1.stride.max())  # model stride
#     names1 = model1.module.names if hasattr(model1, 'module') else model1.names
#     # Load model nhan dien serial
#     weights2 = './weights/serial.pt'
#     model2 = attempt_load(weights2, map_location=device)  # load FP32 model
#     stride2 = int(model2.stride.max())  # model stride
#     names2 = model2.module.names if hasattr(model2, 'module') else model2.names
#     # Load model nhan dien ISO
#     weights3 = './weights/iso.pt'
#     model3 = attempt_load(weights3, map_location=device)  # load FP32 model
#     stride3 = int(model3.stride.max())  # model stride
#     names3 = model3.module.names if hasattr(model3, 'module') else model3.names
#     # Load model nhan dien check
#     weights4 = './weights/check.pt'
#     model4 = attempt_load(weights4, map_location=device)  # load FP32 model
#     stride4 = int(model4.stride.max())  # model stride
#     names4 = model4.module.names if hasattr(model4, 'module') else model4.names
#     if half:
#         model.half()
#         model1.half()
#         model2.half()
#         model3.half()
#         model4.half()# to FP16
#     t2_loadmodel = time.time()
#     model_total['detect'] = (model, stride, names)
#     model_total['owner'] = (model1, stride1, names1)
#     model_total['serial'] = (model2, stride2, names2)
#     model_total['iso'] = (model3, stride4, names3)
#     model_total['digit'] = (model4, stride4, names4)
#     print('Loaded model in %f s'%(t2_loadmodel-t1_loadmodel))
#     detect = DETECT(model_total)
#     img = detect.catchframe()
