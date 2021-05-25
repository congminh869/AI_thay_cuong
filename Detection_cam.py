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
def detect_ocr(model, stride, labels, xyxy, names1, img_ocr = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
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
    im0_1 = torch.from_numpy(im0_1).to(device)
    im0_1 = im0_1.half() if half else im0_1.float()
    im0_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im0_1.ndimension() == 3:
        im0_1 = im0_1.unsqueeze(0)
    # Inference
    t1_1 = time.time()
    pred_1 = model(im0_1, augment= augment_1)[0]
    # Apply NMS
    classes = None
    pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres,
                               classes = classes, agnostic=agnostic_nms_1)
    
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
        result_1=rmsame(result_1)
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





def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
    imgsz = img_size
    high, weight = img_detect.shape[:2]
    #if half:
        #model.half()
    classify = False
    agnostic_nms = False
    augment = False
    names = names
    # Set Dataloader
    #vid_path, vid_writer = None, None
    # Get names and colors

    # Run inference
    t0 = time.time()
    #processing images
    '''
    Tiền xử lí ảnh
    '''
    im0 = letterbox(img_detect, 640, stride= 32)[0]
    im0 = im0[:, :, ::-1].transpose(2, 0, 1)
    im0 = np.ascontiguousarray(im0)
    
    #####################################
    im0 = torch.from_numpy(im0).to(device)
    im0 = im0.half() if half else im0.float()
    im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im0.ndimension() == 3:
        im0 = im0.unsqueeze(0)
    # Inference
    t1 = time.time()
    pred = model(im0, augment= augment)[0]
    # Apply NMS
    classes = None
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
    
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, im0, img_ocr)
    gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
    result = []
    string = []
    container = dict()
    for key in names:
        container[key] = ''
    if len(pred[0]):
        pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
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
            if label != 'check':
                string = detect_ocr(model1, stride, label, xyxy, names1, img_ocr = img_crop, iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
                container[label] = string
                string_result = container['owner'] +' '+ container['serial'] +' '+ container['ISO']
                #image_detect = cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 1)
                #cv2.putText(image_detect, label, c1, cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 0, 0), 1)
                count += 1
                print(str(label)+'      '+''.join(string))
            cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 1)
        cv2.putText(img_detect, string_result , (0,high-3), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
    return img_detect
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='container.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            t1_loadmodel = time.time()
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA
            # Load model
            weights = opt.weights
            char_weights = 'last.pt'
            #im_path = glob.glob('./test/*.jpg')
            model = attempt_load(opt.weights, map_location=device)  # load FP32 model
            model1 = attempt_load(char_weights, map_location=device)
            stride = int(model.stride.max())  # model stride
            stride1 = int(model1.stride.max())
            imgsz = check_img_size(opt.img_size, s = stride)  # check img_size
            names = model.module.names if hasattr(model, 'module') else model.names
            names1 = model1.module.names if hasattr(model1, 'module') else model1.names
            t2_loadmodel = time.time()
            if half:
                model.half()  # to FP16
            if half:
                model1.half()  # to FP16
            print('Loaded model in %f s'%(t2_loadmodel-t1_loadmodel))

            cap = cv2.VideoCapture('rtsp://admin:123@123a@192.168.6.32:554/cam/realmonitor?channel=1&subtype=0')
            fps = cap.get(cv2.CAP_PROP_FPS)
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                # Display the resulting frame
                t2 = time.time()
                frame = detect_obj(model, stride, names, img_detect =frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 640)
                t3 = time.time()
                print('Processing in %.3f'%(t3-t2))
                cv2.imshow('frame', frame)
                #cv2.imshow('frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

