"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import copy

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False

detector = Predictor(config)

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
dist_limit = 6

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

#Generate two text boxes a larger one that covers them
def merge_boxes(box1, box2):
    texttopleft, texttopright, textbottomright, textbottomleft = box1
    objtopleft, objtopright, objbottomright, objbottomleft = box2
    return [[min(texttopleft[0], objtopleft[0]), min(texttopleft[1], objtopleft[1])],
           [max(texttopright[0], objtopright[0]), min(texttopright[1], objtopright[1])],
           [max(textbottomright[0], objbottomright[0]), max(textbottomright[1], objbottomright[1])],
           [min(textbottomleft[0], objbottomleft[0]), max(textbottomleft[1], objbottomleft[1])]]
    # return [min(box1[0], box2[0]), 
    #      min(box1[1], box2[1]), 
    #      max(box1[2], box2[2]),
    #      max(box1[3], box2[3])]
def IOU(box1, box2):

  box1topleft, box1topright, box1bottomright, box1bottomleft = box1
  box2topleft, box2topright, box2bottomright, box2bottomleft = box2

  xA = max(box1topleft[0],box2topleft[0])
  yA = max(box1topleft[1],box2topleft[1])
  xB = min(box1bottomright[0],box2bottomright[0])
  yB = min(box1bottomright[1],box2bottomright[1])
  
  # respective area of ​​the two boxes
  boxAArea=(box1bottomright[0]-box1topleft[0])*(box1bottomright[1]-box1topleft[1])
  boxBArea=(box2bottomright[0]-box2topleft[0])*(box2bottomright[1]-box2topleft[1])
  
  # overlap area
  interArea=max(xB-xA,0)*max(yB-yA,0)
  
  # IOU
  iou = interArea/(boxAArea+boxBArea-interArea)
  #print(iou)
  return iou

#Computer a Matrix similarity of distances of the text and object
def calc_sim(text, obj):
    # text: ymin, xmin, ymax, xmax
    # obj: ymin, xmin, ymax, xmax
    texttopleft, texttopright, textbottomright, textbottomleft = text
    objtopleft, objtopright, objbottomright, objbottomleft = obj
 
    x_dist = min(abs(textbottomright[0]-objbottomleft[0]), abs(textbottomleft[0]-objbottomright[0]))
    y_dist = abs(textbottomleft[1]-objbottomleft[1])
    
    h = (abs(texttopleft[1] - textbottomleft[1]))/3
    #print(h)
    if IOU(text, obj) == 0 and (x_dist > h*100):
        return False

    return y_dist < h

def is_hoz(text):
    texttopleft, texttopright, textbottomright, textbottomleft = text[0]
    w = texttopright[0] - texttopleft[0]
    h = textbottomleft[1] - texttopleft[1]
    ratio = h/w
    if (ratio > 1.5):
       return False
    else:
       return True

def is_hoz1(text):
    texttopleft, texttopright, textbottomright, textbottomleft = text
    w = texttopright[0] - texttopleft[0]
    h = textbottomleft[1] - texttopleft[1]
    ratio = h/w
    if (ratio > 1.5):
       return False
    else:
       return True

#Principal algorithm for merge text 
def merge_algo(texts_boxes):
    for i, (text_box_1) in enumerate(zip(texts_boxes)):
        if (text_box_1 == np.NaN) or (is_hoz(text_box_1) == False):
          continue

        for j, (text_box_2) in enumerate(zip(texts_boxes)):
            if j <= i:
                continue
            if (text_box_2 == np.NaN) or (is_hoz(text_box_2) == False):
                continue

            # Create a new box if a distances is less than disctance limit defined 
            if calc_sim(text_box_1[0], text_box_2[0]) == True:
            # Create a new box  
                new_box = merge_boxes(text_box_1[0], text_box_2[0]) 
                #print(new_box)           
             # Create a new text string 
                #new_text = text_1 + ' ' + text_2

                #texts[i] = new_text
                #delete previous text 
                #del texts[j]
                texts_boxes[i] = new_box
                #delete previous text boxes
                #print(texts_boxes[j])
                #del texts_boxes[j]
                #print(texts_boxes)
                #texts_boxes = texts_boxes[j:j+1]
                texts_boxes = np.delete(texts_boxes, j, 0)
                #texts_boxes[j] = np.NaN
                #print("=======")
                #print(texts_boxes)
                #return a new boxes and new text string that are close
                return True, texts_boxes

    return False, texts_boxes

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        texts_boxes_copied = copy.deepcopy(bboxes)
        need_to_merge = True
        #print(texts_boxes_copied)
        #Merge full text 
        while need_to_merge:
             need_to_merge, texts_boxes_copied = merge_algo(texts_boxes_copied)
        
        #p = []
        #for k in range(len(texts_boxes_copied)):
        #     if texts_boxes_copied[k] != np.NaN: p[k] = texts_boxes_copied[k]
        #texts_boxes_copied = texts_boxes_copied[np.logical_not(np.isnan(texts_boxes_copied))]
        #print(texts_boxes_copied)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], texts_boxes_copied, dirname=result_folder)
        img = Image.fromarray(image)
        width, height = img.size
        for k in range(len(texts_boxes_copied)):
            if (is_hoz1(texts_boxes_copied[k]) == False): continue
            texttopleft, texttopright, textbottomright, textbottomleft = texts_boxes_copied[k]
            top_left_x = min([texttopleft[0],texttopright[0],textbottomright[0],textbottomleft[0]])
            top_left_y = min([texttopleft[1],texttopright[1],textbottomright[1],textbottomleft[1]])
            bot_right_x = max([texttopleft[0],texttopright[0],textbottomright[0],textbottomleft[0]])
            bot_right_y = max([texttopleft[1],texttopright[1],textbottomright[1],textbottomleft[1]])

            area = (max(0, top_left_x - 5), max(0, top_left_y - 5), min(width, bot_right_x + 5), min(height, bot_right_y + 5))
            
            cropped_img = img.crop(area)
            cropped_img.save("tmp_" + str(k) + ".png")
            s = detector.predict(cropped_img)
            print(s)
    print("elapsed time : {}s".format(time.time() - t))
