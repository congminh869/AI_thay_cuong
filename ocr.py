import sys
import cv2
import numpy as np
import traceback
import numpy
import darknet.python.darknet as dn

from os.path     import splitext, basename
from glob    import glob
from darknet.python.darknet import detect, detect_np
from src.label    import dknet_label_conversion
from src.utils     import nms
from skimage.filters import threshold_local
import imutils
from data_utils import order_points, convert2Square, draw_labels_and_boxes
from model import CNN_Model
import time
from PIL import Image

# recogChar = CNN_Model(trainable=False).model
# recogChar.load_weights('./weights/weight.h5')
    
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# #config = Cfg.load_config_from_name('vgg_transformer')
# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
# config['cnn']['pretrained']=False
# #config['device'] = 'cuda:0'
# config['device'] = 'cpu'
# config['predictor']['beamsearch']=False

# detector = Predictor(config)

# ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
#               13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
#               25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


def check_outlier_character(l):

    width = l.br()[0]-l.tl()[0]
    height = l.br()[1]-l.tl()[1]
    # print(height)


    return height<1


def preprocess(img,alpha,beta):

    # img_all = img.copy()

    heigth, width = img.shape[:2]
    # imgP1 = cv2.imread("part1/part1.png")
    imgP1 =img[:int(heigth / 2),]
    # # imgP1 = cv2.resize(img[:int(heigth / 2),], (200,80), interpolation=cv2.INTER_AREA)
    #imgP1 = imgP1[:,20:-20]
    # bg1 = np.zeros([94,288,3])
    # bg1.fill(255)
    imgP1 = cv2.convertScaleAbs(imgP1, alpha=alpha, beta=beta)
    # bg1[7:-7,34:-34] =imgP1
    # imgP1 = cv2.medianBlur(imgP1, 3)
    # imgP1 = cv2.GaussianBlur(imgP1, (3, 3), 0)
    imgP2 = img[int(heigth / 2):,]
    # imgP2 = cv2.resize(img[int(heigth / 2):,], (200,80), interpolation=cv2.INTER_AREA)
    #imgP2 = imgP2[:,10:-10]


    imgP2 = cv2.convertScaleAbs(imgP2, alpha=alpha, beta=beta)
    # imgP2 = cv2.medianBlur(imgP2, 5)
    # imgP2 = cv2.GaussianBlur(imgP2, (3, 3), 0)


    # print("imggg", img)
    # cv2.imwrite('part1.png', imgP1)
    # cv2.imwrite('part2.png', imgP2)
    # cv2.imwrite('part.png', img)
    return imgP1,imgP2


# def detect_ocr_2(ocr_net, ocr_meta, img):

    
#     ocr_threshold = .4

#     print ('Performing OCR 2...')
#     # image_path = 'out_lp.png'
#     recogChar = CNN_Model(trainable=False).model
#     recogChar.load_weights('./weights/weight.h5')


#     List_R1 = []
#     List_R2 = []
    
#     imgP1,imgP2=preprocess(img, 1.45, -50)

#     ht, wd, cc= imgP1.shape
#     ww = 240
#     hh = 80
#     color = (255,255,255)
#     result1 = np.full((hh,ww,cc), color, dtype=np.uint8)

#     # compute center offset
#     xx = (ww - wd) // 2
#     yy = (hh - ht) // 2

#     # copy img image into center of result image
#     result1[yy:yy+ht, xx:xx+wd] = imgP1

#     cv2.imwrite("p1.png", result1)

#     # imgP1 = cv2.imread("part1/part1.png")
#     R1, (width1, height1) = detect_np(ocr_net, ocr_meta, result1, thresh=ocr_threshold,nms=None)


#     ht, wd, cc= imgP2.shape
#     result2 = np.full((hh,ww,cc), color, dtype=np.uint8)

#     # compute center offset
#     xx = (ww - wd) // 2
#     yy = (hh - ht) // 2

#     # copy img image into center of result image
#     result2[yy:yy+ht, xx:xx+wd] = imgP2
#     cv2.imwrite("p2.png", result2)

#     img = Image.open("p1.png")
#     lp_str1 = detector.predict(img)

#     img = Image.open("p2.png")
#     lp_str2 = detector.predict(img)

#     lp_str = lp_str1 + "-" + lp_str2

#     print('\t\tLP: %s' % lp_str)

#     return lp_str


# def detect_ocr_1(ocr_net, ocr_meta, img):

#     ocr_threshold = .4

#     print ('Performing OCR 1...')

#     img = cv2.resize(img, (240, 80), interpolation = cv2.INTER_AREA)
#     #t1 = int(time.time())
#     t1 = int(time.time())
#     image_path = './lp/lp_%d.jpg'%t1
    
#     cv2.imwrite(image_path, img)

#     img = Image.open(image_path)
#     lp_str1 = detector.predict(img)
    
#     return lp_str1


def detect_ocr_2(ocr_net, ocr_meta, img):

    
    ocr_threshold = .4

    #print ('Performing OCR 2...')
    # image_path = 'out_lp.png'


    List_R1 = []
    List_R2 = []
    
    imgP1,imgP2=preprocess(img, 1.45, -50)

    ht, wd, cc= imgP1.shape
    ww = 240
    hh = 80
    color = (255,255,255)
    result1 = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result1[yy:yy+ht, xx:xx+wd] = imgP1

    cv2.imwrite("p1.png", result1)

    # imgP1 = cv2.imread("part1/part1.png")
    R1, (width1, height1) = detect_np(ocr_net, ocr_meta, result1, thresh=ocr_threshold,nms=None)


    ht, wd, cc= imgP2.shape
    result2 = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result2[yy:yy+ht, xx:xx+wd] = imgP2
    cv2.imwrite("p2.png", result2)

    R2, (width2, height2) = detect_np(ocr_net, ocr_meta, result2, thresh=ocr_threshold,nms=None)
    # print("???",R1)
    # print("???", R2)
    # R1, (width1, height1) = detect(ocr_net, ocr_meta, b'part1/part1.png', thresh=ocr_threshold,
    #                                nms=None)
    # R2, (width2, height2) = detect(ocr_net, ocr_meta, b'part2/part2.png', thresh=ocr_threshold,
    #                                nms=None)
    List_R1.extend(R1)
    List_R2.extend(R2)

    lp_str1=[""]
    lp_str2=[""]
    if len(List_R1):
        L1 = dknet_label_conversion(List_R1, width1, height1)

        L1 = nms(L1, 0.2)
        # tmp = [l.cl() for l in data_lost]
        # print(":::::::::", tmp)
        L1.sort(key=lambda x: x.br()[0])
        # print(">>>",imgP1.shape)
        # if len(L1)==6:
        #     L1=L1[1:5]
        # print((1-L1[-1].br()[0])-L1[0].tl()[0])

        # L1_result=[]
        # #check outlier
        # # adaptive threshold
        # V = cv2.split(cv2.cvtColor(result1, cv2.COLOR_BGR2HSV))[2]
        # T = threshold_local(V, 27, offset=10, method="gaussian")
        # thresh = (V > T).astype("uint8") * 255
        # # convert black pixel of digits to white pixel
        # thresh = cv2.bitwise_not(thresh)
        # thresh = imutils.resize(thresh, width=240)
        # thresh = cv2.medianBlur(thresh, 5)
        # characters = []
        # for i in L1:
        #     if check_outlier_character(i) :
        #        #print(i.tl())
        #        #print(i.br())
        #        # print(i)
        #        # print((int)(i.tl()[0]*240))
        #        # print((int)(i.br()[0]*240))
        #        # print((int)(i.tl()[1]*80))
        #        # print((int)(i.br()[1]*80))
               

        #        candidate = thresh[(int)(i.tl()[1]*80):(int)(i.br()[1]*80),(int)(i.tl()[0]*240):(int)(i.br()[0]*240)]
        #        square_candidate = convert2Square(candidate)
        #        square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
        #        square_candidate = square_candidate.reshape((28, 28, 1))
        #        cv2.imwrite("1_" + chr(i.cl()) + ".png", square_candidate)
        #        characters.append(square_candidate)
        #        L1_result.append(i)
        # # print(L1_result[0])
        # #print("plot-------------",L1_result[-1].br()[0]-1)
        # #*
        # characters = np.array(characters)
        # #print(len(characters))
        # result = recogChar.predict_on_batch(characters)
        # result_idx = np.argmax(result, axis=1)
        # #print(result_idx)
        # candidates = []
        # for i in range(len(result_idx)):
        #     if result_idx[i] == 31:    # if is background or noise, ignore it
        #         continue
        #     candidates.append((ALPHA_DICT[result_idx[i]]))
        # print(candidates)
        # for i in range(int((len(L1_result)-4)/2)+1):

        #     #print((1-L1_result[-1].br()[0])-L1_result[0].tl()[0])
        #     if len(L1_result)>4:
        #         if (1-L1_result[-1].br()[0])-L1_result[0].tl()[0]>0.15 or L1_result[0].tl()[0]<0:
        #             L1_result = L1_result[1:]
        #         elif (1-L1_result[-1].br()[0])-L1_result[0].tl()[0]<-0.15 or  L1_result[-1].br()[0]-1>0:
        #             L1_result = L1_result[:-1]
        # if len(L1_result)==6:
        #     L1_result=L1_result[1:5]
        # if len(L1_result) > 4 :
        #     # L1.sort(key=lambda  x: x.prob())
        #     if (L1_result[3].cl()>=ord('A') and L1_result[3].cl()<=ord('Z')):
        #         # k = crop_region(imgP1, L1[2])
        #         # cv2.imwrite('part1/' + name + "_" + str(L1[2].cl()) + ".png", k)
        #         L1_result=L1_result[:2]+L1_result[3:]
        #     elif L1_result[2].cl() == ord('I') or L1_result[2].cl() == ord('1') or L1_result[2].cl() == ord('F') :
        #         L1_result=L1_result[:2]+L1_result[3:]


        # for i,l in enumerate(L1_result):
        #     if l.cl() ==ord('1'):
        #         print(l)
        #         k=crop_region(imgP1,l)
        #         cv2.imwrite('part1/' + name + "_" +str(i)+ str(l.cl()) + ".png", k)
        #     if check_outlier_character(k):
        #         L1_result.append(l)
        #         cv2.imwrite('part1/'+name+"_"+str(l.cl())+".png",k)

        #lp_str1 = list(map(lambda x: chr(x.cl()), L1_result))
        lp_str1 = list(map(lambda x: chr(x.cl()), L1))
        # print("warninggggggggg:",lp_str1)

        for i, c in enumerate(lp_str1):
            if i != 2:
                if c == 'L':
                    lp_str1[i] = '4'
                if c == 'Z':
                    lp_str1[i] = '7'
                if c == 'B':
                    lp_str1[i] = '8'
                if c == 'Q' or c == 'O' or c == 'D' or c == 'U':
                    lp_str1[i] = '0'
                if c== 'G':
                    lp_str1[i] = '6'
                if c == 'I' or c=='T':
                    lp_str1[i] = '1'
                if c == 'S':
                    lp_str1[i] = '9'
                if c == 'F':
                    lp_str1[i] = '1'
                if c == 'J':
                    lp_str1[i] = '1'
                if c == 'E':
                    lp_str1[i] = '3'

            else:
                if c == '0':
                    lp_str1[i] = 'U'
                if c=='8':
                    lp_str1[i] = 'B'
                if c=='7' or c =='2':
                    lp_str1[i] = 'Z'
                if c == '6':
                    lp_str1[i] = 'G'
                if c == '5':
                    lp_str1[i] = 'C'
    if len(List_R2) and len(List_R1):
        L2 = dknet_label_conversion(List_R2, width2, height2)
        L2  = nms(L2, .2)

    # lp_str1 = ''.join([chr(l.cl()) for l in L1])
    # lp_str1 = list(lp_str1)
        L2.sort(key=lambda x: x.br()[0])

        for i,l in enumerate(L2):
            if not check_outlier_character(l):
                if i == len(L2)-1:
                    L2= L2[:i]
                else:
                    L2 = L2[:i]+L2[i+1:]
        # print("____",(L2[(len(L2)-1)].br()[0]-L1_result[(len(L1_result)-1)].br()[0])-(L1_result[0].tl()[0]-L2[0].tl()[0]))
        if len(L2) > 5 or L2[-1].cl()==ord('K'):
            # L2.sort(key=lambda x: x.prob())
            L2 = L2[:len(L2)-1]
        mesure =(L2[(len(L2)-1)].br()[0]-L1[(len(L1)-1)].br()[0])-(L1[0].tl()[0]-L2[0].tl()[0])
        if len(L2 )> 4:
            if mesure>0.1 :
                L2 = L2[0:(len(L2) - 1)]

            elif mesure<-0.1:
                L2 = L2[1:]
        if len(L2 )==6:
            if mesure>0 :
                L2 = L2[0:(len(L2) - 1)]

            elif mesure<0:
                L2 = L2[1:]
        #print(L2[0])
        # imgP2 = cv2.imread('part2/part2.png')
        # L2_result=[]
        # for l in L2:
        #     # print("------%s \n"%l.cl())
        #     if l.cl() ==ord('6'):
        #         k = crop_region(imgP2, l)
        #         cv2.imwrite('part2/' + name + "_" + str(l.cl()) + ".png", k)
        # #         break
            # if check_outlier_character(k):
            #     L2.append(l)
            #     cv2.imwrite('part2/' + name + "_" + str(l.cl()) + ".png", k)
        lp_str2 = list(map(lambda x: chr(x.cl()), L2))
        for i, c in enumerate(lp_str2):
            if c == 'L':
                lp_str2[i] = "4"
            if c == 'I' or c=="F" or c == 'T':
                lp_str2[i] = '1'
            if c == "Z":
                lp_str2[i] = "7"
            if c == 'Q' or c == 'U' or c == 'D':
                lp_str2[i] = "0"
            if c == "B":
                lp_str2[i] = "8"
            if c == 'G' or c =='S':
                lp_str2[i] = '6'
            if c == 'E':
                lp_str2[i] = '3'
        #print(lp_str2)
    #print(";;;;",lp_str1)
    lp_str1 = ''.join(lp_str1)
    lp_str2 = ''.join(lp_str2)
    lp_str = lp_str1 + lp_str2

    #print('\t\tLP: %s' % lp_str)

    return lp_str

def detect_ocr_1(ocr_net, ocr_meta, img):

    ocr_threshold = .4

    #print ('Performing OCR 1...')

    img = cv2.resize(img, (240, 80), interpolation = cv2.INTER_AREA)
    # image_path = 'out_lp.png'
    # cv2.imwrite(image_path, img)
    # image_path = bytes(image_path, encoding="utf-8")

    ht, wd, cc= img.shape
    ww = 240
    hh = 80
    color = (255,255,255)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img

    R,(width,height) = detect_np(ocr_net, ocr_meta, result, thresh=ocr_threshold,nms=None)
    first_line = []
    second_line = []
    if len(R):    
        start = R[0]
        first_line.append(start)
        order_down = True
    for r in R:
        if (r[2][1] > start[2][1] + 25):
            second_line.append(r)
        elif (r[2][1] < start[2][1] - 25):
            second_line.append(r)
            order_down = False
        else:
            first_line.append(r)

    #print("first_line:",first_line)
    #print("second_line:",second_line)
    # print("R:",R)
    # print("width:",width)
    # print("height:",height)
    if len(R):

        L1 = dknet_label_conversion(R,width,height)
        L1 = nms(L1,.45)

        L1.sort(key=lambda x: x.tl()[0])

        lp_str = ''.join([chr(l.cl()) for l in L1])

        # if len(second_line):
        #     L2 = dknet_label_conversion(second_line,width,height)
        #     L2 = nms(L2,.45)

        #     L2.sort(key=lambda x: x.tl()[0])

        # if (order_down):
        #     lp_str = lp_str + ''.join([chr(l.cl()) for l in L2])
        # else:
        #     lp_str = ''.join([chr(l.cl()) for l in L2]) + lp_str
    

        # with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
        #     f.write(lp_str + '\n')

        #print ('\t\tLP: %s' % lp_str)

        return lp_str

    else:

        print ('No characters found')
    return ""
