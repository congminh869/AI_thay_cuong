from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw, ImageFilter
import numpy as np
import cv2
import random
from random import randint
import glob
import random as rd
from tqdm import tqdm 
from time import sleep
#number_images = int(input("Nhập vào số lượng ảnh muốn tạo: "))
base_name_vertical = './data/'
number_lst = ['0', '1', '2', '3', '4','5','6', '7', '8', '9']
char = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# khoi tao index cho yolo
dic_class = dict()
for i in range(36):
    dic_class[int(i)] = char[i]
val_list= list(dic_class.values())
# khai bao bien can dung
dir_color = ('white','white')
background_path = glob.glob('./backgrounds/*.jpg')
blur_max = 120
font = "univers59ultracondensed.ttf"
#chọn background random
def random_background(img,img_mask, background_path):
    background = random.choice(background_path)
    im = Image.open(background)
    weight, height = im.size
    img, img_mask = resize_img(img,img_mask, im)
    w, h = img.size
    point_start_left = rd.randint(0, weight - w)
    point_start_upper = rd.randint(0, height - h)
    area_crop = (point_start_left, point_start_upper, point_start_left+w, point_start_upper+h)
    #print(area_crop)
    im_background = im
    return im_background, img, img_mask, area_crop
def blur_images(images, images_mask):
    blur = random.randint(0, blur_max)/100
    blur_img = images.filter(ImageFilter.BoxBlur(blur))
    blur_img_mask = images.filter(ImageFilter.BoxBlur(blur))
    return blur_img, blur_img_mask
# tạo ảnh chuỗi và mash từ chuỗi đã cho
def generate_char(char, space_char,font):
    color = rd.choice(dir_color)
    font_size = randint(5, 64)
    font = ImageFont.truetype(font, font_size)
    char_info = dict()
    w, h = char.getsize
    im_new = Image.new('RGB', (w, h), (255, 255, 255))
    im_mask = Image.new('L', (w, h), 0)
    draw_mask = ImageDraw.Draw(im_mask)
    draw = ImageDraw.Draw(im_new)
    draw.text((0, 0),i,color ,font=font)
    draw_mask.text((0, 0),i,255 ,font=font)
    return im_new, im_mask
#xoay ảnh với góc nghiêng được chọn
def rotate_img(images,images_mask,skew_angle):
    im_rotate = images.rotate(skew_angle,expand=True, resample=Image.BICUBIC)
    im_rotate_mask = images_mask.rotate(skew_angle,expand=True, resample=Image.BICUBIC)
    w_ro, h_ro = im_rotate.size
    return im_rotate, images_mask, w_ro, h_ro
for check in tqdm(range(number_images),desc ="Images"):
    char_select = rd.choice(char)
    space_char = randint(1, 4)
    im_gen, im_gen_mask = generate_char(char_select, space_char, font)
    
