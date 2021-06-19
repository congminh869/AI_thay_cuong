import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw, ImageFilter
import random as rd
import math
from tqdm import tqdm

import numpy
import skimage

#///////////////////////////////////////////////////////////
# khai báo
path_dir_font = glob.glob('/media/minh/New Volume2/hoctaps/nam4/tri_tue_nhan_tao_va_ung_dung/hocKy2/codebtl/Make_data_Textfusenet1/fonts/*.ttf')
mode_noise = ["gaussian", "poisson", "pepper", None]
#dir_font = "./fonts/univers59ultracondensed.ttf"
path_background = glob.glob('/media/minh/New Volume2/hoctaps/nam4/tri_tue_nhan_tao_va_ung_dung/hocKy2/codebtl/Make_data_Textfusenet1/backgrounds/*.jpg')
class_char = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
              'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
gt_path = '/media/minh/New Volume2/hoctaps/nam4/tri_tue_nhan_tao_va_ung_dung/hocKy2/codebtl/Make_data_Textfusenet1/custom/gt/'
img_path_save = '/media/minh/New Volume2/hoctaps/nam4/tri_tue_nhan_tao_va_ung_dung/hocKy2/codebtl/Make_data_Textfusenet1/custom/images/'
class_txt = open(gt_path +'classes.txt', 'w')
for i in class_char:
    class_txt.write(i+'\n')
class_txt.close()
number_images = int(input("Nhập vào số lượng ảnh cần tạo: "))
dict_char = dict()
arr_check = ('0', '1')
#///////////////////////////////////////////////////////////

for i in range(len(class_char)):
    dict_char[class_char[i]] = i
with open('container_3.txt', 'r') as file:
    data = file.read()
    dir_string = data.split('\n')

# Make noise
def noise(img):
    mode = rd.choice(mode_noise)
    data = numpy.asarray(img)
    if mode is not None:
        data_noise = skimage.util.random_noise(data, mode=mode)*255.0
        im_noise = Image.fromarray(numpy.uint8(data_noise))
    else:
        im_noise = Image.fromarray(numpy.uint8(data))
    return im_noise
# Vertical
def generate_img_vertical(string, fonts, fonts_size, space_char):
    w, h = fonts.getsize('W')
    img = Image.new('RGB', (w, h*len(string)+space_char*(len(string)-1)), (255, 255, 255))
    img_mask = Image.new('L', (w, h*len(string)+space_char*(len(string)-1)), 0)
    draw = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(img_mask)
    number = 0
    color = rd.choice([(255, 255, 255), (0, 0, 0)])
    for i in string:
        w_ch, h_ch = fonts.getsize(i)
        w_point = round((w-w_ch)/2)
        draw.text((w_point,number*(h+space_char)),i, color ,font=fonts)
        draw_mask.text((w_point,number*(h+space_char)),i,255,font=fonts)
        number+=1
    return img, img_mask, w, h
def detect_local_vertical(im_ro, w_o, h_o, skew, space_char):
    w_n, h_n = im_ro.size
    skew = math.radians(skew)
    x1 = math.cos(skew)*w_o
    y1 = math.sin(skew)*w_o
    local_char = dict()
    local_result= dict()
    count = 0
    point = [(x1, 0), (0, 0), (0, 0), (0, y1)]
    for i in range(len(string)):
        count += 1
        point_result = [(0, 0), (0, 0), (0, 0), (0, 0)]
        point[1] = (x1+(count*h_o+(count-1)*space_char)*math.sin(skew), (count*h_o+(count-1)*space_char)*math.cos(skew))
        point[2] = ((count*h_o+(count-1)*space_char)*math.sin(skew), y1+(count*h_o+(count-1)*space_char)*math.cos(skew))
        for k in range(4):
            point_result[k] = (round(point[k][0]), round(point[k][1]))
        local_char[i] = point_result
        point[0] = (point[1][0]+space_char*math.sin(skew), point[1][1] +space_char*math.cos(skew))
        point[3] = (point[2][0]+space_char*math.sin(skew), point[2][1]+space_char*math.cos(skew))
    draw_im_ro = ImageDraw.Draw(im_ro)
    for k in range(len(string)):
        char = string[k]
        [shape_1, shape_2, shape_3, shape_4] = local_char[k]
        x_min = min(shape_1[0], shape_2[0], shape_3[0], shape_4[0])
        x_max = max(shape_1[0], shape_2[0], shape_3[0], shape_4[0])
        y_min = min(shape_1[1], shape_2[1], shape_3[1], shape_4[1])
        y_max = max(shape_1[1], shape_2[1], shape_3[1], shape_4[1])
        shape = [(x_min , y_min), (x_max, y_max)]
        #draw_im_ro.rectangle(shape, fill =None, outline ="red")
        local_result[k] = shape 
    return local_result
#Horizon
def generate_img_horizon(string, fonts, fonts_size, space_char):
    w, h = fonts.getsize('W')
    w_max = 0
    for i in string:
        w_i, h_i = fonts.getsize(i)
        w_max += w_i + space_char
    w_max = w_max - space_char
    img = Image.new('RGB', (w_max, h+2), (255, 255, 255))
    img_mask = Image.new('L', (w_max, h+2), 0)
    draw = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(img_mask)
    w_conti = 0
    color = rd.choice([(255, 255, 255), (0, 0, 0)])
    for i in string:
        w_ch, h_ch = fonts.getsize(i)
        draw.text((w_conti,0),i,color ,font=fonts)
        draw_mask.text((w_conti,0),i,255,font=fonts)
        w_conti += w_ch + space_char
    return img, img_mask, w_max
def detect_local_horizon(string, im_ro, fonts, w_max, skew, space_char):
    w_n, h_n = im_ro.size
    #print(w_n, h_n)
    skew = math.radians(skew)
    _, h_0 = fonts.getsize(string[0])
    y1 = math.sin(skew)*w_max
    x1 = math.sin(skew)*h_0
    local_char = dict()
    local_result= dict()
    point = [(0, y1), (0, 0), (0, 0), (x1, h_n)]
    w_point_1 = 0
    w_point_2 = x1
    h_point_1 = y1
    h_point_2 = h_n
    for i in range(len(string)):
        char = string[i]
        w_i, h_i = fonts.getsize(char)
        #print(char, w_i, h_i)
        point_result = [(0, 0), (0, 0), (0, 0), (0, 0)]
        w_point_1 += w_i*math.cos(skew)
        h_point_1 -= math.sin(skew)*w_i
        w_point_2 += math.cos(skew)*w_i
        h_point_2 -= w_i*math.sin(skew)
        point[1] = (w_point_1, h_point_1)
        point[2] = (w_point_2, h_point_2)
        for k in range(4):
            point_result[k] = (round(point[k][0]), round(point[k][1]))
        #print(char, point_result)
        local_char[i] = point_result
        w_point_1 = w_point_1 + space_char*math.cos(skew)
        h_point_1 = h_point_1 - space_char*math.sin(skew)
        w_point_2 = w_point_2 + space_char*math.cos(skew)
        h_point_2 = h_point_2 - space_char*math.sin(skew)
        point[0] = (w_point_1, h_point_1)
        point[3] = (w_point_2, h_point_2)
    #draw_im_ro = ImageDraw.Draw(im_ro)
    for k in range(len(string)):
        char = string[k]
        [shape_1, shape_2, shape_3, shape_4] = local_char[k]
        x_min = min(shape_1[0], shape_2[0], shape_3[0], shape_4[0])
        x_max = max(shape_1[0], shape_2[0], shape_3[0], shape_4[0])
        y_min = min(shape_1[1], shape_2[1], shape_3[1], shape_4[1])
        y_max = max(shape_1[1], shape_2[1], shape_3[1], shape_4[1])
        shape = [(x_min , y_min), (x_max, y_max)]
        local_result[k] = shape
        #draw_im_ro.rectangle(shape, fill = None, outline ="red")
    return local_result
#///////////////////////////////////////////////////////////
# Rotate Images
def rotate_img(img, img_mask, skew):
    img_ro = img.rotate(skew, expand = True, resample=Image.BICUBIC)
    img_ro_mask = img_mask.rotate(skew, expand = True, resample=Image.BICUBIC)
    return img_ro, img_ro_mask
# Blur Images
def blur_img(img_mask, blur):
    img_blur_mask = img_mask.filter(ImageFilter.BoxBlur(blur))
    return img_blur_mask
# Choose_background
def choose_background(img_text, path_background):
    background = rd.choice(path_background)
    #print(background)
    img_bg = Image.open(background)
    w_bg, h_bg = img_bg.size
    w_txt, h_txt = img_text.size
    if w_bg < w_txt:
        rate = w_txt/w_bg
        img_bg = img_bg.resize((round(w_bg*rate),round(h_bg*rate)))
    w_bg, h_bg = img_bg.size
    if h_bg < h_txt:
        rate = h_txt/h_bg
        img_bg = img_bg.resize((round(w_bg*rate),round(h_bg*rate)))
    w_bg_re, h_bg_re = img_bg.size
    #print(w_bg_re-w_txt)
    start_w_point = rd.randint(0, w_bg_re-w_txt)
    start_h_point = rd.randint(0, h_bg_re-h_txt)
    shape = (start_w_point, start_h_point, start_w_point+w_txt, start_h_point+h_txt)
    img_result = img_bg.crop(shape)
    return img_result
#///////////////////////////////////////////////////////////////////
#main
for i in tqdm(range(number_images),desc ="Images"):
    gt_path = '/media/minh/New Volume2/hoctaps/nam4/tri_tue_nhan_tao_va_ung_dung/hocKy2/codebtl/Make_data_Textfusenet1/custom/gt/'+ 'img_' + '%09d'%i + '.txt'
    images_path = img_path_save + 'img_' + '%09d'%i + '.jpg'
    f = open(gt_path, 'w')
    string = rd.choice(dir_string)
    fonts_size = rd.randint(13,80)
    dir_font = rd.choice(path_dir_font)
    fonts = ImageFont.truetype(dir_font, fonts_size)
    if fonts_size < 30:
        blur = 0
    elif fonts_size < 50:
        blur = rd.randint(50, 150)/100
    else:
        blur = rd.randint(150, 250)/100
    choice_st = int(rd.choice(arr_check))
    #print(choice_st)
    if choice_st == 1:
        space_char = 0
        img, img_mask, w, h = generate_img_vertical(string, fonts, fonts_size, space_char)
        skew = rd.randint(-500, 500)/100
        im_rotate, im_ro_mask = rotate_img(img, img_mask, skew)
        img_background = choose_background(im_rotate, path_background)
        local_char = detect_local_vertical(im_rotate, w, h, skew, space_char)
        #print(local_char)
        im_ro_mask = noise(im_ro_mask)
        im_blur_mask = blur_img(im_ro_mask, blur)
        img_bg = img_background.copy()
        img_bg.paste(im_rotate,(0,0),im_blur_mask)
        #draw_result = ImageDraw.Draw(img_bg)
        w, h = img_bg.size
        for i in range(len(string)):
            char = string[i]
            [(x_min , y_min), (x_max, y_max)] = local_char[i]
            center_x = (x_min+x_max)/(2*w)
            center_y = (y_min+y_max)/(2*h)
            w_txt = (x_max-x_min)/w
            h_txt = (y_max-y_min)/h
            index = dict_char[char]
            arr = [str(index), str(round(center_x,6)), str(round(center_y,6)), str(round(w_txt,6)), str(round(h_txt,6))]
            local = [str(index), str(x_min), str(y_min), str(x_max), str(y_max)]
            #f.write(' '.join(local)+'\n')
            f.write(' '.join(arr) + '\n')
            #draw_result.rectangle((local_char[i]), fill = None, outline ="red")
            #print(arr)
        img_bg.save(images_path)
        f.close()
        img_bg.save(images_path)
    else :
        skew = rd.randint(-500, 500)/100
        space_char = rd.randint(0, 5)
        img, img_mask, w_max = generate_img_horizon(string, fonts, fonts_size, space_char)
        skew = rd.randint(0, 500)/100
        im_rotate, im_ro_mask = rotate_img(img, img_mask, skew)
        local_char = detect_local_horizon(string, im_rotate, fonts, w_max, skew, space_char)
        #print(local_char)
        im_ro_mask = noise(im_ro_mask)
        im_blur_mask = blur_img(im_ro_mask, blur)
        img_background = choose_background(im_rotate, path_background)
        img_bg = img_background.copy()
        img_bg.paste(im_rotate,(0,0),im_blur_mask)
        w, h = img_bg.size
        draw_result = ImageDraw.Draw(img_bg)
        for i in range(len(string)):
            char = string[i]
            [(x_min , y_min), (x_max, y_max)] = local_char[i]
            center_x = (x_min+x_max)/(2*w)
            center_y = (y_min+y_max)/(2*h)
            w_txt = (x_max-x_min)/w
            h_txt = (y_max-y_min)/h
            index = dict_char[char]
            arr = [str(index), str(round(center_x,6)), str(round(center_y,6)), str(round(w_txt,6)), str(round(h_txt,6))]
            #local = [str(index), str(x_min), str(y_min), str(x_max), str(y_max)]
            #print(arr)
            #f.write(' '.join(local)+'\n')
            f.write(' '.join(arr) + '\n')
            #draw_result.rectangle((local_char[i]), fill = None, outline ="red")
        #img_bg.show()
        img_bg.save(images_path)
        f.close()
    
