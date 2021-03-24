import numpy as np
from PIL import Image, ImageDraw,ImageChops
from os import listdir
import random
import math,operator
import cv2

def open_image(color = (0,0,0,0)):
    im_1 = Image.open("./resized_images/1.png")
    im_2 = Image.open("./resized_images/2.png")
    mask = Image.open('mask.png')
    new_im = Image.new('RGBA', size=(512,512), color=color)
    #print(im_1.width)
    new_im.paste(im_1, (-15,0), mask)
    new_im.paste(im_2, (im_1.width - 50, 0),mask)
    return new_im

def save_mask():
    img = Image.open(f"./resized_images/{listdir('./images')[0]}")
    out = make_mask(img)
    out.save('mask.png')

def make_mask(prototype):
    img = prototype
    pix = img.load()

    out = Image.new('RGB', size=img.size)
    draw = ImageDraw.Draw(out)

    width = img.size[0]
    height = img.size[1]
    #print("------------------------------------------------------")
    for i in range(width):
        for j in range(height):
            a = pix[i, j]
            #print(a)
            b = pix[i, j]
            c = pix[i, j]
            S = a + b + c
            if (S > 0):
                a, b, c = 255, 255, 255
            else:
                a, b, c = 0, 0, 0
            draw.point((i, j), (a,b,c))
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    return out.convert("L")

def prepare_images(size:int=16):
    for i in listdir('./images'):
        img = Image.open(f"./images/{i}")
        new_img = img.resize((size,size))
        new_img.save(f"./resized_images/{i}")
    save_mask()

#def save_random(amount_of_genes:int = 10):


def random_placing(n:int = 80, img_out = Image.new('RGBA', size=(512,512))):
    mask = Image.open('mask.png')
    #img_out = Image.open("./output/output.png")
    images_c = len(listdir('./resized_images'))
    for i in range(n):
        img = Image.open(f"./resized_images/{random.randint(1, images_c)}.png")
        x = random.randint(0-img.size[0]/2,512-img.size[0]/2)
        y = random.randint(0-img.size[1]/2,512-img.size[1]/2)
        img_out.paste(img, (x, y), mask)
    return  img_out
    #img_out.save("./output/outputr.png")



def init_arr(size:int=16):
    max_width = math.ceil(512 / size) + 1
    output_arr = np.random.randint(len(listdir('./resized_images'))-1, size=(max_width+64//size,max_width))
    return output_arr+1

def arr_to_image(arr, mask):
    new_im = Image.new('RGBA', size=(512, 512), color=(0, 0, 0))

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            img = Image.open(f"./resized_images/{arr[i][j]}.png")
            new_im.paste(img,(j*img.width-img.width*(i%2)//2,i*(img.width-img.width//8)),mask)
    return new_im
#(j*img.width-img.width*(i%2)//2,i*(img.width-img.width//8)) - position of block om image by using j(column) and i(row)
def start_preparations(size_of_blocks:int = 16):
    prepare_images(size_of_blocks)
    x = init_arr(size_of_blocks)
    return x

def new_generation(best_previous, amount_of_genes:int = 2, amount_of_changes:int=3):
    x = len(best_previous)
    y = len(best_previous[0])
    genes_location = np.array([])
    images_amount = len(listdir('./resized_images'))
    for i in range(1,amount_of_genes+1):
        for j in range(amount_of_changes):
            #arr_to_image(copy).save(f"./genes/{i}.png")
            genes_location = np.append(genes_location, (np.random.randint(0,x),
            np.random.randint(0,y),
            np.random.randint(0,images_amount)+1))
    return genes_location.reshape(amount_of_genes,amount_of_changes,3)

def create_crop(i,j,size_of_img = 16):
    #(j * img.width - img.width * (i % 2) // 2, i * (img.width - img.width // 8))
    #[22, 10, 22]
    a = Image.open("./output/kek.png")
    w = size_of_img
    x = (j * w - w * (i % 2) // 2)
    y = i * (w - w // 8)
    #z = Image.open("./resized_images/22.png")
    crop = (x, y, x + w, y + w)
    img_crop = a.crop(crop)
    img_crop.save("./output/crop.png")

def create_apr_crop(i,j,size_of_img = 16):
    #(j * img.width - img.width * (i % 2) // 2, i * (img.width - img.width // 8))
    #[22, 10, 22]
    a = Image.open("./input/input.png")
    w = size_of_img
    x = (j * w - w * (i % 2) // 2)
    y = i * (w - w // 8)
    #z = Image.open("./resized_images/22.png")
    crop = (x, y, x + w, y + w)
    img_crop = a.crop(crop)
    new_im = Image.new('RGBA', size=(512, 512), color=(0, 0, 0, 0))
    img_crop.save("./output/apr_crop.png")

def check_which_better():
    output_arr = np.array([])
    images_amount = len(listdir('./resized_images'))
    for i in range(1,images_amount+1):
        image = cv2.imread(f"./resized_images/{i}.png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_image], [0],
                                 None, [256], [0, 256])

        image = cv2.imread('./output/apr_crop.png')
        gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram1 = cv2.calcHist([gray_image1], [0],
                                  None, [256], [0, 256])

        c1 = 0
        i = 0
        while i < len(histogram) and i < len(histogram1):
            c1 += (histogram[i] - histogram1[i]) ** 2
            i += 1
        c1 = c1 ** (1 / 2)
        output_arr = np.append(output_arr, c1)
    return output_arr

def check_crop():
    image = cv2.imread(f"./output/crop.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0],
                             None, [256], [0, 256])
    image = cv2.imread('./output/apr_crop.png')
    gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram1 = cv2.calcHist([gray_image1], [0],
                              None, [256], [0, 256])

    c1 = 0
    i = 0
    while i < len(histogram) and i < len(histogram1):
        c1 += (histogram[i] - histogram1[i]) ** 2
        i += 1
    c1 = c1 ** (1 / 2)
    return c1

if __name__ == '__main__':
    create_apr_crop(20,16)
    z = check_which_better()
    print(z)
    print(z.argmin()+1,z.min())
    print(z.argmax()+1, z.max())
    print(check_crop())
    #create_crop(2,4)
    #create_apr_crop(0,0)
    #random_placing(2000).save("./output/outputr.png")
    #new_im = Image.new('RGBA', size=(512, 512), color=(153,153,255))
    #new_im = open_image((153,153,255))
    ap_ing = Image.open('./input/input.png')
    #x = start_preparations(16)
    mask = Image.open('mask.png')
    #arr_to_image(x,mask).save(f"./output/kek.png")
    #for i in range(1):
    #    arr = new_generation(x)
    #print(arr)
    #arr = np.array([[[22, 10, 22],[ 8, 17, 25],[35, 32,  5]],[[35,  6,  6,],[25,27, 19],[29, 15,  2]]])

"""
def check_which_better_from_re_img():
    output_arr = np.array([])
    images_amount = len(listdir('./resized_images'))
    for i in range(1,images_amount+1):
        image1 = cv2.imread(f"./resized_images/{i}.png")
        image2 = cv2.imread('./output/apr_crop.png')

        err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
        err /= float(image1.shape[0] * image1.shape[1])
        output_arr = np.append(output_arr, err)
    return output_arr

def check_which_better():
    image1 = cv2.imread(f"./output/crop.png")
    image2 = cv2.imread('./output/apr_crop.png')

    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

if __name__ == '__main__':
    #create_crop(2,4)
    create_apr_crop(20,10)
    z = check_which_better_from_re_img()
    print(z)
    print(z.argmin()+1,z.min())
    print(z.argmax()+1, z.max())
    print(check_which_better())
"""