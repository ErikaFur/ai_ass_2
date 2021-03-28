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
    output_arr = np.random.randint(len(listdir('./resized_images'))-1, size=(max_width+64//size-1,max_width-2))
    return output_arr+1

def arr_to_image(arr, mask):
    new_im = Image.new('RGBA', size=(512, 512), color=(0, 0, 0))

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            img = Image.open(f"./resized_images/{arr[i][j]}.png")
            new_im.paste(img,(j*img.width+img.width*(i%2)//2+img.width//4, i*(img.width-img.width//8)+img.width//4),mask)
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

def create_crop(i, j, arr, mask):
    #(j * img.width - img.width * (i % 2) // 2, i * (img.width - img.width // 8))
    #[22, 10, 22]
    a = Image.open(f"./resized_images/{arr[i][j]}.png")
    new_im = Image.new('RGBA', size=a.size, color=(0, 0, 0, 0))
    new_im.paste(a,(0,0),mask)
    new_im.save("./output/crop.png")

def create_apr_crop(i,j,mask,size_of_img = 16):
    #(j * img.width - img.width * (i % 2) // 2, i * (img.width - img.width // 8))
    #[22, 10, 22]
    a = Image.open("./input/input.png")
    w = size_of_img
    x = (j * w + w * (i % 2) // 2) + w // 4
    y = i * (w - w // 8) + w // 4
    crop = (x, y, x + w, y + w)
    img_crop = a.crop(crop)
    new_im = Image.new('RGBA', size=img_crop.size, color=(0, 0, 0, 0))
    new_im.paste(img_crop, (0, 0), mask)
    # new_im.save("./output/apr_crop.png")
    return new_im


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

def mean_colour(input_img):
    img = input_img.convert("RGB").load()
    output = np.array([0, 0, 0])
    count = 0
    for i in range(input_img.size[0]):
        for j in range(input_img.size[1]):
            if img[j, i] != (0, 0, 0):
                count += 1
                output += img[j, i]
                output += (1, 1, 1)
    output //= count
    heh = [hex(x)[2:] for x in output]
    for i in range(3):
        if len(heh[i]) == 1:
            heh[i] = '0' + heh[i]
    heh = heh[0] + heh[1] + heh[2]
    return int(heh,16)


def jf1():
    d = {}
    arr = np.array([])
    images_amount = len(listdir('./resized_images'))
    for i in range(1,images_amount+1):
        img = Image.open(f'./resized_images/{i}.png')
        a = mean_colour(img)
        d[a] = i
        arr = np.append(arr,a)
    arr = np.sort(arr)

    for i in range(images_amount):
        print(int(arr[i]), i+1)

def jf2(a):
    """
    this function returns number of image in folder (starting with 1)
    which is best suited and error for it.
    :param a: part of image (crop) for which we searching
    :return: number of best suited and error
    """
    a = a.convert('RGB')
    img = a.load()
    min_num = 0
    min_err = 100000000
    for k in range(1, 1 + len(listdir('./resized_images'))):
        b = Image.open(f"./resized_images/{k}.png")
        b = b.convert('RGB')
        imgb = b.load()
        output = 0
        count = 0
        for i in range(a.size[0]):
            for j in range(a.size[1]):
                if img[j, i] != (0, 0, 0):
                    count += 1
                    output += (sum((np.array(img[j, i]) - np.array(imgb[j, i])) ** 2)) ** (1 / 2)
        if output < min_err:
            min_err = output
            min_num = k
    min_err /= count
    return min_num, min_err

def best_apr_image(size, mask):
    """
    this function gets size of small images and mask and approximate
    and show best possible result
    :param size: int size of images, which is atomic units of picture construction
    :param mask: mask needed to make proper form of images with invisible borders
    :return: Just shows image, returns None
    """
    w = size
    max_width = math.ceil(512 / size) + 1
    new_im = Image.new('RGB', size=(512,512), color=(0, 0, 0))
    size_y = max_width + 64 // size - 1  # size of arrays of small images arr[y][x]
    size_x = max_width - 2
    for i in range(size_y):
        for j in range(size_x):
            apr_img = create_apr_crop(i,j,mask,size)
            number_of_reimg, error = jf2(apr_img)
            print(i,j)
            print(number_of_reimg, error)
            needed_img = Image.open(f'./resized_images/{number_of_reimg}.png')
            new_im.paste(needed_img,((j * w + w * (i % 2) // 2) + w // 4,i * (w - w // 8) + w // 4),mask)
    new_im.show()


#x = (j * w + w * (i % 2) // 2) + w // 4
#y = i * (w - w // 8) + w // 4
if __name__ == '__main__':
    size = 8
    x = start_preparations(size)
    ap_img = Image.open('./input/input.png')
    mask = Image.open('mask.png')
    best_apr_image(size, mask)
    #arr_to_image(x, mask).save(f"./output/kek.png")
    #create_apr_crop(0,0,mask,size)
    #create_crop(2, 4, x, mask)
    ap_img_c = Image.open('./output/apr_crop.png')
    exit()


    #arr_to_image(x, mask).save(f"./output/kek.png")
    #create_crop(2,4, 16)
    #create_apr_crop(20,10, 16)
    z = check_which_better_from_re_img()
    print(z)
    print(z.argmin()+1,z.min())
    print(z.argmax()+1, z.max())
    print(check_which_better())
    #create_crop(2,4)
    #create_apr_crop(0,0)
    #random_placing(2000).save("./output/outputr.png")
    #new_im = Image.new('RGBA', size=(512, 512), color=(153,153,255))
    #new_im = open_image((153,153,255))

    #x = start_preparations(16)
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