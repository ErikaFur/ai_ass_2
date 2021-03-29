import numpy as np
from PIL import Image, ImageDraw
from os import listdir
import random
import math
import cv2 #this lib needs only for mask

def open_image(color = (0,0,0,0)): #debag function
    im_1 = Image.open("./resized_images/1.png")
    im_2 = Image.open("./resized_images/2.png")
    mask = Image.open('mask.png')
    new_im = Image.new('RGBA', size=(512,512), color=color)
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
    for i in range(width):
        for j in range(height):
            a = pix[i, j]
            b = pix[i, j]
            c = pix[i, j]
            S = a + b + c
            if (S > 0):
                a, b, c = 255, 255, 255
            else:
                a, b, c = 0, 0, 0
            draw.point((i, j), (a,b,c))
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

def arr_to_image(arr, arr_imgs, mask):
    new_im = Image.new('RGBA', size=(512, 512), color=(0, 0, 0))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            img = arr_imgs[arr[i][j]]
            new_im.paste(img,(j*img.width+img.width*(i%2)//2+img.width//4, i*(img.width-img.width//8)+img.width//4),mask)
    return new_im
#(j*img.width-img.width*(i%2)//2,i*(img.width-img.width//8)) - position of block om image by using j(column) and i(row)
def start_preparations(size_of_blocks:int = 16):
    """
    function which do preparations. 1) Prepare building cells images, 2) create initial arrays
    :param size_of_blocks: size of cells images
    :return: array of arrays of indexes of images
    """
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

def create_apr_crop(i,j,mask,size_of_img, inp_img):
    """
    function creates crop of inp_img image on indexes [i,j]
    :param i: column index
    :param j: row index
    :param mask: mask
    :param size_of_img: size of resized images
    :param inp_img: image from which it is needed to take crop
    :return: image (size_of_img x size_of_img)
    """
    a = inp_img
    w = size_of_img
    x = (j * w + w * (i % 2) // 2) + w // 4
    y = i * (w - w // 8) + w // 4
    crop = (x, y, x + w, y + w)
    img_crop = a.crop(crop)
    new_im = Image.new('RGBA', size=img_crop.size, color=(0, 0, 0, 0))
    new_im.paste(img_crop, (0, 0), mask)
    # new_im.save("./output/apr_crop.png")
    return new_im

def check_error():
    image1 = cv2.imread(f"./output/crop.png")
    image2 = cv2.imread('./output/apr_crop.png')

    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

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

def best_apr_image(size, mask, input_img):
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
            apr_img = create_apr_crop(i,j,mask,size,input_img)
            number_of_reimg, error = jf2(apr_img)
            #print(i,j)
            #print(number_of_reimg, error)
            needed_img = Image.open(f'./resized_images/{number_of_reimg}.png')
            new_im.paste(needed_img,((j * w + w * (i % 2) // 2) + w // 4,i * (w - w // 8) + w // 4),mask)
    new_im.show()

def break_input(size,mask,input_img):
    """
    divide image on cells
    :param size: size of cells
    :param mask: mask to delete black borders
    :param input_img: input image for processing
    :return: array of images on corresponding places
    """
    max_width = math.ceil(512 / size) + 1
    size_y = max_width + 64 // size - 1  # size of arrays of small images arr[y][x]
    size_x = max_width - 2
    output_arr = []
    for i in range(size_y):
        output_arr.append([])
        for j in range(size_x):
            apr_img = create_apr_crop(i,j,mask,size, input_img)
            output_arr[i].append(apr_img)
            #print(i,j)
    return output_arr

def open_resized(mask):
    """
    function which returns array of resized images
    :param mask: mask added as 0 element in array
    :return: array of resized images
    """
    output_arr = [mask]
    for i in range(1, 1 + len(listdir('./resized_images'))):
        output_arr.append(Image.open(f"./resized_images/{i}.png"))
    return output_arr
#x = (j * w + w * (i % 2) // 2) + w // 4
#y = i * (w - w // 8) + w // 4

if __name__ == '__main__':
    size = 16
    x = start_preparations(size)
    ap_img = Image.open('./input/input.png')
    mask = Image.open('mask.png')
    input_cells = break_input(size, mask, ap_img)
    building_cells = open_resized(mask)
    input_cells[1][0].show()
    #print(z)
    #arr_to_image(x, mask).save(f"./output/kek.png")
    #create_apr_crop(0,0,mask,size)
    #create_crop(2, 4, x, mask)
    exit()
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