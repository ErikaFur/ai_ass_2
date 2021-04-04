import numpy as np
from PIL import Image, ImageDraw
from os import listdir
import math


def save_mask():
    """
    function calls make_mask() with first resized image and save it
    :return: None
    """
    img = Image.open(f"./resized_images/{listdir('./images')[0]}")
    out = make_mask(img)
    out.save('mask.png')

def make_mask(prototype):
    """
    function creates and returns mask for deleting borders
    :param prototype: image on which base mask is created
    :return: mask image
    """
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
    """
    function which for each picture in folder 'image' doing resize
    :param size: size of image
    :return: None
    """
    for i in listdir('./images'):
        img = Image.open(f"./images/{i}")
        new_img = img.resize((size,size))
        new_img.save(f"./resized_images/{i}")
    save_mask()

def init_arr(size:int=16):
    """
    function creates random array with indexes of images
    :param size: size of images
    :return: array with indexes
    """
    max_width = math.ceil(512 / size) + 1
    output_arr = np.random.randint(len(listdir('./resized_images'))-1, size=(max_width+64//size-1,max_width-2))
    return output_arr+1

def arr_to_image(arr, arr_imgs, mask):
    """
    function accept array of indexes of images, array of images and mask
    and creates new image which represents given array
    :param arr: array of indexes of images
    :param arr_imgs: array of images
    :param mask: mask
    :return: corresponding to given arr array image
    """
    new_im = Image.new('RGBA', size=(512, 512), color=(0, 0, 0))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            img = arr_imgs[arr[i][j]]
            new_im.paste(img,(j*img.width+img.width*(i%2)//2+img.width//4, i*(img.width-img.width//8)+img.width//4),mask)
    return new_im

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
    return new_im

def jf2(a, bc):
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
        b = bc[k]
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

def best_apr_image(size, mask, input_img, build_cells):
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
    output = np.zeros((size_y,size_x), dtype=int)
    for i in range(size_y):
        for j in range(size_x):
            apr_img = create_apr_crop(i,j,mask,size,input_img)
            number_of_reimg, error = jf2(apr_img, build_cells)
            output[i][j] = number_of_reimg
            needed_img = build_cells[number_of_reimg]
            new_im.paste(needed_img,((j * w + w * (i % 2) // 2) + w // 4,i * (w - w // 8) + w // 4),mask)
    new_im.show()
    return output

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

def error_chech(img_a, img_b):
    """
    function which returns error for img_b in compare with img_a
    :param img_a: image 1
    :param img_b: image 2
    :return: square root of sum of squares of pixel's errors
    """
    a = img_a.convert('RGB')
    img = a.load()
    b = img_b.convert('RGB')
    imgb = b.load()
    output = 0
    count = 0
    for i in range(a.size[0]):
        for j in range(a.size[1]):
            if img[j, i] != (0, 0, 0):
                count += 1
                output += (sum((np.array(img[j, i]) - np.array(imgb[j, i])) ** 2)) #** (1 / 2)
    output /= count
    return output

def create_init_arrs(size, amount):
    """
    function creates initial random arrays
    :param size: size of one side of picture
    :param amount: amount of initial arrays
    :return: arrays
    """
    output_arr = np.array([init_arr(size)])
    for i in range(amount-1):
        output_arr =  np.append(output_arr, [init_arr(size)], axis=0)
    return output_arr

def err_list_from(input_cells, building_cells, gen, shape):
    """
    compute error of one gen
    :param input_cells: cells of input
    :param building_cells: array of minecraft blocks
    :param gen: one gen
    :param shape: shape of gen
    :return: error arrays
    """
    output_err_arr = np.array([])
    for i in range(shape[0]):
        for j in range(shape[1]):
            output_err_arr = np.append(output_err_arr, error_chech(input_cells[i][j],building_cells[gen[i][j]]))
    return np.reshape(output_err_arr, shape)

def sum_err_list(list_err):
    """
    compute sum of errors for each gen
    :param list_err: array of errors
    :return: array of sum of errors
    """
    output = np.array(np.sum(list_err[0]))
    for i in range(1, len(list_err)):
        output = np.append(output,np.sum(list_err[i]))
    return output

def del_half(err_list):
    """
    delete worst half of genes
    :param err_list: list of sum error
    :return: indexes of genes to delete
    """
    output = np.argwhere(err_list > np.median(err_list))
    return output.T[0]

def clean_population(genes, genes_err_list, sum_genes_err_list, arr_mask):
    """
    apply mask to clear genes array from worst genes
    :param genes: array of genes
    :param genes_err_list: array of errors
    :param sum_genes_err_list: array of sum of errors
    :param arr_mask: array of indexes to delete
    :return: new genes, array of errors, array of sum of errors
    """
    genes = np.delete(genes, arr_mask, axis = 0)
    genes_err_list = np.delete(genes_err_list, arr_mask, axis = 0)
    sum_genes_err_list = np.delete(sum_genes_err_list, arr_mask, axis=0)
    return genes, genes_err_list, sum_genes_err_list

def mutations(genes, genes_err_list, shape, amount_of_mutations, input_cells, building_cells):
    """
    function which do mutations. It generates random mutations of chromosomes on random places in gen
    :param genes: array of genes
    :param genes_err_list: array of errors
    :param shape: shape of gen
    :param amount_of_mutations: amount of mutations in one gen
    :param input_cells: array of cells of input image
    :param building_cells: array of minecraft blocks images
    :return: new array of genes, new array of errors, new array of sum of errors
    """
    remains = len(genes)
    for j in range(remains):
        new_gen = np.copy(genes[j])
        new_genes_err_list = np.copy(genes_err_list[j])
        for i in range(amount_of_mutations):
            y = np.random.randint(shape[0])
            x = np.random.randint(shape[1])
            new_gen[y][x] = np.random.randint(1,len(building_cells))
            new_genes_err_list[y][x] = error_chech(input_cells[y][x], building_cells[new_gen[y][x]])
        genes = np.append(genes,np.array([new_gen]),axis=0)
        genes_err_list = np.append(genes_err_list, np.array([new_genes_err_list]), axis=0)
    sum_err = sum_err_list(genes_err_list)
    return genes, genes_err_list, sum_err