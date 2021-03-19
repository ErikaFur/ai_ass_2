import numpy as np
from PIL import Image, ImageDraw
from os import listdir
import random

def open_image():
    im_1 = Image.open("./resized_images/1.png")
    im_2 = Image.open("./resized_images/2.png")
    mask = Image.open('mask.png')
    new_im = Image.new('RGBA', size=(512,512))

    new_im.paste(im_1, (0,16), mask)
    new_im.paste(im_2, (8, 8),mask)
    new_im.save("./output/output.png")

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

def prepare_images(size:int=8):
    for i in listdir('./images'):
        img = Image.open(f"./images/{i}")
        new_img = img.resize((size,size))
        new_img.save(f"./resized_images/{i}")
    save_mask()

def random_placing(n:int = 80):
    #mask = Image.open('mask.png')
    img_out = Image.open("./output/output.png")
    images_c = len(listdir('./resized_images'))
    for i in range(n):
        x = random.randint(0,512)
        y = random.randint(0,512)
        img = Image.open(f"./resized_images/{random.randint(1,images_c)}.png")
        img_out.paste(img, (x, y), make_mask(img))
    img_out.save("./output/outputr.png")
if __name__ == '__main__':
    random_placing()
    #prepare_images(16)
    #open_image()
    #save_mask()
