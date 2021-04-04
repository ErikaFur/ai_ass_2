import numpy as np
from PIL import Image
from functions import prepare_images, break_input, open_resized, create_init_arrs, err_list_from, sum_err_list, \
    clean_population, mutations, del_half, arr_to_image

if __name__ == '__main__':
    print('preparations')
    size = 8
    population = 10
    prepare_images(size)
    apr_img = Image.open('./input/input.png')  # initial picture
    mask = Image.open('mask.png')  # mask
    input_cells = break_input(size, mask, apr_img)  # arrays of images of input's cells (size shape)
    img_shape = (len(input_cells), len(input_cells[0]))  # shape of picture
    building_cells = open_resized(mask)  # array of images of building blocks (0's element is a mask)
    genes = create_init_arrs(size, population)  # arrays of number of building cells (size (population, shape)
    genes_err_list = np.array(
        [err_list_from(input_cells, building_cells, genes[0], img_shape)])  # (size (population, shape)
    for i in range(1, population):
        gene_err_list = err_list_from(input_cells, building_cells, genes[i], img_shape)
        genes_err_list = np.append(genes_err_list, [gene_err_list], axis=0)
    sum_genes_err_list = sum_err_list(genes_err_list)
    # end of preparations
    print('end of the preparations')
    init_err = max(sum_genes_err_list)
    for i in range(100001):
        genes, genes_err_list, sum_genes_err_list = clean_population(genes, genes_err_list, sum_genes_err_list,
                                                                     del_half(sum_genes_err_list))
        genes, genes_err_list, sum_genes_err_list = mutations(genes, genes_err_list, sum_genes_err_list, img_shape, 4,
                                                              input_cells, building_cells)
        if i % 5000 == 0:
            arr_to_image(genes[sum_genes_err_list.argmax()], building_cells, mask).save(f'./output/{i}.png')
            print(i + 1, "current error:", max(sum_genes_err_list), "improvement:",
                  (1 - max(sum_genes_err_list) / init_err) * 100, '%')
    print("the end")
