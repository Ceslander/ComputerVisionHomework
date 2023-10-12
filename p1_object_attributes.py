# #!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
    # TODO: 255 if intensity >= thresh_val else 0
    binary_image = np.where(gray_image >= thresh_val, 1, 0)
    return binary_image


def label(binary_image):
    # TODO
    
    height, width, channels = np.shape(binary_image)
    
    # union-find set
    parent = np.zeros((height, width))
    def index(i, j):
        # (i,j) to i*width+j.
        assert i < height and j < width, "Illegal index."
        return i*width+j
    def coordinate(index):
        # i*width+j to (i,j).
        j = index % width
        i = round((index-j)/width)
        return i,j
    # def union_find_init(parent):
    #     i = j = 0
    #     while i < height:
    #         # if binary_image[i][j][0] == 0:
    #         #     parent[i][j] = -1
    
    #         parent[i][j] = index(i, j)
    #         if j < width - 1:
    #             j += 1
    #         else:
    #             i += 1
    #             j = 0
    def find(i, j):
        if parent[i][j] == index(i, j) or parent[i][j] == -1:
            return parent[i][j]
        else: 
            parent_i, parent_j = coordinate(parent[i][j])
            return find(parent_i, parent_j)
    def merge(i, j, parent_i, parent_j):
        parent[i][j] = index(parent_i, parent_j)
    
    # union_find_init(parent)

    # sequence labeling
    i = j = 0
    while i < height:
        if binary_image[i][j][0] == 0:
            parent[i][j] = -1
        else:
            if i > 0 and j > 0:
                # if parent[i-1][j-1] != -1:
                if binary_image[i-1][j-1][0] == 1:
                    parent[i][j] = index(i-1,j-1)
                # elif parent[i-1][j] == -1 and parent[i][j-1] != -1:
                elif binary_image[i-1][j][0] == 0 and binary_image[i][j-1][0] == 1:
                    parent[i][j] = index(i,j-1)
                    # parent[i][j] = parent[i][j-1]
                elif binary_image[i-1][j][0] == 1 and binary_image[i][j-1][0] == 0:
                    parent[i][j] = index(i-1,j)
                elif binary_image[i-1][j][0] == 1 and binary_image[i][j-1][0] == 1:
                    merge(i-1,j,i,j-1)
                    parent[i][j] = index(i-1,j)
                else:
                    assert parent[i-1][j] == -1 and parent[i][j-1] == -1, "Labeling error."
                    parent[i][j] = index(i,j)
            elif i == 0 and j > 0:
                if binary_image[i][j-1][0] == 1:
                    parent[i][j] = index(i,j-1)
                else:
                    parent[i][j] = index(i,j)
            elif i > 0 and j == 0: 
                if binary_image[i-1][j][0] == 1:
                    parent[i][j] = index(i-1,j)
                else:
                    parent[i][j] = index(i,j)
            elif i == 0 and j == 0:
                parent[i][j] = index(i,j)

        if j < width - 1:
            j += 1
        else:
            i += 1
            j = 0 

    labeled_imag = np.zeros((height,width))  
    label_dict = {}     # parent-label reflection
    count = 0           # number of labels
    i = j = 0
    while i < height:
        current_parent = find(i,j)
        if current_parent != -1:
            if current_parent in label_dict:
                labeled_imag[i][j] = label_dict[current_parent]
            else:
                label_dict[current_parent] = count
                count += 1
                labeled_imag[i][j] = label_dict[current_parent]

        if j < width - 1:
            j += 1
        else:
            i += 1
            j = 0         
    return labeled_imag


def get_attribute(labeled_image):
    # TODO
    height, width = labeled_image.shape

    max_object = 100    # Assume no more than `max_object` labels.

    x_bar = np.zeros(max_object)
    y_bar = np.zeros(max_object)

    i = j = 0
    while i < height:
        

        if j < width - 1:
            j += 1
        else:
            i += 1
            j = 0    

    return attribute_list


def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_image = binarize(gray_image, thresh_val=thresh_val)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)

    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
    cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
    print(attribute_list)


if __name__ == '__main__':
    main(sys.argv[1:])
