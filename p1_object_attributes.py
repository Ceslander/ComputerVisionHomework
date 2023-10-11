# #!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
    # TODO: 255 if intensity >= thresh_val else 0
    binary_image = np.where(gray_image >= thresh_val, 255, 0)
    return binary_image


def label(binary_image):
    # TODO
   
    length, width, channels = np.shape(binary_image)
    parent = np.zeros((length, width))

    # union-find set
    def index(i, j):
        # (i,j) to i*width+j.
        assert i < length and j < width, "Illegal index."
        return i*width+j
    def coordinate(index):
        # i*width+j to (i,j).
        j = index % width
        i = round((index-j)/width)
        return i,j
    def union_find_init(parent):
        i = j = 0
        while i < length:
            parent[i][j] = index(i, j)
            if j < width - 1:
                j += 1
            else:
                i += 1
                j = 0
    def find(i, j):
        if parent[i][j] == index(i, j):
            return index(i,j)
        else: 
            parent_i, parent_j = coordinate(parent[i][j])
            return find(parent_i, parent_j)
    def merge(i, j, parent_i, parent_j):
        parent[i][j] = index(parent_i, parent_j)

    return labeled_imag


def get_attribute(labeled_image):
    # TODO
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
