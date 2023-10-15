# #!/usr/bin/env python3
import cv2
import numpy as np


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """

    height, width = image.shape

    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    padded_image = np.zeros((height+2,width+2), dtype=image.dtype)
    padded_image[1:height+1, 1:width+1] = image
    edge_image = np.zeros((height,width), dtype=np.float32)

    for i in range(0,height):
        for j in range(0,width):
            conv_region = padded_image[i:i+3, j:j+3]

            gradient_x = np.sum(conv_region*gx)
            gradient_y = np.sum(conv_region*gy)

            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            edge_image[i,j] = gradient_magnitude
    edge_image = (edge_image / np.max(edge_image)) * 255
    return edge_image

def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """

    height, width = edge_image.shape
    thresh_edge_image = np.where(edge_image >= edge_thresh, 255, 0)
    accum_array = np.zeros((len(radius_values), height, width), dtype=np.uint32)
    edge_pixels = np.argwhere(thresh_edge_image > 0)
    
    for r_index, radius in enumerate(radius_values):
        for (x,y) in edge_pixels:
            for theta in range(0,360,5):
                a = round(x-radius*np.cos(np.deg2rad(theta)))
                b = round(y-radius*np.sin(np.deg2rad(theta)))

                if 0 <= a < height and 0 <= b < width:
                    accum_array[r_index, a, b] += 1

    return thresh_edge_image, accum_array

def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    
    circles = []
    centers = np.argwhere(accum_array >= hough_thresh)
    circle_image = image

    for (r_index, a, b) in centers:
        exist = False
        r = radius_values[r_index]
        for (rr, aa, bb) in circles:
            if (aa-a)**2 + (bb-b)**2 < 2 and (rr - r)**2 < 2:
                exist = True
        if exist:
            continue
        cv2.circle(circle_image, (b,a), r, (0,255,0), thickness=2)
        circles.append((r,a,b))
    return circles, circle_image

def main():
    img_name = "coins"
    edge_thresh = 105
    radius_values = list(range(23,32))
    hough_thresh = 50
    
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge_image = detect_edges(gray_image)
    thresh_edge_image, accum_array = hough_circles(edge_image, edge_thresh, radius_values)
    # print(accum_array[:10,:10,:10])
    circles_list, circle_image = find_circles(img, accum_array, radius_values, hough_thresh)

    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_edge.png", edge_image)
    cv2.imwrite('output/' + img_name + "_thresh_edge.png", thresh_edge_image)
    cv2.imwrite('output/' + img_name + "_circle.png", circle_image)

    print(len(circles_list))
    print(circles_list)


if __name__ == '__main__':
    main()
