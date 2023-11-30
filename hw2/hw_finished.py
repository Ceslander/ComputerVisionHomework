import cv2
import numpy as np
import glob

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        
        world_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        world_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
   
        return world_points
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)

        if ret:
            return corners
        else:
            return None

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    A = []
    B = []
    K = np.zeros((4, 4))
    P = None

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R

    vij = [[[None, None, None], [None, None, None], [None, None, None]] for _ in range(num_points)]
    V_matrix = []

    for image_index in range(num_points):
        A = []
        for point_index in range(world_points[image_index].shape[0]):
        
            xw = world_points[image_index][point_index,0]
            yw = world_points[image_index][point_index,1]
            u = image_points[image_index][point_index,0,0]
            v = image_points[image_index][point_index,0,1]
            
            A.append([xw, yw, 1, 0, 0, 0, -u*xw, -u*yw, -u])
            A.append([0, 0, 0, xw, yw, 1, -v*xw, -v*yw, -v])

        A = np.array(A)

        eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
        min_eigenvalue_index = np.argmin(eigenvalues)
       
        h_vector = eigenvectors[:, min_eigenvalue_index]

        h = h_vector.reshape(3,3)

        htmp = np.zeros((4,4))
        htmp[1:, 1:] = h
        for i in range(1, 3):
            for j in range(1, 3):
                vij[image_index][i][j] = np.array([htmp[1,i]*htmp[1,j], htmp[1,i]*htmp[2,j]+htmp[2,i]*htmp[1,j], htmp[2,i]*htmp[2,j], htmp[1,i]*htmp[3,j]+htmp[3,i]*htmp[1,j], htmp[2,i]*htmp[3,j]+htmp[3,i]*htmp[2,j], htmp[3,i]*htmp[3,j]]).reshape(6,1)
        V_matrix.append(vij[image_index][1][2])  
        V_matrix.append(vij[image_index][1][1]-vij[image_index][2][2]) 
        
    V_matrix = np.array(V_matrix).reshape(len(V_matrix), 6)

    eigenvalues, eigenvectors = np.linalg.eigh(V_matrix.T @ V_matrix)
    min_eigenvalue_index = np.argmin(eigenvalues)
    b_vector = eigenvectors[:, min_eigenvalue_index]
    b11, b12, b22, b13, b23, b33 = b_vector

    B_matrix = np.array([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]])

    try:
        K = np.linalg.cholesky(B_matrix)
    except np.linalg.LinAlgError:
        K = np.linalg.cholesky(-B_matrix)
    
    K = np.linalg.inv(K.T)
    K = K/K[2,2]

    # Please ensure that the diagonal elements of K are positive
    if K[0,0] < 0:
        K = -K
    return K, P

def cal_reprojection_error(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    A = []
    B = []
    K = np.zeros((4, 4))
    P = None
    Hs = []

    vij = [[[None, None, None], [None, None, None], [None, None, None]] for _ in range(num_points)]
    V_matrix = []

    for image_index in range(num_points):
        A = []
        for point_index in range(world_points[image_index].shape[0]):
   
            xw = world_points[image_index][point_index,0]
            yw = world_points[image_index][point_index,1]
            u = image_points[image_index][point_index,0,0]
            v = image_points[image_index][point_index,0,1]
       
            A.append([xw, yw, 1, 0, 0, 0, -u*xw, -u*yw, -u])
            A.append([0, 0, 0, xw, yw, 1, -v*xw, -v*yw, -v])

        A = np.array(A)

        eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
        min_eigenvalue_index = np.argmin(eigenvalues)
        h_vector = eigenvectors[:, min_eigenvalue_index]

        h = h_vector.reshape(3,3)
        Hs.append(h)

    world_points_copy = world_points.copy()
    for image_index in range(num_points):
        points_estimated = []
        world_points_copy[image_index][:,2] = 1
        for point_index in range(world_points_copy[image_index].shape[0]):
            uv = Hs[image_index] @ world_points_copy[image_index][point_index]
            
            uv = uv/uv[2]
            uv = uv[:2].reshape(1,2)
          
            points_estimated.append(uv)
        points_estimated = np.array(points_estimated)


        error = np.linalg.norm(image_points[image_index] - points_estimated) / len(points_estimated)
        print(f"Image {image_index} error:", "%.3f" % error)




# Main process
image_path = 'calibration'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (9,6)  # The pattern size of the chessboard 

world_points, image_points = find_image_points(images, pattern_size)

camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)
cal_reprojection_error(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)

def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO

    images = np.array(read_images(image_directory))
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, images[0].transpose(2,0,1).shape[1:], None, None)

    # print("Camera Calibration Matrix by OpenCV:")
    print(camera_matrix)

def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image

    images = np.array(read_images("data/calibration"))
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, images[0].transpose(2,0,1).shape[1:], None, None)

    for i in range(len(world_points)):
        img_points, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
        error = cv2.norm(image_points[i], img_points, cv2.NORM_L2) / len(img_points)
        print(f"Image {i} error calculated by OpenCV:", "%.3f" % error)

world_points, image_points = find_image_points(images, pattern_size)

print("Camera Calibration Matrix by OpenCV:")
test(image_path, pattern_size)
reprojection_error(world_points, image_points, camera_matrix)

