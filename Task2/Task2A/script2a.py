import cv2
import numpy as np
import glob

def localize_bot():
    """
    Task 2A: Camera Calibration and ArUco Pose Estimation
    """
    # Initialize the output dictionary with exact keys required by the evaluator
    result = {
        "camera_matrix_trace": 0.0,
        "markers": {}
    }

    # ==========================================
    # STEP 1: Camera Calibration
    # ==========================================
    # TODO: Define the real-world 3D coordinates for the checkerboard corners (9x6 grid, 2.5cm square size)
    
    # TODO: Use glob to read all images from the 'calibration_images' folder
    
    # TODO: Loop through the images, convert to grayscale, and use cv2.findChessboardCorners()
    
    # TODO: Use cv2.calibrateCamera() to calculate the camera matrix (mtx) and distortion coefficients (dist)
    
    # TODO: Calculate the trace of the camera matrix (sum of the main diagonal elements)
    # result["camera_matrix_trace"] = round(trace_value, 2)
    
    
    CHECKERBOARD = (9, 6)
    SQUARE_SIZE = 2.5 #cm
 
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                            0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
 
    objpoints = []
    imgpoints = []
    img_shape = None

    images = glob.glob('calibration_images/*.jpg') + \
             glob.glob('calibration_images/*.png') + \
             glob.glob('calibration_images/*.jpeg')
 
    for fname in sorted(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
 
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    trace_value = float(np.trace(mtx))
    result["camera_matrix_trace"] = round(trace_value, 2)
 


    # ==========================================
    # STEP 2: Image Undistortion
    # ==========================================
    # TODO: Read the target image 'test_images/test_arena.png'
    
    # TODO: Use cv2.undistort() with your calculated mtx and dist to fix the image
    

    raw_image = cv2.imread('test_images/test_arena.jpg')
    if raw_image is None:
        raw_image = cv2.imread('test_arena.jpg')

    undistorted = cv2.undistort(raw_image, mtx, dist, None, mtx)


    # ==========================================
    # STEP 3: ArUco Detection & Pose Estimation
    # ==========================================
    # TODO: Initialize the ArUco detector for DICT_4X4_50
    
    # TODO: Detect markers in the UNDISTORTED image
    
    # TODO: For each detected marker, use cv2.solvePnP() to estimate its pose
    # Hint: You need the real-world 3D coordinates of the marker corners (Marker size is 5.0 cm)
    
    # TODO: Extract the z-distance and x-offset from the translation vector (tvec)
    # Populate the result dictionary in the format: result["markers"]["id_<num>"] = {"distance_z": <val>, "x_offset": <val>}
    
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
 
    corners, ids, rejected = detector.detectMarkers(undistorted)
 
    MARKER_SIZE = 5.0
    half = MARKER_SIZE / 2.0
    marker_3d_edges = np.array([
        [-half, half, 0],
        [ half, half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)
 
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners_2d = corners[i][0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                marker_3d_edges,
                marker_corners_2d,
                mtx,
                dist
            )
 
            if success:
                distance_z = round(float(tvec[2][0]), 1)
                x_offset = round(float(tvec[0][0]), 1)
 
                result["markers"][f"id_{marker_id}"] = {
                    "distance_z": distance_z,
                    "x_offset": x_offset,
                }
 


    # ==========================================
    # SORT MARKERS BY ARUCO ID
    # ==========================================
    result["markers"] = dict(

        sorted(

            result["markers"].items(),

            key=lambda item: int(
                item[0].split("_")[1]
            ),
            reverse=True
        )

    )

    # ==========================================
    # RETURN FINAL OUTPUT
    # ==========================================

    return result

if __name__ == "__main__":
    # Test your function
    output = localize_bot()
    print("Task 2A Output:")
    print(output)
