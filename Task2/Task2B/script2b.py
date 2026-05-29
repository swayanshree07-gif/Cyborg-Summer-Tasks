import cv2
import numpy as np

def map_arena():
    """
    Task 2B: Perspective Transformation and Coordinate Mapping
    """
    # Initialize the output dictionary
    result = {
        "corner_points_detected": [],
        "robot_pixel_coord": [],
        "robot_real_world_coord": []
    }

    # ==========================================
    # STEP 1: Corner Detection (Color Tracking)
    # ==========================================
    # TODO: Read the target image 'test_images/angled_arena.png'
    
    # TODO: Convert the image to HSV color space
    
    # TODO: Create HSV masks to isolate the Red, Green, Blue, and Yellow corners
    
    # TODO: Find contours for each mask and calculate the centroid (cx, cy) using moments (M["m10"] / M["m00"])
    
    # TODO: Store the coordinates in the exact order: [Top-Left(Red), Top-Right(Green), Bottom-Right(Blue), Bottom-Left(Yellow)]
    # result["corner_points_detected"] = [[cx_r, cy_r], [cx_g, cy_g], [cx_b, cy_b], [cx_y, cy_y]]


    img = cv2.imread('test_images/angled_arena.png')
    if img is None:
        img = cv2.imread('angled_arena.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "green": [
            (np.array([40, 50, 50]), np.array([90, 255, 255])),
        ],
        "blue": [
            (np.array([100, 80, 50]), np.array([130, 255, 255])),
        ],
        "yellow": [
            (np.array([20, 100, 100]), np.array([40, 255, 255])),
        ],
    }
 
    def get_centroid(color_name):
      
        ranges = color_ranges[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
 
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return [cx, cy]

    red_pt = get_centroid("red") #Top-Left
    green_pt = get_centroid("green") #Top-Right
    blue_pt = get_centroid("blue") #Bottom-Right
    yellow_pt = get_centroid("yellow") #Bottom-Left
 
    result["corner_points_detected"] = [red_pt, green_pt, blue_pt, yellow_pt]
 
 
 
    # ==========================================
    # STEP 2: Perspective Transformation
    # ==========================================
    # TODO: Define your source points as a float32 numpy array (the 4 centroids calculated above)
    
    # TODO: Define your destination points as a flat 500x500 pixel square
    # Example: [[0,0], [500,0], [500,500], [0,500]]
    
    # TODO: Use cv2.getPerspectiveTransform() to calculate the 3x3 Homography Matrix
    
    # TODO: Apply cv2.warpPerspective() to flatten the angled arena into a 500x500 top-down view


    OUTPUT_SIZE = 500
    pts_src = np.float32([red_pt, green_pt, blue_pt, yellow_pt])
    pts_dst = np.float32([
        [0, 0], #Top-Left
        [OUTPUT_SIZE, 0], #Top-Right
        [OUTPUT_SIZE, OUTPUT_SIZE], #Bottom-Right
        [0, OUTPUT_SIZE], #Bottom-Left
    ])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    flat_img = cv2.warpPerspective(img, matrix, (OUTPUT_SIZE, OUTPUT_SIZE))
    cv2.imwrite('birdsview.png', flat_img)
 

    # ==========================================
    # STEP 3: Robot Detection on Warped Arena
    # ==========================================
    # TODO: On the NEW warped 500x500 image, initialize an ArUco detector (DICT_4X4_50)
    
    # TODO: Detect the marker representing the robot (ID 1)
    
    # TODO: Calculate the center pixel coordinates (cx, cy) of the detected marker
    # result["robot_pixel_coord"] = [cx, cy]
    
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
 
    corners, ids, _ = detector.detectMarkers(flat_img)
 
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 1:
                # Calculate center pixel of the marker
                marker_corners = corners[i][0]  # shape (4, 2)
                cx = int(np.mean(marker_corners[:, 0]))
                cy = int(np.mean(marker_corners[:, 1]))
                result["robot_pixel_coord"] = [cx, cy]
                break
 


    # ==========================================
    # STEP 4: Real-World Coordinate Conversion
    # ==========================================
    # Context: The 500x500 pixel warped image represents a physical arena of 200cm x 200cm.
    # The top-left corner is the origin [0.0, 0.0] cm.
    
    # TODO: Use linear scaling to convert the robot's pixel coordinates to real-world centimeters.
    # result["robot_real_world_coord"] = [x_cm, y_cm]
    

    ARENA_CM = 200.0
    scale = ARENA_CM / OUTPUT_SIZE
 
    cx_px, cy_px = result["robot_pixel_coord"]
    x_cm = round(cx_px * scale, 1)
    y_cm = round(cy_px * scale, 1)
    result["robot_real_world_coord"] = [x_cm, y_cm]


    return result

if __name__ == "__main__":
    # Test your function
    output = map_arena()
    print("Task 2B Output:")
    print(output)
