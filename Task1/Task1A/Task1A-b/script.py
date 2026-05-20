import cv2
import numpy as np


def analyze_video(video_path):

    # ==========================================
    # OUTPUT DICTIONARY
    # ==========================================

    result = {

        "top_wall_hits": 0,
        "bottom_wall_hits": 0,
        "left_wall_hits": 0,
        "right_wall_hits": 0

    }

    # ==========================================
    # OPEN VIDEO
    # ==========================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("Error opening video")
        return result

    # ==========================================
    # GREEN COLOR RANGE (HSV)
    # ==========================================

    # Students may modify/tune these values

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([85, 255, 255])

    # ==========================================
    # FRAME DIMENSIONS
    # ==========================================

    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ==========================================
    # WRITE YOUR LOGIC BELOW
    # ==========================================

    all_color_ranges = [
        (np.array([40, 80, 80]),  np.array([85, 255, 255])),
        (np.array([0, 120, 100]), np.array([10, 255, 255])), 
        (np.array([170, 120, 100]), np.array([179, 255, 255])), 
        (np.array([100, 100, 100]), np.array([130, 255, 255])), 
        (np.array([25, 100, 100]), np.array([35, 255, 255])),
    ]
 
    all_x, all_y = [], []
 
    for _ in range(60):
        ret_s, frame_s = cap.read()
        if not ret_s:
            break
        hsv_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2HSV)
        for lo, hi in all_color_ranges:
            m  = cv2.inRange(hsv_s, lo, hi)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) > 300:
                    Mc = cv2.moments(c)
                    if Mc["m00"] > 0:
                        all_x.append(int(Mc["m10"] / Mc["m00"]))
                        all_y.append(int(Mc["m01"] / Mc["m00"]))
 
    WALL_L = min(all_x) if all_x else 0
    WALL_R = max(all_x) if all_x else WIDTH - 1
    WALL_T = min(all_y) if all_y else 0
    WALL_B = max(all_y) if all_y else HEIGHT - 1
 
    wall_threshold = 26

    left_collision = False
    right_collision = False
    top_collision = False
    bottom_collision = False

    WARMUP_FRAMES = 10

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
    frame_num = 0
 
    while True:
        ret, frame = cap.read() 
        if not ret:
            break 
        frame_num += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
 
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
 
        if cv2.contourArea(largest) < 500:
            continue

        M = cv2.moments(largest)
 
        if M["m00"] == 0:
            continue
 
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        counting = frame_num > WARMUP_FRAMES

        if cx <= WALL_L + wall_threshold:
            if not left_collision and counting:
                left_collision = True
                result["left_wall_hits"] += 1
        else:
            left_collision = False

        if cx >= WALL_R - wall_threshold:
            if not right_collision and counting:
                right_collision = True
                result["right_wall_hits"] += 1
        else:
            right_collision = False

        if cy <= WALL_T + wall_threshold:
            if not top_collision and counting:
                top_collision = True
                result["top_wall_hits"] += 1
        else:
            top_collision = False

        if cy >= WALL_B - wall_threshold:
            if not bottom_collision and counting:
                bottom_collision = True
                result["bottom_wall_hits"] += 1
        else:
            bottom_collision = False


        '''
        Suggested Steps:

        1. Find the largest contour
        2. Ignore very small contours
        3. Compute contour centroid
        4. Detect wall collisions
        5. Count:
           - top wall hits
           - bottom wall hits
           - left wall hits
           - right wall hits

        '''

        # ==========================================
        # HINT FOR WALL DETECTION
        # ==========================================

        '''
        Example:

        if cx <= wall_threshold:
            # Left wall touched

        if cx >= WIDTH - wall_threshold:
            # Right wall touched

        if cy <= wall_threshold:
            # Top wall touched

        if cy >= HEIGHT - wall_threshold:
            # Bottom wall touched
        '''

    # ==========================================
    # RELEASE VIDEO
    # ==========================================

    cap.release()

    return result
