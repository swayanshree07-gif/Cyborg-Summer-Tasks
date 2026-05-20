#!/usr/bin/env python3
import cv2
import numpy as np


def analyze_arena(input_image):

    # ==========================================
    # LOAD IMAGE
    # ==========================================

    image = cv2.imread(input_image)

    if image is None:

        print("Error loading image.")
        return {}

    # ==========================================
    # INITIALIZE OUTPUT
    # ==========================================

    result = {

        "arena_size": None,
        "start": None,
        "goal": None,
        "special_cells": {}

    }

    # ==========================================
    # WRITE YOUR LOGIC BELOW
    # ==========================================
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
 
    mid_col = w // 2
    mid_row = h // 2

    arena_x1 = next(j for j in range(w)          if gray[mid_row, j] > 100)
    arena_x2 = next(j for j in range(w - 1, -1, -1) if gray[mid_row, j] > 100)
    arena_y1 = next(i for i in range(h)          if gray[i, mid_col] > 100)
    arena_y2 = next(i for i in range(h - 1, -1, -1) if gray[i, mid_col] > 100)
 
    arena_w = arena_x2 - arena_x1 + 1
    arena_h = arena_y2 - arena_y1 + 1
 
    arena_crop = gray[arena_y1:arena_y2 + 1, arena_x1:arena_x2 + 1]
    edges = cv2.Canny(arena_crop, 50, 150)
 
    col_proj = np.sum(edges, axis=0).astype(float)
    col_proj /= (col_proj.max() + 1e-9)
    row_proj = np.sum(edges, axis=1).astype(float)
    row_proj /= (row_proj.max() + 1e-9)
 
    PEAK_THRESH = 0.25
 
    def count_lines(proj):
        peaks = np.where(proj > PEAK_THRESH)[0]
        if len(peaks) == 0:
            return 0
        lines, prev = 1, int(peaks[0])
        for p in peaks[1:]:
            if int(p) - prev > 5:  
                lines += 1
            prev = int(p)
        return lines
 
    n_vert  = count_lines(col_proj)  
    n_horiz = count_lines(row_proj)   

    cols_detected = max(n_vert  - 2 + 1, 1)
    rows_detected = max(n_horiz - 2 + 1, 1)
    raw_size = round((cols_detected + rows_detected) / 2)

    best_size = min([6, 8, 10, 12], key=lambda s: abs(s - raw_size))
 
    result["arena_size"] = best_size
    size   = best_size
    cell_w = arena_w / size
    cell_h = arena_h / size

    COLOR_RANGES = [
        ("RED", np.array([0,   120, 70]),  np.array([10,  255, 255])),
        ("RED2", np.array([170, 120, 70]),  np.array([179, 255, 255])),
        ("GREEN", np.array([40,  100, 100]), np.array([80,  255, 255])),
        ("BLUE", np.array([100, 150, 50]),  np.array([130, 255, 255])),
        ("ORANGE", np.array([10,  120, 120]), np.array([25,  255, 255])),
        ("YELLOW", np.array([20,  100, 100]), np.array([35,  255, 255])),
        ("CYAN", np.array([80,  100, 100]), np.array([100, 255, 255])),
    ]
 
    COLOR_MAP = {
        "RED": "DANGER",
        "RED2": "DANGER",
        "GREEN": "SAFE",
        "BLUE": "REFUEL",
        "ORANGE": "SLOW",
    }

    def get_cell_color(cx, cy, sample_size=20):
        half = sample_size // 2
        region = hsv[cy - half:cy + half, cx - half:cx + half]
        for name, lo, hi in COLOR_RANGES:
            mask = cv2.inRange(region, lo, hi)
            if np.sum(mask > 0) > 10:
                return name
        return None
 
    for row_idx in range(size):
        for col_idx in range(size):
            cx = int(arena_x1 + (col_idx + 0.5) * cell_w)
            cy = int(arena_y1 + (row_idx + 0.5) * cell_h)
 
            color = get_cell_color(cx, cy)
            if color is None:
                continue

            col_letter = chr(ord('A') + col_idx)
            grid_row = size - row_idx
            coord = col_letter + str(grid_row)
 
            if color == "CYAN":
                result["goal"] = coord
            elif color == "YELLOW":
                result["start"] = coord
            elif color in COLOR_MAP:
                result["special_cells"][coord] = COLOR_MAP[color]



    '''
    Steps you may follow:

    1. Detect arena size
    2. Divide arena into grid cells
    3. Convert image to HSV 
    4. Detect START cell     
    5. Detect GOAL cell
    6. Detect special colored cells
    7. Map cells to arena coordinates
    8. Store outputs in result dictionary

    Color Meaninglow
    Red : Danger Zone
    Green : Safe Zone
    Blue : Refuel Station
    Orange : Slow Terrain
    Yellow : Start Position
    Cyan : Goal Position
    '''
    # Example:

    # result["arena_size"] = 8
    # result["start"] = "A1"
    # result["goal"] = "H8"

    # result["special_cells"]["B2"] = "DANGER"
    # result["special_cells"]["D5"] = "SAFE"

    # ==========================================
    # SORT SPECIAL CELLS
    # ==========================================

    sorted_cells = dict(

        sorted(

            result["special_cells"].items(),

            key=lambda item: (

                item[0][0],
                int(item[0][1:])

            )
        )
    )

    result["special_cells"] = sorted_cells

    # ==========================================
    # RETURN FINAL OUTPUT
    # ==========================================

    return result
