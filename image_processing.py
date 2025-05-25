from PIL import Image, ImageEnhance
from io import BytesIO
import requests

def processing_image(file_path):
    
   
    img = Image.open(file_path)
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(3)  # Increase contrast
    # enhanced_img.show()  # Display the image
    return enhanced_img


def get_image_coordinates(image):
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    print(f"Image size: {width}x{height}")
    print(f"Image center coordinates: ({center_x}, {center_y})")
    return center_x, center_y

    

import cv2
import numpy as np

def detect_and_highlight_wires(image):

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.show()
    return result_image

def detect_and_highlight_wires_test(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # result_image.show()
    return result_image

def detect_and_highlight_wires_test2(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > 2.5:
            cv2.drawContours(cv_image, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
        
    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.show()
            
    return result_image

def find_incomplete_endpoint(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    center_y = height // 2

    red_mask = cv2.inRange(cv_image, (0, 0, 255), (0, 0, 255))

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    endpoints = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        top = y
        bottom = y + h

        if top > 0 and bottom < height:
            endpoints.append(top - center_y)
            endpoints.append(bottom - center_y)
        elif top> 0:
            endpoints.append(top - center_y)
        elif bottom < height:
            endpoints.append(bottom - center_y)
    

    return endpoints

def mark_closest_endpoints(image, endpoints):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    center_y = height // 2
    center_x = width // 2

    if not endpoints:
        print("No endpoints found.")
        return image
    
    closest_y_offset = min(endpoints, key=lambda y: abs(y))
    closest_y = center_y + closest_y_offset

    size = 3
    top_left = (center_x - size, closest_y - size)
    bottom_right = (center_x + size, closest_y + size)
    cv2.rectangle(cv_image, top_left, bottom_right, (255, 0, 0), thickness=-1)

    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.show()
    return result_image

def mark_closest_endpoints_test(image, endpoints):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    center_y = height // 2
    center_x = width // 2

    size = 2
    for y_offset in endpoints:
        y_img = center_y + y_offset
        top_left = (center_x - size, y_img - size)
        bottom_right = (center_x + size, y_img + size)
        cv2.rectangle(cv_image, top_left, bottom_right, (255, 0, 0), thickness=-1)
    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.show()
    return result_image

def draw_fitted_lines(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    height, width = cv_image.shape[:2]
    red_mask = cv2.inRange(cv_image, (0, 0, 255), (0, 0, 255))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 2:
            continue

        # Fit a line to the contour points
        [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        x1 = 0
        y1 = int(y0 + (y0 * vy) / vx)
        x2 = width - 1
        y2 = int(y0 + ((width - 1 - x0) * vy) / vx)

    result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    result_image.show()
    return result_image

path = "single.jpg"



# get_image_coordinates(processing_image(path))
img = Image.open(path)
get_image_coordinates(img)
# detect_and_highlight_wires_test(processing_image(path))
processed_img = detect_and_highlight_wires_test(img)
# save processed image as jpg
processed_img.save("processed_image.jpg")

# endpoints = find_incomplete_endpoint(processed_img)

# mark_closest_endpoints(processed_img, endpoints)