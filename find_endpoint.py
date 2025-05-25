from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from math import atan2, pi

# "#" initial values"
def find_endpoints(img_pil: Image.Image,
                  *,
                  houth_thresh: int = 25, #50 (threshold)
                  max_len: int = 15, #80 (minLineLength)
                  max_gap: int = 7, #20 (maxLineGap)
                  tol_theta_deg: float = 5.0, 
                  tol_rho_px: float = 20.0, 
                  dot_radius: int = 7,) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:

    # PIL->OpenCV BGR
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    cx, cy = W // 2.0, H // 2.0

    # red mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask |= cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #(5, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    # Hough Transform
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=houth_thresh,
        minLineLength=max_len,
        maxLineGap=max_gap
    )

    if segments is None:
        return img_pil.copy(), []
    
    segs = segments[:, 0]

    # (theta, rho) clustering
    tol_theta = tol_theta_deg * pi / 180
    groups = []
    for x1, y1, x2, y2 in segs:
        theta = atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += pi
        n = np.array([np.sin(theta), -np.cos(theta)])
        rho = n.dot((x1, y1))

        added = False

        for g in groups:
            mean_theta, mean_rho, gsegs =g
            dtheata = min(abs(theta - mean_theta), pi - abs(theta - mean_theta))
            if dtheata < tol_theta and abs(rho - mean_rho) < tol_rho_px:
                gsegs.append((x1, y1, x2, y2))
                k = len(gsegs)
                g[0] = (mean_theta * (k - 1) + theta) / k
                g[1] = (mean_rho * (k - 1) + rho) / k
                added = True
                break
        if not added:
            groups.append([theta, rho, [(x1, y1, x2, y2)]])

    endpoints = []
    for _,_, gsegs in groups:
        pts = np.array([(x1, y1) for x1, y1, _, _ in gsegs] + [(x2, y2) for _, _, x2, y2 in gsegs])
        d2 = np.sum((pts[:, None] - pts[None]) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        (x1, y1), (x2, y2) = pts[i], pts[j]
        endpoints.append((x1 - cx, cy -y1, x2 - cx, cy - y2))


    

    # draw endpoints    
    # colours = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255), (255, 128, 0)]

    vis = bgr.copy()
    for idx, (x1_, y1_, x2_, y2_) in enumerate(endpoints):
        # col = colours[idx % len(colours)]
        col = (255, 0, 0)
        cv2.circle(vis, (int(x1_ + cx), int(cy - y1_)), dot_radius, col, -1)
        cv2.circle(vis, (int(x2_ + cx), int(cy - y2_)), dot_radius, col, -1)

    annot_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return annot_pil, endpoints




img_path = Path("processed_image.jpg")
img_pil = Image.open(img_path)

annotated, pts = find_endpoints(img_pil)

plt.imshow(annotated)
plt.axis('off')
plt.title("Detected Endpoints")
plt.show()