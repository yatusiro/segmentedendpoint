import cv2
import numpy as np


class CascadeClassifier:
    def __init__(
        self,
        cascade_path,
        min_size=(30, 30),
        max_size=(300, 300),
        min_neighbors=3,
        scale_factor=1.1,
    ):
        """
        Initialize the CascadeClassifier with the path to the cascade classifier XML file.
        Args:
            cascade_path: Path to the cascade classifier XML file.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        """
        Detect only one object in the image.
        Args:
            image: Input image in BGR format.

        Returns:
            success: bool, True if an object is detected, False otherwise.
            position: tuple, (x, y) center coordinates of the detected object.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        object = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            maxSize=self.max_size,
        )
        if len(object) > 0:
            # detect only one object
            x, y, w, h = object[0]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            return True, (center_x, center_y)
        else:
            return False, (None, None)