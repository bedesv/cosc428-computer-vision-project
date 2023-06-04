import cv2
import numpy as np
from numpy.linalg import norm
import imutils

MASK_BOUNDS = [(80,70,50), (100,255,255)]
IMAGE_CENTER_BOUNDS = 960
ROPE_WIDTH_UPPER_BOUND = 60
ROPE_WIDTH_LOWER_BOUND = 50

Z_AXIS_MOVE_UP = 1
Z_AXIS_MOVE_DOWN = -1
Y_AXIS_MOVE_RIGHT = 1
Y_AXIS_MOVE_LEFT = -1


class RopeDetector:

    def __init__(self):
        pass

    def get_red_mask(self, frame, bounds):
        """
            Converts the image to HSV, filters to the threshold for the rope,
            then erodes and dilates to join small holes and broken lines
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.inRange(hsv, bounds[0], bounds[1])
        mask = cv2.erode(mask, kernel, iterations=8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        
        return mask

    def get_contour_center(self, contour):
        """
            Calculates the center of the given contour
            using the moments of the contour
        """
        M = cv2.moments(contour)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        return (center_X, center_Y)

    def calculate_rope_width(self, contour, poly):
        """
            Find the center of the rope contour, iterates over each line in 
            the bounding polygon and calculates the distance from that line 
            to the center of the rope contour. The rope with is the sum of 
            the distances to the closest two lines to the center of the rope
        """

        # Find the center of the rope contour
        rope_center = np.asarray(self.get_contour_center(contour))

        # Iterate over each line in the bounding polygon and calculate the distance
        # from that line to the center of the rope contour
        distances_from_rope_center_to_edge = []
        for i in range(len(poly)):
            line = [np.asarray(poly[i-1]), np.asarray(poly[i])]
            distance = norm(np.cross(line[1]-line[0], line[0]-rope_center))/norm(line[1]-line[0])
            distances_from_rope_center_to_edge.append(distance)

        # The rope with is the sum of the distances to the closest 
        # two lines to the center of the rope
        rope_width = sum(sorted(distances_from_rope_center_to_edge)[:2])
        return rope_width
            
        
    def calculate_required_movements(self, rope_center, rope_width, frame_width):
        """
            Takes the center of the rope, the width of the rope, and the width of the frame,
            then calculates the movement in the Y and Z axes.

            The movement in the Z axis is determined by the width of the rope. If it's too big 
            in the frame then move up, if it's too small in the frame then move down.

            The movement in the Y axis is determined by the center of the rope and the width
            of the frame. It keeps the center of the rope within the middle third of the frame.
        """

        # Calculate movement in the Z axis
        movements = [0, 0]
        if rope_width > ROPE_WIDTH_UPPER_BOUND:
            movements[1] = Z_AXIS_MOVE_DOWN
        elif rope_width < ROPE_WIDTH_LOWER_BOUND:
            movements[1] = Z_AXIS_MOVE_UP

        # Calculate movement in the Y axis
        if rope_center[0] < frame_width // 3:
            movements[0] = Y_AXIS_MOVE_RIGHT
        elif rope_center[0] > (frame_width // 3) * 2:
            movements[0] = Y_AXIS_MOVE_LEFT

        return movements
        
    def detect_rope(self, img):

        # Invert and blur the image
        img_inv = cv2.bitwise_not(img)
        blur_img_inv = cv2.GaussianBlur(img_inv, (5, 5), cv2.BORDER_DEFAULT)

        # Get the mask of the detected rope
        mask = self.get_red_mask(blur_img_inv, MASK_BOUNDS)

        # Apply morphological operations to join broken lines and fill holes
        kernel = np.ones((13, 13), np.uint8)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel) 
                
        # Find the contours of the image
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        movements = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, epsilon, True)

            rope_center = self.get_contour_center(c)

            rope_width = self.calculate_rope_width(c, poly)
            
            # Draw the center of the rope and the bounding box on the image
            cv2.circle(img, rope_center, 5, (255, 0, 0))
            cv2.drawContours(img, [poly], -1, (0, 255, 0), 2)
            
            movements = self.calculate_required_movements(rope_center, rope_width, img.shape[1])

        return img, movements

        




if __name__ == "__main__":
    detector = RopeDetector()
    for i in range(2, 3):
        img = cv2.imread(f"picture{i}.png")
        img, movement = detector.detect_rope(img)
        print(movement)
        cv2.imshow("Image", img)
        cv2.waitKey()