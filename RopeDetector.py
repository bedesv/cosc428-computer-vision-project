import cv2
import numpy as np
from numpy.linalg import norm
import imutils

MASK_BOUNDS = [(80,70,50), (100,255,255)]
IMAGE_CENTER_BOUNDS = 960
class RopeDetector:

    def __init__(self):
        pass

    def get_red_mask(self, frame, bounds):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.inRange(hsv, bounds[0], bounds[1])
        mask = cv2.erode(mask, kernel, iterations=8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        
        return mask

    def get_contour_center(self, contour):
    
        M = cv2.moments(contour)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        return (center_X, center_Y)

    def calculate_rope_width(self, contour, poly):
        rope_center = np.asarray(self.get_contour_center(contour))
        distances_from_rope_center_to_edge = []
        for i in range(len(poly)):
            line = [np.asarray(poly[i-1]), np.asarray(poly[i])]
            # print(line)
            distance = norm(np.cross(line[1]-line[0], line[0]-rope_center))/norm(line[1]-line[0])
            distances_from_rope_center_to_edge.append(distance)
        rope_width = sum(sorted(distances_from_rope_center_to_edge)[:2])
        return rope_width
            
        
    def calculate_required_movements(self, rope_center, rope_width, frame_width):

        movements = [0, 0]
        if rope_width > 60:
            movements[1] = rope_width - 60
        elif rope_width < 50:
            movements[1] = rope_width - 50

        if rope_center[0] < frame_width // 3:
            movements[0] = (frame_width // 3) - rope_center[0]
        elif rope_center[0] > (frame_width // 3) * 2:
            movements[0] = -1 * (rope_center[0] - ((frame_width // 3) * 2))

        return movements
        
    def detect_rope(self, img):
        img_inv = cv2.bitwise_not(img)
        blur_img_inv = cv2.GaussianBlur(img_inv, (5, 5), cv2.BORDER_DEFAULT)
        img_hsv = cv2.cvtColor(blur_img_inv, cv2.COLOR_BGR2HSV)
        # img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0) 
        img_gray = cv2.cvtColor(blur_img_inv, cv2.COLOR_BGR2GRAY)

        mask = self.get_red_mask(blur_img_inv, MASK_BOUNDS)

        mask = cv2.dilate(mask, (5, 5))
        mask = cv2.erode(mask, (5, 5))

        cropped = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)

        # cv2.imshow("", cropped)

        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE,
	    cv2.CHAIN_APPROX_SIMPLE)

        movements = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, epsilon, True)

            rope_center = self.get_contour_center(c)

            (frame_h, frame_w) = img.shape[:2] #w:image-width and h:image-height

            rope_width = self.calculate_rope_width(c, poly)
            

            cv2.circle(img, (frame_w//2, frame_h//2), 7, (255, 100, 0), 1) 
            cv2.circle(img, rope_center, 5, (255, 0, 0))
            
            cv2.drawContours(img, [poly], -1, (0, 255, 0), 2)
        
            movements = self.calculate_required_movements(rope_center, rope_width, img.shape[1])

            print(movements)

        return img, movements
    
        # cv2.imshow("Image", img)
        
        # cv2.imshow("mask1", mask)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey()

        




if __name__ == "__main__":
    detector = RopeDetector()
    for i in range(2, 11):
        img = cv2.imread(f"picture{i}.png")
        img, movement = detector.detect_rope(img)
        print(movement)
        cv2.imshow("Image", img)
        cv2.waitKey()