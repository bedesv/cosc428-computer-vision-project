import cv2
import numpy as np
import imutils

RED_MASK_BOUNDS = [(0,100,150), (190,255,255)]
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
        if M["m00"] == 0:
            return None

        return int(M["m10"] / M["m00"]), int(M["m01"] / M[["m00"]])

    def detect_rope(self, img):
        # cv2.imshow("", img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = self.get_red_mask(img, RED_MASK_BOUNDS)

        cropped = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)

        # cv2.imshow("", cropped)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)

        
        if cnts:
            c = max(cnts, key = cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, epsilon, True)

            M = cv2.moments(c)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = (center_X, center_Y)

            (frame_h, frame_w) = img.shape[:2] #w:image-width and h:image-height

            cv2.circle(img, (frame_w//2, frame_h//2), 7, (255, 100, 0), 1) 
            cv2.circle(img, contour_center, 5, (255, 0, 0))
            
            cv2.drawContours(img, [poly], -1, (0, 255, 0), 2)
            
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

            if center_X is not None and center_Y is not None:
                error_x = center_X - frame_w / 2
                error_y = center_Y - frame_h / 2
            else:
                error_x, error_y = 0, 0
    
        cv2.imshow("Image", img)
        
        # cv2.imshow("mask1", mask)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey()

        




if __name__ == "__main__":
    detector = RopeDetector()
    for i in range(2, 11):
        img = cv2.imread(f"picture{i}.png")
        detector.detect_rope(img)
        cv2.waitKey()