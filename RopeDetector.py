import cv2
import numpy as np
class RopeDetector:

    def __init__():
        pass

    def detect_rope(img):
        # cv2.imshow("", img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## Gen lower mask (0-5) and upper mask (175-180) of RED
        mask1 = cv2.inRange(img_hsv, (0,100,150), (190,255,255))
        # mask2 = cv2.inRange(img_hsv, (175,70,70), (180,255,255))

        ## Merge the mask and crop the red regions
        # mask = cv2.bitwise_or(mask1, mask2 )
        cropped = cv2.bitwise_and(img_hsv, img_hsv, mask=mask1)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        
        cv2.imshow("cropped", cropped)
        cv2.waitKey()

        




if __name__ == "__main__":
    img = cv2.imread("picture10.png")
    RopeDetector.detect_rope(img)