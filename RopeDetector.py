import cv2
import numpy as np
import imutils
class RopeDetector:

    def __init__():
        pass

    def detect_rope(img):
        # cv2.imshow("", img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ## Gen lower mask (0-5) and upper mask (175-180) of RED
        mask = cv2.inRange(img_hsv, (0,100,150), (190,255,255))


        cropped = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)

        

        c = max(cnts, key = cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, epsilon, True)

        M = cv2.moments(c)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_center = (center_X, center_Y)

        print(img.shape)
        image_center = np.asarray(img.shape)[:2] 
        image_center = tuple(image_center.astype('int32'))
        print(image_center)
        cv2.circle(img, image_center, 10, (255, 100, 0), 2)
        # cv2.circle(img, contour_center, 5, (255, 0, 0))
        
        cv2.drawContours(img, [poly], -1, (0, 255, 0), 2)
            
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    
        
        # cv2.imshow("mask1", mask)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey()

        




if __name__ == "__main__":
    img = cv2.imread("picture10.png")
    RopeDetector.detect_rope(img)