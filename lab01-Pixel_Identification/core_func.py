import cv2
import numpy as np

class Segmentation:
    @staticmethod
    def HSVThresholdingSegment(img):
        height, width, channel = img.shape
        # convert image from BGR space to HSV space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get blue and red area
        mask_blue = np.int8(img_hsv[:,:,0] >= 105) & np.int8(img_hsv[:,:,0] <= 135)
        mask_red = np.int8(img_hsv[:,:,0] <= 30) | (np.int8(img_hsv[:,:,0] >= 165) & np.int8(img_hsv[:,:,0] < 180))
        mask = mask_red | mask_blue

        # close morpho
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask * 255.0, cv2.MORPH_CLOSE, kernel)

        # find contours
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img_hsv, contours, -1, (0,255,0), 3)

        # filter contours that only exist on the top
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if y < height // 4:
                filtered_contours.append(contour)

        # apply convex hull to all the contours
        use_convex = False
        if use_convex:
            convex_hulls = [cv2.convexHull(contour) for contour in filtered_contours]
        else:
            convex_hulls = filtered_contours

        # fill contours
        filled_mask = np.zeros_like(mask)
        cv2.fillPoly(filled_mask, convex_hulls, 255)


        return np.uint8(filled_mask / 255.0)

    @staticmethod
    def EdgeDetectionSegment(img):
        height, width, channel = img.shape
        # convert image from BGR to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # y-direction sobel edge detection
        dx = 0  # dx
        dy = 1  # dy
        ksize = 3  # kernel size
        edges_sobel = cv2.Sobel(img_gray, cv2.CV_64F, dx, dy, ksize)
        edge_map = cv2.convertScaleAbs(edges_sobel)

        # fill the top/left/right borders of the edge map with 255
        edge_map[0,:] = 255
        edge_map[:, 0] = 255
        edge_map[:,-1] = 255

        # dilate edge map
        kernel = np.ones((3, 3), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=6)
        ret, thresh = cv2.threshold(edge_map, 80, 255, cv2.THRESH_BINARY)
        edge_contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # filter edge contours based on 2 conditions:
        # 1. only exist on the top
        # 2. only the second level contour(inner contour)
        edge_filtered_contours = []
        for i, contour in enumerate(edge_contours):
            x, y, w, h = cv2.boundingRect(contour)
            if 0 <= y <= 30 and hierarchy[0][i][3] != -1:
                edge_filtered_contours.append(contour)

        # fill contours
        edge_filled_mask = np.zeros_like(img_gray)
        cv2.fillPoly(edge_filled_mask, edge_filtered_contours, 255)
        # morpho filled mask
        edge_filled_mask = cv2.dilate(edge_filled_mask, kernel, iterations=7)

        # use close operation to fill inner holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(5):
            edge_filled_mask = cv2.morphologyEx(edge_filled_mask, cv2.MORPH_CLOSE, kernel)

        # find inner hole and fill them
        all_contours, _ = cv2.findContours(edge_filled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        hole_contours = []
        for cnt in all_contours:
            if cv2.contourArea(cnt) < 20000:
                hole_contours.append(cnt)
        final_mask = np.zeros_like(edge_filled_mask)
        cv2.fillPoly(final_mask, hole_contours, 255)
        final_mask = cv2.bitwise_or(final_mask, edge_filled_mask)

        # draw contours on original image
        result = img.copy()
        cv2.drawContours(result, all_contours, -1, (0, 255, 255), 3)

        return np.uint8(final_mask / 255.0)

