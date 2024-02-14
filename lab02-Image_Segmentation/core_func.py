import cv2
import numpy as np

class Segmentation:
    @staticmethod
    def ThresholdingSegment(img):
        # convert to gray scale image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 271, 1)

        contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # filter edge contours based on 2 conditions:
        # 1. bounding rectangle is nearly a square
        # 2. contour area is larger than 1000 and smaller than 15000
        filtered_contours = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if w > 0 and 0.6 < h / w < 1.4 and 1000 < area < 15000:
                filtered_contours.append(contour)

        # fill contours
        filled_mask = np.zeros_like(img_gray)
        cv2.fillPoly(filled_mask, filtered_contours, 255)

        # apply green mask to mask out some noise
        # the lower bound and upper bound are calculate using calc_thres.py
        lower_green = np.array([3, 20, 30])  # green upper-bound
        upper_green = np.array([75, 93, 101])  # green lower-bound
        mask = cv2.inRange(img, lower_green, upper_green)

        # apply close operation the green mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = np.uint8(mask / 255.0)
        filled_mask = filled_mask * mask

        # use open operation to further denoise
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel)

        # calculate final contours
        final_contours, final_hierarchy = cv2.findContours(np.uint8(filled_mask / 255.0), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filtered_final_contours = []
        for i, contour in enumerate(final_contours):
            area = cv2.contourArea(contour)
            if final_hierarchy[0][i][3] == -1 and area > 200:
                filtered_final_contours.append(contour)

        return np.uint8(filled_mask / 255.0), filtered_final_contours

    @staticmethod
    def EdgeDetectionSegment(img):
        # convert image from BGR to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use canny operator to detect the edge map
        # note: low threshold can help capture more details
        edge_map = cv2.Canny(img_gray, 32, 75, apertureSize=3, L2gradient=True)

        # apply close operation to the edge map
        kernel = np.ones((5, 5), np.uint8)
        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=6)
        ret, thresh = cv2.threshold(edge_map, 80, 255, cv2.THRESH_BINARY)
        edge_contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # filter edge contours based on 3 conditions:
        # 1. bounding rectangle is nearly a square
        # 2. contour area is larger than 1000 and smaller than 15000
        # 3. only the second level contour(inner contour)
        edge_filtered_contours = []
        for i, contour in enumerate(edge_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if w > 0 and 0.6 < h / w < 1.4 and 1000 < area < 15000 and hierarchy[0][i][3] == -1:
                edge_filtered_contours.append(contour)

        # fill contours
        edge_filled_mask = np.zeros_like(img_gray)
        cv2.fillPoly(edge_filled_mask, edge_filtered_contours, 255)

        # # draw contours on original image
        # result = img.copy()
        # cv2.drawContours(result, edge_contours, -1, (0, 255, 255), 3)

        return np.uint8(edge_filled_mask / 255.0), edge_filtered_contours

    @staticmethod
    def RgChromaticitySegment(img, patch_coord):
        def gaussian(p, mean, std):
            return np.exp(-(p - mean) ** 2 / (2 * std ** 2)) * (1 / (std * ((2 * np.pi) ** 0.5)))

        def rg_chromaticity(img, axis):
            img_c = img[:,:,axis]
            img_sum = img.sum(axis=2)
            img_sum[img_sum == 0] = 1
            return img_c / img_sum

        # convert image to rg space
        img_r = rg_chromaticity(img, 2)
        img_g = rg_chromaticity(img, 1)

        # convert patch to rg space
        patch = img[patch_coord[0]:patch_coord[1], patch_coord[2]:patch_coord[3]]
        patch_r = rg_chromaticity(patch, 2)
        patch_g = rg_chromaticity(patch, 1)

        # calculate the stds and means for patch
        mean_r = np.mean(patch_r.flatten())
        mean_g = np.mean(patch_g.flatten())
        std_r = np.std(patch_r.flatten())
        std_g = np.std(patch_g.flatten())

        # generate mask using gaussian distribution
        masked_img_r = gaussian(img_r, mean_r, std_r)
        masked_img_g = gaussian(img_g, mean_g, std_g)
        final_mask = masked_img_r * masked_img_g

        binary_mask = np.uint8(final_mask > final_mask.mean())

        # apply close operation to denoise
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask * 255.0, cv2.MORPH_CLOSE, kernel)

        # use filtered contours to further denoise
        contours, hierarchy = cv2.findContours(np.uint8(binary_mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (w > 30 or h > 30) and 200 < area < 15000 and hierarchy[0][i][3] == -1:
                filtered_contours.append(contour)

        # fill contours
        filled_mask = np.zeros_like(img_r)
        cv2.fillPoly(filled_mask, filtered_contours, 255)

        return np.uint8(filled_mask / 255.0), filtered_contours

