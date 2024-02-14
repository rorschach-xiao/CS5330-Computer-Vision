import cv2
import numpy as np
from tqdm import tqdm
import os

def main():
    img_dir = './dataset/images'
    label_dir = './dataset/labels'
    max_R, max_G, max_B = -1, -1, -1
    min_R, min_G, min_B = 256, 256, 256
    all_red_vals = np.array([])
    all_green_vals = np.array([])
    for img_name in tqdm(os.listdir(img_dir)):
        # loading data
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.split('.')[0] + '.png')
        # original image
        img = cv2.imread(img_path)
        # ground truth
        mask = cv2.imread(label_path, 0)

        # load all the pixel values in the mask area
        blue_vals = img[:, :, 0][mask == 1]
        green_vals = img[:, :, 1][mask == 1]
        red_vals = img[:, :, 2][mask == 1]
        sum_vals = np.sum(img, axis = 2)[mask == 1]
        sum_vals[sum_vals == 0] = 1

        # rg chromaticity values
        all_red_vals = np.append(all_red_vals, red_vals / sum_vals)
        all_green_vals = np.append(all_green_vals, green_vals / sum_vals)

        area = len(blue_vals)
        blue_vals = sorted(blue_vals)
        green_vals = sorted(green_vals)
        red_vals = sorted(red_vals)

        # use the decile as minimum value
        blue_min = np.median(blue_vals[:area // 10])
        green_min = np.median(green_vals[:area // 10])
        red_min = np.median(red_vals[:area // 10])

        # use the ninth decile as maximum value
        blue_max = np.median(blue_vals[-area // 10:])
        green_max = np.median(green_vals[-area // 10:])
        red_max = np.median(red_vals[-area // 10:])

        max_B = max(max_B, blue_max)
        max_G = max(max_G, green_max)
        max_R = max(max_R, red_max)
        min_B = min(min_B, blue_min)
        min_G = min(min_G, green_min)
        min_R = min(min_R, red_min)

    # # calculate standard deviation and mean for rg chromaticity
    # std_R = np.std(all_red_vals)
    # std_G = np.std(all_green_vals)
    # mean_R = np.mean(all_red_vals)
    # mean_G = np.mean(all_green_vals)

    print(f'==> lower bound:{min_B, min_G, min_R}')
    print(f'==> upper bound:{max_B, max_G, max_R}')
    # print(f'==> standard deviation for red: {std_R}')
    # print(f'==> standard deviation for green: {std_G}')
    # print(f'==> mean for red: {mean_R}')
    # print(f'==> mean for green: {mean_G}')



if __name__ == '__main__':
    main()