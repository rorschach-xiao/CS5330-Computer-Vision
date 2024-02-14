import cv2
import numpy as np
import os
import core_func
from tqdm import tqdm
import json

# evaluate our algorithm on the dataset
def eval():
    img_dir = './dataset/images'
    label_dir = './dataset/labels'
    out_dir = './result'
    os.makedirs(out_dir, exist_ok=True)
    TP_1, FN_1, FP_1 = 0, 0, 0  # True Position/False Negative/False Positive for thresholding based method
    TP_2, FN_2, FP_2 = 0, 0, 0  # True Position/False Negative/False Positive for edge detection based method
    TP_3, FN_3, FP_3 = 0, 0, 0  # True Position/False Negative/False Positive for rg chromaticity based method

    # load image patch (only used in rg chromaticity segmentation)
    with open('./dataset/patch_coord.json', 'r') as f:
        patch_coord = json.load(f)

    for img_name in tqdm(os.listdir(img_dir)):
        # loading data
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.split('.')[0] + '.png')
        # original image
        img = cv2.imread(img_path)
        # ground truth
        label = cv2.imread(label_path, 0)

        # exec the three segmentation functions
        pred_mask_thres, _ = core_func.Segmentation.ThresholdingSegment(img)
        pred_mask_edge, _ = core_func.Segmentation.EdgeDetectionSegment(img)
        pred_mask_rg, _ = core_func.Segmentation.RgChromaticitySegment(img, patch_coord[img_name])

        # visualization
        visuals = [pred_mask_thres, pred_mask_edge, pred_mask_rg, label]
        concat_imgs = [img]
        # concat the all the visual result
        for result in visuals:
            if len(result.shape) == 2:
                result = np.repeat(result[:,:,None], 3, axis=2) * 255
            concat_imgs.append(result)
        concat_results = np.concatenate(concat_imgs, axis=1)

        # save the visualization result
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, concat_results)

        # calculate metrics
        TP_1 += np.sum(cv2.bitwise_and(pred_mask_thres, label))
        FN_1 += np.sum(cv2.bitwise_and(label, cv2.bitwise_not(pred_mask_thres)))
        FP_1 += np.sum(cv2.bitwise_and(pred_mask_thres, cv2.bitwise_not(label)))

        TP_2 += np.sum(cv2.bitwise_and(pred_mask_edge, label))
        FN_2 += np.sum(cv2.bitwise_and(label, cv2.bitwise_not(pred_mask_edge)))
        FP_2 += np.sum(cv2.bitwise_and(pred_mask_edge, cv2.bitwise_not(label)))

        TP_3 += np.sum(cv2.bitwise_and(pred_mask_rg, label))
        FN_3 += np.sum(cv2.bitwise_and(label, cv2.bitwise_not(pred_mask_rg)))
        FP_3 += np.sum(cv2.bitwise_and(pred_mask_rg, cv2.bitwise_not(label)))

    # calculate IOU = TP / (TP + FN + FP)
    IOU_thres = TP_1 / (TP_1 + FN_1 + FP_1)
    IOU_edge = TP_2 / (TP_2 + FN_2 + FP_2)
    IOU_rg = TP_3 / (TP_3 + FN_3 + FP_3)

    print(f'IOU for thresholding-based method: {round(IOU_thres,4)}')
    print(f'IOU for edge detection-based method: {round(IOU_edge, 4)}')
    print(f'IOU for rg-chromaticity-based method: {round(IOU_rg, 4)}')


if __name__ == '__main__':
    eval()