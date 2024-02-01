import cv2
import numpy as np
import os
import core_func
from tqdm import tqdm
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-thres',help='use thresholding based method',action='store_true')
    parser.add_argument('-edge',help='use edge detection based method', action='store_true')
    return parser.parse_args()

# evaluate our algorithm on the dataset
def eval(seg_func=core_func.Segmentation.EdgeDetectionSegment):
    img_dir = './dataset/images'
    label_dir = './dataset/labels'
    out_dir = './result'
    os.makedirs(out_dir, exist_ok=True)
    TP, FN, FP = 0, 0, 0  # True Position/False Negative/False Positive
    for img_name in tqdm(os.listdir(img_dir)):
        # loading data
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.split('.')[0] + '.png')
        # original image
        img = cv2.imread(img_path)
        # ground truth
        label = cv2.imread(label_path, 0)

        # exec the segmentation function
        pred_mask = seg_func(img)

        # visualization
        visuals = [pred_mask, label]
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

        TP += np.sum(cv2.bitwise_and(pred_mask, label))
        FN += np.sum(cv2.bitwise_and(label, cv2.bitwise_not(pred_mask)))
        FP += np.sum(cv2.bitwise_and(pred_mask, cv2.bitwise_not(label)))

    # calculate IOU = TP / (TP + FN + FP)
    IOU = TP / (TP + FN + FP)
    print(f'IOU: {round(IOU,4)}')


# sky replacement inference
def infernece(original_img, sky_img, seg_func_name):
    print(seg_func_name)
    if seg_func_name[0] == 'HSV Thresholding':
        seg_func = core_func.Segmentation.HSVThresholdingSegment
    else:
        seg_func = core_func.Segmentation.EdgeDetectionSegment
    h, w, _ = original_img.shape
    bg_h, bg_w, _ = sky_img.shape

    # resize sky image
    sky_img_resize = cv2.resize(sky_img, (w, h), interpolation = cv2.INTER_LINEAR)

    # generate sky mask
    sky_mask = seg_func(original_img)
    sky_mask = sky_mask[:,:,None]

    # replace sky area
    result = original_img * (1 - sky_mask) + sky_img_resize * sky_mask

    return [np.repeat(sky_mask,3, axis=2) * 255, result]


if __name__ == '__main__':
    args = parse_arg()
    if args.thres:
        eval(core_func.Segmentation.HSVThresholdingSegment)
    else:
        eval(core_func.Segmentation.EdgeDetectionSegment)