import os.path

import core_func
import cv2
import argparse
import numpy as np
import core_func

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-thres', action='store_true', help='use thresholding based method.')
    parser.add_argument('-edge', action='store_true', help='use edge detection based method.')
    parser.add_argument('-rg', action='store_true', help='use rg-chromaticity based method.')
    parser.add_argument('--image_path', type=str, help='input image path')
    parser.add_argument('--out_dir', type=str, help='output image directory')
    parser.add_argument('--patch_coord', type=str, help='reference patch coordinate for rg chromaticity method.')
    return parser.parse_args()

# sky replacement inference
def infernece(img_path, out_dir, seg_func, patch_coord=None):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    os.makedirs(out_dir, exist_ok=True)

    if seg_func == core_func.Segmentation.RgChromaticitySegment:
        mask, contours = core_func.Segmentation.RgChromaticitySegment(img, patch_coord)
    else:
        mask, contours = seg_func(img)
    h, w, c = img.shape

    for i, contour in enumerate(contours):
        print(f'==> Leaf No.{i} area: {cv2.contourArea(contour)}')
        # draw index on the image
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img, f"No.{i}", (x + w // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    print(f'==> Image Area: {h * w}')
    out_path = os.path.join(out_dir, img_name)
    print(f'==> Save output image to: {out_path}')
    # blending
    alpha = 0.8
    img = img * (1 - mask[:,:,None]) + (img * alpha + np.ones_like(img) * 255 * (1-alpha)) * mask[:,:,None]
    cv2.imwrite(out_path, img)

if __name__ == '__main__':
    args = parse_args()
    if args.thres:
        infernece(args.image_path, args.out_dir, core_func.Segmentation.ThresholdingSegment)
    elif args.edge:
        infernece(args.image_path, args.out_dir, core_func.Segmentation.EdgeDetectionSegment)
    elif args.rg:
        patch_coord = [int(i) for i in args.patch_coord.split(',')]
        infernece(args.image_path, args.out_dir, core_func.Segmentation.RgChromaticitySegment, patch_coord)


