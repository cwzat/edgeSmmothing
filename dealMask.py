import cv2
import numpy as np

def dealMask(mask_path):
    erodeSize = 6
    blurSize = 3
    mask_ori = cv2.imread(mask_path)
    mask = mask_ori.copy()
    mask[mask_ori != 0] = 0
    mask[mask_ori == 0] = 255
    # for i in mask[:, :, :]:
    #     print(i)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)


    element = np.ones((erodeSize, erodeSize), np.uint8)
    #mask = cv2.erode(mask, element)
    mask = cv2.dilate(mask, element)
    mask = cv2.blur(mask, (blurSize, blurSize))
    mask.astype(np.float)
    # cv2.imshow("mask_deal", mask)
    # cv2.waitKey(0)
    mask = mask / 255.0
    return mask