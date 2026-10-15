import cv2
from src import segmentation

img = cv2.imread("dataset/images/normal_01.png", 0)
gt = cv2.imread("dataset/ground_truth/normal_01.png", 0) > 0

print("block   C   IoU  recall  precision")
for block in (11, 31, 51, 75, 99):
    for C in (2, 8, 15, 25):
        m = segmentation.segment_adaptive(img, block, C, "gaussian", "darker")["mask"] > 0
        tp = (m & gt).sum()
        print(f"{block:5d} {C:3d} {tp / (m | gt).sum():.2f}   {tp / gt.sum():.2f}   "
              f"{tp / max(m.sum(), 1):.2f}")

# What the default mask looks like (open this to see speckle vs rings)
m = segmentation.segment_adaptive(img, 11, 2, "gaussian", "darker")["mask"]
cv2.imwrite("adaptive_default_normal_01.png", m)