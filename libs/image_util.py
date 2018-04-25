import imutils
import cv2

def resize2squware(img, w, h):
    curr_h, curr_w = img.shape[:2]

    if curr_h < curr_w:
        img = imutils.resize(img, width=w)
    else:
        img = imutils.resize(img, height=h)

    padH = int((h - img.shape[0]) / 2.0)
    padW = int((w - img.shape[1]) / 2.0)

    img = cv2.copyMakeBorder( img
                            , padH, padH, padW, padW
                            , cv2.BORDER_REPLICATE
                            )
    img = cv2.resize(img, (w, h))

    return img

