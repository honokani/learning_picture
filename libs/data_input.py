import cv2, os, re, glob, math
import numpy as np
from imutils import paths
from .image_util import resize2squware
#from .image_util import resize2squware

def getImages(picDirPath, label):
    tgtDir = os.path.join(picDirPath, label)
    images = glob.glob(os.path.join(tgtDir, "*"))
    datas = []
    labels = []
    c=0

    for img in paths.list_images(tgtDir):
        # Load the image
        img = cv2.imread(img)
        # Resize the image to 20x20 squware
        img = resize2squware(img, 20, 20)
        # Add a third channel dimension to the image to make Keras happy
        img = np.expand_dims(img, axis=2)
        datas.append(img)
        labels.append(label)
        if(0 < c):
            break
        c = c+1

    return datas, labels

def getDataset(picsetDirPath):
    ls = os.listdir(picsetDirPath)
    datas = []
    labels = []

    for f in ls:
        picDirPath = os.path.join(picsetDirPath, f)
        if( os.path.isdir(picDirPath) ):
            if(not 1 < len(f)):
                ds,ls = getImages(picsetDirPath, f)
                datas.extend(ds)
                labels.extend(ls)

    return datas, labels


def main():
    COMMON_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    b = os.path.dirname(COMMON_DIR_PATH)
    CAPTCHA_PICS_DIR_NAME = "pics"
    CAPTCHA_PICS_DIR_PATH = os.path.join(b, CAPTCHA_PICS_DIR_NAME)
    return getDataset(CAPTCHA_PICS_DIR_PATH)

if __name__ == '__main__':
    main()

