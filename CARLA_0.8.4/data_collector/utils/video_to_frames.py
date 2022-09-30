""" Python file to convert MP4 to dataset of image frames  """

import cv2
import sys
import os

if __name__=="main":
    path = sys.argv[1]
    out = sys.argv[2]
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    if not os.path.exists(out):
        os.makedirs(out)
    while success:
        cv2.imwrite(out + "frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

