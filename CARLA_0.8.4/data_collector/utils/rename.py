""" Script to rotate images as those recorded by gopro are flipped """

import cv2
import sys
import os
import PIL.Image as Image

if __name__=="__main__":
    path = sys.argv[1]
    out = sys.argv[2]
    rgb_data = sorted(os.listdir(path))
    for file_name in rgb_data:
      print(file_name)
      if "frame" in file_name:
        frame = file_name.split("frame")
        print(frame[1].zfill(9))
        im = Image.open(path + file_name)
        im.save(out +'/' + "image_" + frame[1].zfill(9))
        #print("output")
