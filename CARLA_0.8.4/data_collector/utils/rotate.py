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
        im = Image.open(path + file_name)
        angle = -90
        output = im.rotate(angle)
        output.save(out +'/' + file_name)
        print("output")


