""" Script to adjust frames since one camera on husky recorded twice as many frames"""

import cv2
import sys
import os
import PIL.Image as Image

if __name__=="__main__":
    path = sys.argv[1]
    out = sys.argv[2]
    rgb_data = sorted(os.listdir(path))
    for file_name in rgb_data:
      if "image" in str(file_name):
        frame = file_name.split("image_")
        print(frame[1])
        frame_number = int(frame[1].split(".")[0])
        if frame_number%2 ==0:
          frame_number = int(frame_number/2)
          im = Image.open(path +"/"+ file_name)
          im.save(out +'/' + "image_" + str(frame_number).zfill(5) + '.jpg')