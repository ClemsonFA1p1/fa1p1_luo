""" Resizes images collected from husky and aligns the camera frames by reducing the left view to every other frame"""


import cv2
import sys
import os
import PIL.Image as Image

if __name__=="__main__":
    path = sys.argv[1]
    out = sys.argv[2]
    rgb_data = sorted(os.listdir(path))
    for file_name in rgb_data:
      image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_COLOR)
      image = cv2.resize(image, (200, 88), interpolation=cv2.INTER_CUBIC)    
      cv2.imwrite(out + '/' + file_name, image)
        
