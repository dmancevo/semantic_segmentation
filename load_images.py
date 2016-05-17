import numpy as np
import scipy.misc
import os
import re

IM_PATH = "TrainVal/VOCdevkit/VOC2011/JPEGImages/{im_id}.jpg"
SE_PATH = "TrainVal/VOCdevkit/VOC2011/SegmentationClass/{im_id}.png"

images = [re.match(r'[^\.]+',f).group(0) \
  for f in os.listdir("TrainVal/VOCdevkit/VOC2011/SegmentationClass/")]
  

RGB = {
  255: (255,255,255), 0: (0,0,0),
  1: (220,20,60), 2: (255,174,185),
  3: (139,71,93), 4: (255,62,150),
  5: (139,71,137), 6: (75,0,130),
  7: (0,0,128), 8: (100,149,237),
  9: (0,0,255), 10: (0,191,255),
  11: (0,245,255), 12: (0,255,127),
  13: (61,145,64), 14: (0,255,0),
  15: (107,142,35), 16: (255,255,0),
  17: (255,193,37), 18: (255,69,0),
  19: (0,139,139), 20: (139,105,20),
}


def save_side2side(im_id):
  '''Save Image Semantic Segmentation'''
  im = scipy.misc.imread(IM_PATH.format(im_id=im_id))
  se = scipy.misc.imread(SE_PATH.format(im_id=im_id)).astype(np.int)
  
  print im_id
  
  im_se = np.zeros(shape=(se.shape+(3,)))
  
  for i in range(se.shape[0]):
    for j in range(se.shape[1]):
      r, g, b = RGB[se[i][j]]
      im_se[i][j][0] = r
      im_se[i][j][1] = g
      im_se[i][j][2] = b
      
  scipy.misc.imsave("semantic_segmentation.png",np.hstack((im,im_se)))
  
def train_test(im_id):
  '''return train image and ground truth segmentation'''
  im = scipy.misc.imread(IM_PATH.format(im_id=im_id)).astype(np.int)
  se = scipy.misc.imread(SE_PATH.format(im_id=im_id)).astype(np.int)
  
  return im, se
  
  
if __name__ == '__main__':
  save_side2side(np.random.choice(images))
