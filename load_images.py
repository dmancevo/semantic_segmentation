import numpy as np
import scipy.misc
import os
import re
import threading
import Image

class Data:
  
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
  
  def __init__(self, train=0.75):
    
    #Training set size
    self.train = set(np.random.choice(Data.images,
      size=int(train*float(len(Data.images)))))
      
    self.batch = None
    self.batch_ready = False
    
  @staticmethod
  def crop(im, se):
    '''Random crop to min height and with'''
    
    width, height = se.shape
    
    left   = np.random.choice(range(0,width-111))
    bottom = np.random.choice(range(0,height-173))
    
    se = se[left:(left+112),bottom:(bottom+174)]
    im = im[left:(left+112),bottom:(bottom+174),:]
    
    return im, se
    
  def get_batch(self, n, train=True):
    
    if self.batch is None:
      self.next_batch(n,train)
      self.batch_ready = False
      t1 = threading.Thread(target=self.next_batch, kwargs={'n':n, 'train':train})
      t1.start()
      return self.batch
    else:
      while not self.batch_ready:
        continue
      self.batch_ready = False
      t1 = threading.Thread(target=self.next_batch, kwargs={'n':n, 'train':train})
      t1.start()
      return self.batch
      
  def next_batch(self,n,train=True):
    
    imgs, segmentations = [], []
    batch = set()
    i=0
    while len(imgs) < n and i<2*len(Data.images):
      i+=1
      
      im_id = np.random.choice(Data.images)
      
      if im_id in batch:
        continue
      
      if train and im_id not in self.train:
        continue
      elif not train and im_id in self.train:
        continue
      
      #Semantic segmentation ground truth
      se = scipy.misc.imread(Data.SE_PATH.format(im_id=im_id)).astype(np.int)
        
      #Load image
      im = scipy.misc.imread(Data.IM_PATH.format(im_id=im_id)).astype(np.float)
      
      #Crop
      im, se = Data.crop(im, se)
      
      #Black is the new white
      se[se==255] = 0
      
      batch.add(im_id)
      imgs.append(im)
      segmentations.append(se.astype(int))
      
    self.batch = [imgs, segmentations]
    self.batch_ready = True

  @classmethod
  def save_side2side(cls, im_id, net_output, title="semantic_segmentation_example.png"):
    '''Save Image Semantic Segmentation'''
    im = scipy.misc.imread(cls.IM_PATH.format(im_id=im_id))
    se = scipy.misc.imread(cls.SE_PATH.format(im_id=im_id)).astype(np.int)
    
    print im_id
    
    im_se = np.zeros(shape=(se.shape+(3,)))
    
    for i in range(se.shape[0]):
      for j in range(se.shape[1]):
        r, g, b = cls.RGB[se[i][j]]
        im_se[i][j][0] = r
        im_se[i][j][1] = g
        im_se[i][j][2] = b
        
    im_hat = np.zeros(shape=im_se.shape)
    for i in range(se.shape[0]):
      for j in range(se.shape[1]):
        r, g, b = cls.RGB[net_output[i][j]]
        im_hat[i][j][0] = r
        im_hat[i][j][1] = g
        im_hat[i][j][2] = b
        
    scipy.misc.imsave(title,np.hstack((im,im_se,im_hat)))
    
  @classmethod
  def get_image(cls, im_id=None):
    '''Return image id, image and semantic segmentation ground truth'''
    
    if im_id is None:
      im_id = np.random.choice(cls.images)
    se = scipy.misc.imread(cls.SE_PATH.format(im_id=im_id)).astype(np.int)
    im = scipy.misc.imread(cls.IM_PATH.format(im_id=im_id)).astype(np.float)
    
    return im_id, im, se
    
  @staticmethod
  def get_crop(im_id=None):
  
    im_id, im, se = Data.get_image(im_id)
    im, se = Data.crop(im, se)
  
    return im_id, im, se
  
  
if __name__ == '__main__':
  
  from datetime import datetime

  # save_side2side(np.random.choice(Data.images))
  
  data_set = Data()
  # print datetime.now()
  # for _ in range(10):
    
  #   batch = data_set.get_batch(20, train=True)

  # print datetime.now()
  
  # batch = data_set.get_batch(7, train=False)
  
  # print batch[0][0].shape, batch[1][0].shape
  # print batch[0][1].shape, batch[1][1].shape
  
  print scipy.misc.imread("TrainVal/VOCdevkit/VOC2011/SegmentationClass/2011_001967.png").astype(np.int).shape
  