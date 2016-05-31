from __future__ import print_function
import collections
from datetime import datetime
import re
import os

import scipy.misc
import numpy as np
import tensorflow as tf

from config import IM_PATH, SE_PATH, TRAIN_VAL_PATH, MAX_HEIGHT, MAX_WIDTH


IM_FILENAME = "{im_id}.jpg"
SE_FILEMANE = "{im_id}.png"

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

RGB = {
    0: (0,0,0), 1: (128, 0, 0),
    2: (0, 128, 0), 3: (128, 128, 0),
    4: (0, 0, 128), 5: (128, 0, 128),
    6: (0, 128, 128), 7: (128, 128, 128),
    8: (64, 0, 0), 9: (192, 0, 0),
    10: (64, 128, 0), 11: (192, 128, 0),
    12: (64, 0, 128), 13: (192, 0, 128),
    14: (64, 128, 128), 15: (192, 128, 128),
    16: (0, 64, 0), 17: (128, 64, 0),
    18: (0, 192, 0), 19: (128, 192, 0),
    20: (0, 64, 128)
}

class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, image_ids):
        self._image_ids = np.array(image_ids, dtype=object)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.num_examples = len(image_ids)


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self._image_ids = self._image_ids[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self._index_in_epoch
        image_ids_batch = self._image_ids[start:end]

        images = []
        segm = []
        width = 1000
        height = 1000
        for im_id in image_ids_batch.tolist():
            #load images
            im = scipy.misc.imread(IM_PATH + IM_FILENAME.format(im_id=im_id)).astype(np.float)
            se = scipy.misc.imread(SE_PATH + SE_FILEMANE.format(im_id=im_id), mode='P').astype(np.int)
            images.append(im)
            segm.append(se)
            se[se==255] = 0
            h, w, _ = im.shape
            height = min(h, height)
            width = min(w, width)

        height = min(height, MAX_HEIGHT)
        width = min(width, MAX_WIDTH)
        #crop to the smallest image in batch
        images, segm = DataSet.crop(images, segm, height, width)

        return [images, segm]

    @staticmethod
    def crop(imgs, sgs, h, w):
        '''Random crop to min height and with'''
        croped_imgs = []
        croped_sgs = []
        for im, se in zip(imgs, sgs):
            height, width, _ = im.shape

            bottom = 0 if height == h else np.random.choice(range(0, height - h))
            left = 0 if width == w else np.random.choice(range(0, width - w))

            se = se[bottom:(bottom + h), left:(left + w)]
            im = im[bottom:(bottom + h), left:(left + w),:]

            croped_imgs.append(im)
            croped_sgs.append(se)
        
        return croped_imgs, croped_sgs


def load_one_image(im_id):
    #load images
    im = scipy.misc.imread(IM_PATH + IM_FILENAME.format(im_id=im_id)).astype(np.float)
    se = scipy.misc.imread(SE_PATH + SE_FILEMANE.format(im_id=im_id), mode='P').astype(np.int)
    se[se==255] = 0
    im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
    se = np.reshape(se, (1, se.shape[0], se.shape[1]))
    return im, se

def save_image(im_id, se, global_step, path):
    '''Save Image Semantic Segmentation'''

    se = se[0]
    rgb_segm = np.zeros(shape=(se.shape[0], se.shape[1], 3))
    
    for i in range(rgb_segm.shape[0]):
      for j in range(rgb_segm.shape[1]):
        r, g, b = RGB[se[i][j]]
        rgb_segm[i][j][0] = r
        rgb_segm[i][j][1] = g
        rgb_segm[i][j][2] = b

    im = scipy.misc.imread(IM_PATH + IM_FILENAME.format(im_id=im_id))
    se = scipy.misc.imread(SE_PATH + SE_FILEMANE.format(im_id=im_id))
    
    scipy.misc.imsave(path + '/segmentation_step_%s_%d.jpg'%(im_id,global_step) ,np.hstack((im,se,rgb_segm)))


def split_into_train_test(train_frac=None):
    train_ids = []
    test_ids = []
    if train_frac is None:
        with open(TRAIN_VAL_PATH + "train.txt", "r") as f:
            train_ids = map(lambda x: x.strip('\n'), f.readlines())
        with open(TRAIN_VAL_PATH + "val.txt", "r") as f:
            test_ids = map(lambda x: x.strip('\n'), f.readlines())
    else:

        with open(TRAIN_VAL_PATH + "trainval.txt", "r") as f:
            trainval_ids = map(lambda x: x.strip('\n'), f.readlines())
            train_len = int(train_frac * len(trainval_ids))
            train_ids = trainval_ids[:train_len]
            test_ids = trainval_ids[train_len:]
    return train_ids, test_ids


def get_datasets(train_frac=0.75):
    train_ids, test_ids = split_into_train_test(train_frac)
    train_data = DataSet(train_ids)
    test_data = DataSet(test_ids)
    return train_data, test_data

if __name__ == '__main__':

    start = datetime.now()
    train_data, test_data = get_datasets(0.75)
    batch_size = 20
    for i in range(10):
        print(i)
        train_batch = train_data.next_batch(batch_size)
        print(train_batch[0][0].shape, train_batch[1][0].shape)
    print(datetime.now()-start)
