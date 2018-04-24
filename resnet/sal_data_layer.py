import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
from PIL import Image
import scipy.io as io
import cv2

import random

class SalDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for PASCAL VOC semantic segmentation.
        example
        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.data_root = params.get('data_root', '')
        self.source = '/Data/Existence/train.lst'
        #self.data_root = params.get('data_root', '/Data/instance/sal_inst/single/')
        #self.source = '/Data/instance/sal_inst/train.lst'
        self.random = True

        #self.source = params.get('source', '/Data/weak_seg/low-level/VOC2012/val_region.txt')
        self.mean = np.array(params.get('mean', [104.070, 116.669, 122.679]))
        self.seed = params.get('seed', None)
        self.seed = random.randint(0, 1024)
        #self.scales = params.get('scales', [0.5, 0.75, 1, 1.5])

        # two tops: data and label
        #if len(top) != 3: # image + pixel-level annos + estimated regions
        #    raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir, self.split)
        self.indices = open(self.source, 'r').read().splitlines()
        print 'Totally {} training images'.format(len(self.indices))
        self.idx = 0
        self.image_lst = [x.split()[0] for x in self.indices]
        self.label_lst = [x.split()[1] for x in self.indices]
        self.sal_lst = [x.split()[2] for x in self.indices]

        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

        self.flip = 1


    def reshape(self, bottom, top):
        #if self.scales != None:
        #    self.scale_ind = random.randint(0, len(self.scales)-1)

        self.flip = random.randint(0, 1)

        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        self.sal = int(self.sal_lst[self.idx])

        # random crop
        #height, width = self.data.shape[-2:]
        #h_start = random.randint(0, np.abs(self.crop_size - height))
        #w_start = random.randint(0, np.abs(self.crop_size - width))
        #self.data = self.data[:, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size]
        #self.label = self.label[:, h_start:h_start+self.crop_size, w_start:w_start+self.crop_size]

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, 1, 1, 1)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.sal

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = cv2.imread(self.data_root + self.image_lst[idx], cv2.IMREAD_COLOR)
        #im = Image.open(self.data_root + self.image_lst[idx])
        im = np.array(im, dtype=np.float32)
        #self.height, self.width = im.shape[:2]
        #if self.scales != None:
        #    im = cv2.resize(im, None, None, fx=self.scales[self.scale_ind], fy=self.scales[self.scale_ind], interpolation=cv2.INTER_CUBIC)

        im = im[:,:,::-1]
        im -= self.mean
        #self.h_off = random.randint(0, int(self.scales[self.scale_ind] * self.height - self.height))
        #self.w_off = random.randint(0, int(self.scales[self.scale_ind] * self.width - self.width))
        #in_ = cv2.copyMakeBorder(in_, 0, max(0, h_off), 0, max(0, w_off), cv2.BORDER_CONSTANT, value=[0,0,0])
        #im = im[self.h_off:self.h_off+self.height, self.w_off:self.w_off+self.width, :]
        im = im.transpose((2,0,1))

        if self.flip == 1:
            im = im[:,:,::-1]
        return im


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.data_root + self.label_lst[idx])
        label = np.array(im) / 255#cv2.imread(self.data_root + self.label_lst[idx], 0) / 255
        #if self.scales != None:
        #    label = cv2.resize(label, None, None, fx=self.scales[self.scale_ind], fy=self.scales[self.scale_ind], \
        #            interpolation=cv2.INTER_NEAREST)
        #height, width = label.shape[:2]
        #h_off = self.crop_size - height
        #w_off = self.crop_size - width
        #label = cv2.copyMakeBorder(label, 0, max(0, h_off), 0, max(0, w_off), cv2.BORDER_CONSTANT, value=[-1,])
        #label = label[self.h_off:self.h_off+self.height, self.w_off:self.w_off+self.width]
        label = label[np.newaxis, ...]
        if self.flip == 1:
            label = label[:,:,::-1]
        return label

    def load_region(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.data_root + self.region_lst[idx])
        region = np.array(im, dtype=np.float32) / 15.0
        #print np.unique(region)
        #if self.scales != None:
        #    label = cv2.resize(label, None, None, fx=self.scales[self.scale_ind], fy=self.scales[self.scale_ind], \
        #            interpolation=cv2.INTER_NEAREST)
        #height, width = label.shape[:2]
        #h_off = self.crop_size - height
        #w_off = self.crop_size - width
        #label = cv2.copyMakeBorder(label, 0, max(0, h_off), 0, max(0, w_off), cv2.BORDER_CONSTANT, value=[-1,])
        region = region[np.newaxis, ...]
        if self.flip == 1:
            region = region[:,:,::-1]
        return region
