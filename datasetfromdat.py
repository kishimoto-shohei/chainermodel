#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import numpy as np
import chainer
import glob
from PIL import Image, ImageFilter
from chainercv.utils import read_image


class DatasetFromDat(chainer.dataset.DatasetMixin):

    def __init__(self, file_path='.dat', label_out='', dtype=np.float32, label_dtype=np.int32):
        
        with open(file_path) as f :
            lines = f.readlines()
        
        self._dat = []
        for line in lines :
            dat = line.split(' ')
            if len(dat) < 8 :
                continue
            img_path = dat[0]
            label = int(dat[2])
            xmin = int(dat[4])
            ymin = int(dat[5])
            xmax = xmin + int(dat[6])
            ymax = ymin + int(dat[7])
            self._dat.append((img_path,label,ymin,xmin,ymax,xmax))
    
    def __len__(self):
        return len(self._dat)

    def get_example(self, i):
        #print('!!!!!!! get_example : ' + str(i))
        img_path,lab,xmin,ymin,xmax,ymax = self._dat[i]
        #print((img_path,lab,xmin,ymin,xmax,ymax))
        image = read_image(img_path, color=True)
        bbox = []
        bbox.append([xmin,ymin,xmax,ymax])
        bbox = np.stack(bbox).astype(np.float32)
        label = []
        label.append(lab)
        label = np.stack(label).astype(np.int32)
        return image, bbox, label

    def create_label_file(self):
        with open(self._label_out, "w") as f:
            for (label_index, label_name) in self._label_table:
                f.write('{},{}\n'.format(label_index, label_name))
