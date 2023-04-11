#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = "../COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.train_image_folder = "train2017"
        self.val_image_folder = "val2017"
        # self.train_ann = "instances_val2017.json"
        # self.train_image_folder = "val2017"

        self.depth = 0.33
        self.width = 0.375
        self.input_size = (640, 640)
        self.mosaic_scale = (0.5, 1.5)
        self.act = "relu"
        self.multiscale_range = 0
        self.mosaic_prob = 0.5
        # self.random_size = (10, 20)
        self.test_size = (640, 640)
        self.eval_interval = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
