#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class configure(object):
    def __init__(self):
        ''''''
        self.train_file             = './data/train.txt'
        self.dev_file               = './data/dev.txt'
        self.test_file              = './data/test.txt'
        self.eval_data              = './data/questions_words.txt'
        self.batch_size             = 100
        self.window_size            = 5
        self.sample_num             = 5
        self.vec_dim                = 200

        self.routing_times          = 3
        self.use_squash             = False
        self.use_targeted_attention = False

        self.use_reduce_sum         = False #True #False
        self.use_matmul             = True#False

        self.lossType               = 'MARGIN'#'MARGIN','CROSS'

        self.margin                 = 0.5
        self.m_plus                 = 0.9
        self.m_minus                = 0.1
        self.lamb                   = 0.5

        self.model_dir              = './model.weights/'
        self.savedir                = './saves/'
        self.nepochs                = 10
        self.lr_method              = 'adam'
        self.initializer            = 'orth'
        self.lr                     = 0.001
        self.lr_decay               = 0.75
        self.clip                   = -1
        self.early_stop             = 3

        self.mfactor                = 1.0
        self.seed                   = 0
        self.mean                   = 0.5
        self.stddev                 = 0.9
        self.min                    = 0.1
        self.max                    = 1.0
        self.scale                  = 0.7

