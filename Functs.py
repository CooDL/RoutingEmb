#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from config import cfg
import tensorflow as tf

# print float(cfg.epsilon)

def compute(cntx, tar, label):
    ''''''
    npclass = [label]*cfg.batch_size
    npclass = tf.one_hot(npclass, depth=2, axis=1, dtype=tf.int32, name='LabelTarget')

    cntx_inputs = 0
    #element

    return npclass

def routing():
    raise NotImplemented








if __name__ == '__main__':
    with tf.Session() as sess:
        print sess.run(compute(1, 1, 0)).shape
