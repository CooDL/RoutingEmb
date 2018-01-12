#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from config import cfg
import tensorflow as tf
from utils import Vocab, Dataset
from Functs import *


class EmTrainer(object):
    ''''''
    def __init__(self, config):
        self.config = config

    def __call__(self, dataset, istraining=True):
        ''''''
        vocab = dataset.vocab
        inputs = dataset.inputs
        targets = dataset.trueTa
        samples = dataset.sampleTa
        cntx_inputs = vocab._looking_up(inputs)
        print cntx_inputs
        cntx_rt = tf.reshape(cntx_inputs, shape=(self.config.batch_size, 2*self.config.windows, 1, self.config.vecdim, 1))

        trueTar = vocab._looking_up(targets)
        Trueloss = compute(cntx_inputs, trueTar, label=1)

        sampleloss = []
        for sslice in range(cfg.sample):
            samTar = samples[:, sslice]
            samTarEmb = vocab._looking_up(samTar)
            sampleloss.append(compute(cntx_inputs, samTar, label=0))

        return cntx_inputs












if __name__ == '__main__':
    ''''''
    voc = Vocab(cfg.train)
    dt = Dataset(cfg.train, voc)
    tr = EmTrainer(cfg)

    de = tr(dt)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for fed, _ in dt.minibatchs():
            print sess.run(de, feed_dict=fed).shape
            break
