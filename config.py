#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse


config = argparse.ArgumentParser(usage='param for the CapSent', description='toSenCnn')

config.add_argument('--epoch', type=int, default=50, help='Epoch number')
# embedding settings
config.add_argument('--vecdim', type=int, default=200, help='training embedding file dimesion')
config.add_argument('--windows', type=int, default=2, help='the windows between left and right')
config.add_argument('--sample', type=int, default=10, help='the random sample number')

# matrix init params
config.add_argument('--initializer', type=str, default='orth',
                    help="initializer to use. Options: 1.'orth', 2. 'randomn', 3. 'randomu', 4. ")
config.add_argument('--mfactor', type=float, default=1.0, help='multiplicative factor to initialize matrix')
config.add_argument('--seed', type=int, default=6, help='random seed')
config.add_argument('--mean', type=float, default=0.0, help='the param for random_normal_initializer')
config.add_argument('--stddev', type=float, default=0.01, help='stddev for Weight initializer')
config.add_argument('--min', type=float, default=-6.0, help='the param for uniform_initializer')
config.add_argument('--max', type=float, default=6.0, help='tha param for uniform_initializer')
config.add_argument('--scale', type=float, default=1.0, help='tha param for varian_initializer')

# dataset and saving settings
config.add_argument('--train', type=str, default='./data/train.txt', help='train.file')
config.add_argument('--dev', type=str, default='./data/dev.txt', help='dev.file')
config.add_argument('--test', type=str, default='./data/test.txt', help='test.file')

config.add_argument('--embeddrop', type=float, default=0.5, help='embedding dropout when training')
config.add_argument('--classno', type=int, default=3, help='the classify class number')
config.add_argument('--ckpt', type=str, default='./ckpt/', help='model dir')
config.add_argument('--savedir', type=str, default='./saves/', help='save dirs')
config.add_argument('--batch_size', type=int, default=10, help='the batch size')

config.add_argument('--iter_routing', type=int, default=3, help='the time using routing algrothm')
config.add_argument('--epsilon', type=float, default=1e-9, help='epsilon')
config.add_argument('--m_plus', type=float, default=0.9, help='m plus')
config.add_argument('--m_minus', type=float, default=0.1, help='m minus')
config.add_argument('--lambdaval', type=float, default=0.5, help='descent rate')
config.add_argument('--istraining', type=bool, default=True, help='training state')
config.add_argument('--summstep', type=int, default=100, help='the summary step')
config.add_argument('--val_sum_freq', type=int, default=500, help='the frequence of val')
config.add_argument('--save_freq', type=int, default=3, help='save every n epochs')

cfg = config.parse_args()

if __name__ == '__main__':
    print(cfg.lr)
