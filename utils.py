#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os, sys, codecs, random
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from config import cfg


class Vocab(object):
    ''''''
    def __init__(self, trainfl, stopwds=False, minoccur=2):
        '''
        :param trainfl:
        '''
        self.infil = trainfl
        self.minoccur = minoccur
        self.counter = Counter()
        self.UNK = 0
        self.stopwords = stopwords.words('english')

        self.readfil()

        self.id2str = ['<unk>', '<s>', '<\s>']
        if stopwds: self.id2str = list(set(self.id2str) - set(self.stopwords)); del self.stopwords
        self.bldindex(); del self.counter
        self.size = len(self.id2str)
        self.str2id = dict(zip(self.id2str, range(len(self.id2str))))

        self.save_voc()

        self.initializer = self.get_initializer()
        self.embvec, self.wtvec = self.emb_init()

    def readfil(self):
        ''''''
        assert os.path.exists(self.infil), 'Cannot find the trainfile at %s' % self.infil
        # train file should one sentence per line
        with open(self.infil, 'r') as filin:
            for line in filin.readlines():
                line = line.strip().lower().split()
                self.counter.update(line)

    def save_voc(self):
        ''''''
        if not os.path.exists(cfg.savedir):
            os.mkdir(cfg.savedir)
        with open(os.path.join(cfg.savedir, 'vocab.fil'), 'w') as vocout:
            for idx, word in enumerate(self.id2str):
                vocout.write(word+'\t'+str(idx)+'\n')
        return


    def bldindex(self):
        ''''''
        for wrd, cnt in self.counter.most_common():
            if cnt < self.minoccur:
                continue
            else:
                self.id2str.append(wrd)

    def word2id(self, words):
        ''''''
        if isinstance(words, list):
            return [self.str2id.get(x, self.UNK) for x in words]
        else:
            return self.str2id.get(words, self.UNK)

    def id2word(self, idxs):
        ''''''
        if isinstance(idxs, list):
            return [self.id2str[x] for x in idxs]
        else:
            return self.id2str[idxs]

    def get_initializer(self):
        ''''''
        if cfg.initializer == 'orth':
            return tf.orthogonal_initializer(gain=cfg.mfactor, seed=cfg.seed)
        elif cfg.initializer == 'randomn':
            return tf.random_normal_initializer(mean=cfg.mean, stddev=cfg.stddev, seed=cfg.seed)
        elif cfg.initializer == 'randomu':
            return tf.random_uniform_initializer(minval=cfg.min, maxval=cfg.max, seed=cfg.seed)
        elif cfg.initializer == 'varian':
            return tf.variance_scaling_initializer(scale=cfg.scale, seed=cfg.seed)

    def emb_init(self):
        ''''''

        embinit = self.initializer([len(self.id2str), cfg.vecdim], dtype=tf.float32)
        wtinit = self.initializer([len(self.id2str), cfg.vecdim], dtype=tf.float32)
        embvec = tf.get_variable('embvec', initializer=embinit)
        wtvec = tf.get_variable('wtvec', initializer=wtinit)

        return embvec, wtvec

    def _looking_up(self, inputs, type='E'):
        ''''''
        if type == 'W':
            embs = tf.nn.embedding_lookup(self.wtvec, inputs)
        else:
            embs = tf.nn.embedding_lookup(self.embvec, inputs)

        return embs

    def _embsave(self):
        raise NotImplemented

class Dataset(object):
    ''''''
    def __init__(self, datafile, vocab):
        self.params = cfg
        self.trainfile = datafile
        self.vocab = vocab
        self.vsize = vocab.size
        self.dataset = self.load_file()
        self.windata = self.expand_set()
        del self.dataset
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[cfg.batch_size, 2*cfg.windows], name='inputs')
        self.trueTa = tf.placeholder(dtype=tf.int32, shape=[cfg.batch_size, ], name='Truetar')
        self.sampleTa = tf.placeholder(dtype=tf.int32, shape=[cfg.batch_size, cfg.sample], name='sampleTar')

    def load_file_up(self):
        # data should one sentence per line
        assert os.path.exists(self.trainfile), 'There does not exist trainfile %s' % self.trainfile
        datalines = []
        with open(self.trainfile, 'r') as trnin:
            line = trnin.readline().strip()
            while line:
                datalines.append(self.vocab.word2id(line.split()))

                if len(datalines) == cfg.batch_size:
                    yield datalines
                    datalines = []

                line = trnin.readline().strip()

                if not line:
                    yield datalines

    def load_file(self):
        # data should one sentence per line
        assert os.path.exists(self.trainfile), 'There does not exist trainfile %s' % self.trainfile
        datalines = []
        with open(self.trainfile, 'r') as trnin:
            line = trnin.readline().strip()
            while line:
                datalines.append(self.vocab.word2id(line.split()))
                line = trnin.readline().strip()

        return datalines

    def expand_set_up(self):
        ''''''

        wdata = []

        for prebatch in self.dataset:
            for sent in prebatch:
                if len(sent) > 2*cfg.windows+1:
                    wdata.extend(self.sent2wset(sent))
                    if len(wdata) > 500:
                        yield wdata
                        wdata = []

    def expand_set(self):
        ''''''

        wdata = []

        for sent in self.dataset:
            if len(sent) > 2*cfg.windows+1:
                wdata.extend(self.sent2wset(sent))

        return wdata


    def sent2wset(self, sentence):
        ''''''
        # arrary items with length: (2*cfg.windows + 1 + cfg.samples)
        wset = []
        for id, word in enumerate(sentence):
            if id+2*cfg.windows < len(sentence):
                wset.append(sentence[id:(id+2*cfg.windows+1)]+self.random_sample(sentence[id+cfg.windows]))
        return wset

    def random_sample(self, intnum):
        ''''''

        samples = random.sample(range(self.vsize), k=cfg.sample+1)

        if intnum not in samples:
            samples.pop()
        else:
            samples.pop(samples.index(intnum))

        return samples


    def batchs(self, keeplast=False):
        ''''''
        # every item [2*cfg.windows + 1 + cfg.sample]
        minibatch = []
        for num, itm in enumerate(self.windata):
            minibatch.append(itm)
            if len(minibatch) == cfg.batch_size:
                yield np.array(minibatch)
                minibatch = []
            if num == len(self.windata)-1 and keeplast:
                yield np.array(minibatch)

    def minibatchs(self):
        ''''''
        for batch in self.batchs():
            fedict = {}
            inputs = batch[:, range(cfg.windows)+range(cfg.windows+1, 2*cfg.windows+1)]
            trueL = batch[:, cfg.windows]
            samples = batch[:, range(2*cfg.windows+1, len(batch[-1]))]
            fedict.update({
                self.inputs: inputs,
                self.trueTa: trueL,
                self.sampleTa: samples
            })
            yield fedict, batch[:, range(2*cfg.windows+1)]



if __name__ == '__main__':
    voc = Vocab('./data/train.txt')
    dt = Dataset('./data/train.txt', voc)

    for itm in dt.batchs():
        print itm
        break
    for itm in dt.minibatchs():
        print itm
        break
    sys.exit(0)




    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print sess.run(voc.looking_up(dt.load_file()[0][0]))
    #     print sess.run(voc.looking_up(dt.load_file()[0][1]))
    #     print sess.run(voc.looking_up(dt.load_file()[0][3]))



