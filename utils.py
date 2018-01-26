#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os, sys, codecs, random
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from Configure import configure


class Vocab(object):
    ''''''
    def __init__(self, trainfl, config, stopwds=False, minoccur=2):
        '''
        :param trainfl:
        '''
        self.config = config
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
        if not os.path.exists(self.config.savedir):
            os.mkdir(self.config.savedir)
        with open(os.path.join(self.config.savedir, 'vocab.fil'), 'w') as vocout:
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
        if self.config.initializer == 'orth':
            return tf.orthogonal_initializer(gain=self.config.mfactor, seed=self.config.seed)
        elif self.config.initializer == 'randomn':
            return tf.random_normal_initializer(mean=self.config.mean, stddev=self.config.stddev, seed=self.config.seed)
        elif self.config.initializer == 'randomu':
            return tf.random_uniform_initializer(minval=self.config.min, maxval=self.config.max, seed=self.config.seed)
        elif self.config.initializer == 'varian':
            return tf.variance_scaling_initializer(scale=self.config.scale, seed=self.config.seed)

    def emb_init(self):
        ''''''
        #embinit = self.initializer([len(self.id2str), self.config.vecdim], dtype=tf.float32)
        #wtinit = self.initializer([len(self.id2str), self.config.vecdim], dtype=tf.float32)
        with tf.variable_scope('embinit', reuse=tf.AUTO_REUSE):
            embvec = tf.get_variable('embvec', shape=[len(self.id2str), self.config.vec_dim], dtype=tf.float32, initializer=tf.random_normal_initializer)
            wtvec = tf.get_variable('wtvec', shape=[len(self.id2str), self.config.vec_dim], dtype=tf.float32, initializer=tf.orthogonal_initializer)
            enorm = tf.sqrt(tf.reduce_sum(tf.square(embvec), 1, keep_dims=True))
            wnorm = tf.sqrt(tf.reduce_sum(tf.square(embvec), 1, keep_dims=True))
        return embvec/enorm, wtvec/wnorm

    def _looking_up(self, inputs, type='EMB'):
        ''''''
        if type == 'WGT':
            embs = tf.nn.embedding_lookup(self.wtvec, inputs)
        else:
            embs = tf.nn.embedding_lookup(self.embvec, inputs)

        return embs

    def _embsave(self, sess):
        ''''''
        embedding_dir = os.path.join(self.config.savedir, 'embeddings/')
        emsaver = tf.train.Saver({'embed_vec': self.embvec, 'weight_vec': self.wtvec}, max_to_keep=5, filename='trained_embeddings')
        emsaver.save(sess, save_path=embedding_dir)

class Dataset(object):
    ''''''
    def __init__(self, datafile, vocab, config):
        self.config = config
        self.trainfile = datafile
        self.vocab = vocab
        self.vsize = vocab.size
        self.dataset = self.load_file()
        self.windata = self.expand_set()
        del self.dataset
        self.inputs = 'inputs'
        self.trueTa = 'truelabel'
        self.sampleTa = 'samplelabel'

    def load_file_up(self):
        # data should one sentence per line
        assert os.path.exists(self.trainfile), 'There does not exist trainfile %s' % self.trainfile
        datalines = []
        with open(self.trainfile, 'r') as trnin:
            line = trnin.readline().strip()
            while line:
                datalines.append(self.vocab.word2id(line.split()))

                if len(datalines) == self.config.batch_size:
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
                if len(sent) > 2*self.config.window_size+1:
                    wdata.extend(self.sent2wset(sent))
                    if len(wdata) > 500:
                        yield wdata
                        wdata = []

    def expand_set(self):
        ''''''

        wdata = []

        for sent in self.dataset:
            if len(sent) > 2*self.config.window_size+1:
                wdata.extend(self.sent2wset(sent))

        return wdata


    def sent2wset(self, sentence):
        ''''''
        # arrary items with length: (2*self.config.windows + 1 + self.config.sample_num)
        wset = []
        for id, word in enumerate(sentence):
            if id+2*self.config.window_size < len(sentence):
                wset.append(sentence[id:(id+2*self.config.window_size+1)]+self.random_sample(sentence[id+self.config.window_size]))
        return wset

    def random_sample(self, intnum):
        ''''''

        samples = random.sample(range(self.vsize), k=self.config.sample_num+1)

        if intnum not in samples:
            samples.pop()
        else:
            samples.pop(samples.index(intnum))
        return samples

    def batchs(self, keeplast=False):
        ''''''
        # every item [2*self.config.windows + 1 + self.config.sample]
        minibatch = []
        for num, itm in enumerate(self.windata):
            minibatch.append(itm)
            if len(minibatch) == self.config.batch_size:
                yield np.array(minibatch)
                minibatch = []
            if num == len(self.windata)-1 and keeplast:
                yield np.array(minibatch)

    def minibatchs(self):
        ''''''
        for batch in self.batchs():
            fedict = {}
            inputs = batch[:, range(self.config.window_size)+range(self.config.window_size+1, 2*self.config.window_size+1)]
            trueL = batch[:, self.config.window_size]
            samples = batch[:, range(2*self.config.window_size+1, len(batch[-1]))]
            fedict.update({
                self.inputs: inputs,
                self.trueTa: trueL,
                self.sampleTa: samples
            })
            yield fedict, batch[:, range(2*self.config.window_size+1)]


def plot_figure(vocabs, embeddings, fig_name='tsne.png'):
    ''''''
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
        plt.savefig(filename)
        plt.close()
    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from tempfile import gettempdir

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        labels = [vocabs[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), fig_name))
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


if __name__ == '__main__':
    config = configure()
    voc = Vocab('./data/train.txt', config)
    dt = Dataset('./data/train.txt', voc, config=config)
    for itm in dt.minibatchs():
        print itm
        break
    sys.exit(0)

