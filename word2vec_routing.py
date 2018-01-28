#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os, sys, codecs
from collections import Counter, deque
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from random import sample
from tqdm import tqdm
import logging
import threading

# parameters
# ======================================================
class configure(object):
    def __init__(self):
        ''''''
        # for vocab_param
        self.voc_file             = './data/train.txt'
        self.min_occur            = 2
        self.word_lower           = False
        self.vocab_save_path      = './vocab/'
        self.vocab_size           = None

        # for dataset
        self.train_file           = './data/snack.txt'
        self.eval_file            = './data/dev.txt'
        self.standard_eval        = './data/questions_words.txt'

        # for all
        self.window_size          = 5
        self.batch_size           = 1000
        self.sample_num           = 5
        self.vector_dim           = 200

        # for training process
        self.epoch_num            = 20
        self.routing_iters        = 3
        self.use_squash           = True
        self.loss_kind            = 'nce' # options: 'nce', 'margin'
        self.l_margin             = 0.5
        self.model_dir            = './model_saves/'
        self.lr                   = 0.001
        self.lr_decay             = 0.85
        self.optim                = 'adam' # options: 'adam', 'adagrad', 'sgd', 'rmsprop'
        self.clip                 = -1
        self.early_stop           = 30

        # other
        self.logging = logging


# build id2str, str2id
# ======================================================
class Vocab(configure):
    ''''''
    def __init__(self):
        super(Vocab, self).__init__()
        self.UNK = 0
        self.counter = Counter()
        self.chunk_reader()
        self.id2str = ['<unk>', '<s>', '</s>']
        self.build_indexs(); del self.counter
        self.vocab_size = len(self.id2str) if (self.vocab_size>len(self.id2str) or self.vocab_size == None) else self.vocab_size
        self.str2id = dict(zip(self.id2str, range(len(self.id2str))))
        self.save_vocab()

    def chunk_reader(self):
        ''''''
        assert os.path.exists(self.voc_file), OSError('There not exists file {}'.format(self.voc_file))
        with open(self.voc_file, 'r') as v_fin:
            for v_lin in v_fin:
                if self.word_lower:
                    self.counter.update(v_lin.strip().lower().split())
                else:
                    self.counter.update(v_lin.strip().split())

    def build_indexs(self):
        ''''''
        for wrd, cnt in self.counter.most_common(self.vocab_size):
            if cnt > self.min_occur - 1:
                self.id2str.append(wrd)

    def save_vocab(self):
        ''''''
        if not os.path.exists(self.vocab_save_path):
            os.mkdir(self.vocab_save_path)
        with open(os.path.join(self.vocab_save_path, 'vocab.fil'), 'w') as v_fot:
            v_fot.write('\n'.join(self.id2str))

    def idx2words(self, idx):
        ''''''
        if isinstance(idx, list):
            return [self.id2str[id] for id in idx]
        else:
            return self.id2str[idx]

    def words2idx(self, words):
        ''''''
        if isinstance(words, list):
            return [self.str2id.get(word, self.UNK) for word in words]
        else:
            return self.str2id.get(words, self.UNK)

# load data and make minibatchs
# ===================================================
class Dataloader(Vocab):
    ''''''
    def __init__(self):
        ''''''
        super(Dataloader, self).__init__()
        self.dataset = self.load_file(self.train_file) \
            if os.path.getsize(self.train_file) > 10240000 else self.fast_load(self.train_file)
        self.minibatchs = self.yield_batch(self.dataset)
        self.eval_set = self.fast_load(self.eval_file)
        self.eval_batchs = self.yield_batch(self.eval_set)
        self._analogy_questions = self.read_analogies()

    def load_file(self, filenm=None):
        ''''''
        with open(filenm, 'r') as tr_fin:
            chunk_lines = []
            for lin in tr_fin:
                if self.word_lower:
                    lin = self.words2idx(lin.strip().lower().split())
                else:
                    lin = self.words2idx(lin.strip().split())
                if len(lin) < 2 * self.window_size + 1:
                    continue
                window_set = self.sent2windows(lin)
                chunk_lines.extend(window_set)
                if len(chunk_lines) > self.batch_size - 1:
                    yield chunk_lines
                    chunk_lines = []

    def fast_load(self, filenm=None):
        ''''''
        with open(filenm, 'r') as t_fin:
            chunk_lines = []
            total_words = t_fin.read().replace('\n', ' ').split()
            for id in tqdm(range(len(total_words))):
                if id+2*self.window_size+1 < len(total_words):
                    chunk_lines.append(self.words2idx(total_words[id:id+2*self.window_size+1]))
                if len(chunk_lines) > self.batch_size - 1:
                    yield chunk_lines
                    chunk_lines = []

    def read_analogies(self):
        questions = []
        questions_skipped = 0
        unk_id = self.words2idx('<unk>')
        with open(self.standard_eval, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                ids = self.words2idx(words)
                if unk_id in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print "Eval analogy file: ", self.standard_eval
        print "Questions: ", len(questions)
        print "Skipped: ", questions_skipped
        return np.array(questions, dtype=np.int32)


    def sent2windows(self, sentence):
        ''''''
        window_set = []
        for idx, wrd in enumerate(sentence):
            if idx + 2 * self.window_size < len(sentence):
                window_set.append(sentence[idx:idx+2*self.window_size+1])
        return window_set

    def yield_batch(self, dataset):
        ''''''
        minibatch = []
        context_ids = range(2*self.window_size+1)
        context_ids.pop(self.window_size)
        for chunk_itm in dataset:
            for sent_item in chunk_itm:
                minibatch.append(sent_item)
                if len(minibatch) == self.batch_size:
                    minibatch = np.array(minibatch, dtype=np.int32)
                    yield minibatch[:, self.window_size], minibatch[:, context_ids]
                    minibatch = []

# real model
# =====================================================
class RoutingEmb(Dataloader):
    ''''''
    def __init__(self):
        super(RoutingEmb, self).__init__()
        self.sample_pool = range(self.vocab_size)
        self.build()

    def build(self):
        self.init_placeholder_emb()
        self.compute_graph()
        self.train_op(self.lr, self.optim, self.loss, self.clip)
        self.init_session_op()

    def negative_sampling(self, truelabels):
        ''''''
        negative_labels = []
        for t_label in truelabels:
            nsamples = sample(self.sample_pool, k=self.sample_num+1)
            if t_label in nsamples:
                nsamples.pop(nsamples.index(t_label))
            else:
                nsamples.pop()
            negative_labels.append(nsamples)
        return negative_labels

    def init_placeholder_emb(self):
        ''''''
        # init placeholder
        self.context_words = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 2*self.window_size],
                                            name='context_words')
        self.true_labels   = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,], name='true_label')
        self.samp_labels   = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sample_num],
                                            name='sample_labels')

        # init embedding looking_up table
        scale = 0.5 / self.vector_dim
        self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.vector_dim],
                               minval=-scale, maxval=scale), name='embeddings')


    def compute_graph(self):
        ''''''
        # embedding looking_up
        self.context_emb = tf.nn.embedding_lookup(self.emb, self.context_words)
        self.t_label_emb = tf.nn.embedding_lookup(self.emb, self.true_labels)
        self.s_label_emb = [tf.nn.embedding_lookup(self.emb, self.samp_labels[:,sid]) for sid in range(self.sample_num)]

        # get element_wise multiply
        t_context_init = tf.reshape(tf.multiply(tf.reshape(self.t_label_emb, [self.batch_size, 1, self.vector_dim]),
                       self.context_emb), shape=[self.batch_size, 2*self.window_size, 1, self.vector_dim])
        s_context_init = [tf.reshape(tf.multiply(tf.reshape(one_samp, [self.batch_size, 1, self.vector_dim]), self.context_emb),
                       shape=[self.batch_size, 2*self.window_size, 1, self.vector_dim]) for one_samp in self.s_label_emb]
        s_context_init = tf.concat(s_context_init, axis=2, name='sample_concat')
        t_B_IJ = tf.zeros(shape=[self.batch_size, 2*self.window_size, 1, 1, 1], name='b_ij_for_true_label')
        s_B_IJ = tf.zeros(shape=[self.batch_size, 2*self.window_size, self.sample_num, 1, 1], name='b_ij_for_sample_label')

        # Routing
        t_Rout = self.routing(t_context_init, t_B_IJ) # shape [B, 2W, 1, VD, 1]
        s_Rout = self.routing(s_context_init, s_B_IJ) # shape [B, 2W, S, VD, 1]

        # reduce to logits
        self.t_logits = tf.reshape(tf.reduce_sum(t_Rout, axis=-2), shape=[self.batch_size, -1])
        self.s_logits = tf.reshape(tf.reduce_sum(s_Rout, axis=-2), shape=[self.batch_size, -1])

        # loss
        self.loss = self.get_loss(self.t_logits, self.s_logits)
        # sigmoid
        self.t_lgt_sg = tf.nn.sigmoid(self.t_logits)
        self.s_lgt_sg = tf.nn.sigmoid(self.s_logits)

        # predict 1 indicate right
        self.t_predict = tf.to_float(tf.greater_equal(self.t_lgt_sg, self.l_margin, name='true_predict'))
        self.s_predict = tf.to_float(tf.less(self.s_lgt_sg, self.l_margin, name='sample_predict')) # shape [B, S]

        self.t_accurate = tf.reduce_sum(self.t_predict)
        self.s_accurate = tf.reduce_sum(tf.reduce_mean(self.s_predict, axis=1), axis=0)

    def routing(self, init_input, B_XX):
        ''''''
        init_input = tf.reshape(init_input, shape=[self.batch_size, 2*self.window_size, -1, self.vector_dim, 1],
                                name='init_input_reshape')
        input_stop = tf.stop_gradient(init_input, name='stop_gradient')

        if self.routing_iters == 0:
            return tf.reduce_sum(init_input, axis=1)

        for rt_turn in range(self.routing_iters):
            C_IJ = tf.nn.softmax(B_XX, axis=1, name='C_IJ') # shape [B, 2*W, 1, 1, 1]
            if rt_turn < self.routing_iters - 1:
                S_J = tf.multiply(C_IJ, input_stop, name='S_IJ_in_step') # shape [B, 2*W, S, VD, 1]
                V_J = tf.reduce_sum(S_J, axis=1, keepdims=True, name='for_V_J') # shape [B, 1, S, VD, 1]
                if self.use_squash:
                    V_J = self.squash(V_J)
                V_J_tiled = tf.tile(V_J, multiples=[1, 2*self.window_size, 1, 1, 1], name='V_J_tiled')
                # shape [B, 2*W, S, VD, 1]
                U_preoduce_V = tf.matmul(input_stop, V_J_tiled, transpose_a=True) # shape [B, 2*W, 1, 1, 1]
                B_XX += U_preoduce_V
            elif rt_turn == self.routing_iters - 1:
                S_J = tf.multiply(C_IJ, init_input, name='S_IJ_in_out')
                V_J = tf.reduce_sum(S_J, axis=1, keepdims=True, name='for_V_J_out')
                if self.use_squash:
                    V_J = self.squash(V_J)
                return V_J  # shape [B, 1, S, VD, 1]

    def squash(self, V_J):
        ''''''
        vector_exp = tf.exp(tf.reduce_sum(V_J, axis=-2, keepdims=True), name='squash_exp')
        exp_factor = vector_exp/(1+vector_exp)
        vector_squashed = exp_factor * V_J
        return  vector_squashed

    def get_loss(self, tl_logits, sl_logits):
        ''''''
        if self.loss_kind == 'nce':
            true_xnt = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tl_logits), logits=tl_logits)
            samp_xnt = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sl_logits), logits=sl_logits)
            nce_loss = (tf.reduce_sum(true_xnt)+tf.reduce_sum(tf.reduce_mean(samp_xnt, axis=1))) / float(self.batch_size)
            return nce_loss
        if self.loss_kind == 'margin':
            true_xnt = tf.ones_like(tl_logits) - tf.sigmoid(tl_logits)
            samp_xnt = tf.sigmoid(sl_logits)

            margin_loss = tf.square(tf.reduce_sum(true_xnt)) + tf.square(tf.reduce_sum(
                tf.reduce_mean(samp_xnt, axis=1), axis=0), name='sample_loss')
            return margin_loss

    def init_session_op(self):
        ''''''
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session_op(self):
        ''''''
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver.save(self.sess, self.model_dir)

    def restore_session(self, modeldir):
        ''''''
        print 'INFO: Reload the model from %s: '%modeldir
        self.saver.restore(self.sess, modeldir)

    def close_session(self):
        ''''''
        self.sess.close()

    def train_op(self, lrate, lmethod, loss, clip):
        ''''''
        lr_method = lmethod.lower()
        with tf.variable_scope('gt_train_op', reuse=tf.AUTO_REUSE):
            if lr_method == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lrate)
            elif lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lrate)
            elif lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lrate)
            elif lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lrate)
            else:
                raise NotImplementedError("Unknown method {}".format(lr_method))

            if clip > 0:
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def train_epoch(self, epoch_num):
        ''''''
        self.logging.info('In epoch {}'.format(epoch_num))
        for trueL, conT in self.minibatchs:
            sampL = self.negative_sampling(trueL)
            fdict = {}
            fdict[self.context_words] = conT
            fdict[self.true_labels] = trueL
            fdict[self.samp_labels] = sampL
            _, loss, accT, accS = self.sess.run([self.train_op, self.loss, self.t_accurate, self.s_accurate], feed_dict=fdict)
            print 'In epoch {}, loss: {:4.4f}\t True_L_acc: {:0.2f}, Sample_L_acc: {:0.2f}\r'.format(epoch_num,
                                    loss, accT, accS)
        results = self.run_eval()
        return results["f1"]


    def run_eval(self):
        ''''''
        accs  = []
        precs = []
        recls = []

        for trueL, conT in self.eval_batchs:
            sampL = self.negative_sampling(trueL)
            fdict = {}
            fdict[self.context_words] = conT
            fdict[self.true_labels] = trueL
            fdict[self.samp_labels] = sampL
            loss, acc_T, acc_S = self.sess.run([self.loss, self.t_accurate, self.s_accurate], feed_dict=fdict)
            accs.append((float(acc_T)+acc_S)/(2*self.batch_size))
            precs.append(acc_T/float(self.batch_size))
            recls.append(float(acc_T)/(acc_T+self.batch_size-acc_S))
        accs = np.mean(accs, dtype=np.float)
        precs = np.mean(precs, dtype=np.float)
        recls = np.mean(recls, dtype=np.float)
        f1 = 2*precs*recls/(precs+recls) if precs > 0 else 0
        results = {'acc': 100 * accs, 'f1': 100 * f1, 'precision': precs, 'recall': recls}

        print 'INFO: Testing on dev set\n'
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in results.items()])
        print msg
        return results


    def train(self):
        ''''''
        self.logging.basicConfig(level=self.logging.INFO)
        best_score = 0
        no_improve = 0
        for epoch in range(self.epoch_num):
            f1_value = self.train_epoch(epoch)
            self.lr *= self.lr_decay
            if f1_value > best_score:
                best_score = f1_value
                no_improve = 0
                self.save_session_op()
                self.save_emb()
                logging.info('Saved model for new best score')
            else:
                no_improve += 1
                if no_improve >= self.early_stop:
                    logging.info('Early stoped training with no improvment in {} epochs'.format(self.early_stop))
                    self.save_emb()
                    break
        return best_score

    def analogy_op_graph(self):
        ''''''
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)
        nemd = tf.nn.l2_normalize(self.emb, 1)

        a_emb = tf.gather(nemd, analogy_a)
        b_emb = tf.gather(nemd, analogy_b)
        c_emb = tf.gather(nemd, analogy_c)
        target = c_emb + (b_emb - a_emb)

        dist = tf.matmul(target, nemd, transpose_b=True)
        _, pred_idx = tf.nn.top_k(dist, 4)

        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemd, nearby_word)

        nearby_dist = tf.matmul(nearby_emb, nemd, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, self.vocab_size))

        self._ana_a = analogy_a
        self._ana_b = analogy_b
        self._ana_c = analogy_c
        self._nearby_word = nearby_word
        self._ana_pred_idx = pred_idx
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

    def _predict(self, analogy):
        ''''''
        idx, = self.sess.run([self._ana_pred_idx], {self._ana_a: analogy[:, 0],
                                                       self._ana_b: analogy[:, 1],
                                                       self._ana_c: analogy[:, 2],})
        return idx

    def count_dist(self):
        ''''''
        correct = 0
        try:
            total = self._analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError('Need to read the analogy data\n')
        start = 0

        while start<total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        correct += 1
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print("Analogy Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))

    def save_emb(self):
        ''''''
        final_emb = tf.nn.l2_normalize(self.emb, dim=1).eval(session=self.sess)
        dicts = self.id2str
        with open(os.path.join(self.vocab_save_path, 'word_embedding_{}'.format(self.vector_dim)), 'w') as emb_fot:
            for word, vector in zip(dicts, final_emb):
                emb_fot.write(word.encode('utf-8')+str(vector).replace('\n', ' '))

        plot_figure(dicts, final_emb, fig_name=os.path.join(self.vocab_save_path, 'tsne.png'))

    def analogy(self, wd0, wd1, wd2):
        '''wd0:wd1 VS wd2:wd3'''
        wid = self.words2idx([wd0, wd1, wd2])
        idx = self._predict(np.array([wid]))
        for c in [self.idx2words(i) for i in idx[0, :]]:
            if c not in [wd0, wd1, wd2]:
                print c
                return

    def nearby(self, words, num=20):
        '''print the nearby words'''

        idxs = np.array([self.words2idx(wd_) for wd_ in words])
        vals, idx = self.sess.run([self._nearby_val, self._nearby_idx], {self._nearby_word: idxs})
        for i in xrange(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self.idx2words(neighbor), distance))

# ==================================================================================================


def plot_figure(vocabs, embeddings, fig_name):
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
        plot_with_labels(low_dim_embs, labels, fig_name)
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)

def _start_shell(local_ns=None):
    ''''''
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


if __name__ == '__main__':

    model = RoutingEmb()
    _start_shell(locals())