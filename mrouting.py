#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os, sys, codecs
from collections import Counter, deque
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from random import sample
from tqdm import tqdm
import logging
import scipy
import pandas as pd
import argparse
reload(sys)
sys.setdefaultencoding('utf8')

# parameters
# ======================================================
class configure(object):
    def __init__(self):
        ''''''
        # for vocab_param
        self.voc_file             = './data/text8.txt'
        self.min_occur            = 5
        self.word_lower           = True
        self.vocab_save_path      = './vocab/'
        self.vocab_size           = None

        # for dataset
        self.train_file           = './data/text8.txt'
        self.eval_file            = './data/dev.txt'
        self.standard_eval        = './data/questions_words.txt'

        # for all
        self.window_size          = 5
        self.batch_size           = 4000
        self.sample_num           = 5
        self.vector_dim           = 200

        # for training process
        self.epoch_num            = 50
        self.routing_iters        = 3
        self.use_squash           = True
        self.loss_kind            = 'nce' # options: 'nce', 'margin'
        self.l_margin             = 0.5
        self.model_dir            = './model_saves/'
        self.lr                   = 0.001
        self.lr_decay             = 0.65
        self.optim                = 'adam' # options: 'adam', 'adagrad', 'sgd', 'rmsprop'
        self.clip                 = -1
        self.early_stop           = 30
        #self.use_two_matrix       = True

        # other
        self.logging = logging
        self.dump_batch = [16, 66] # which epoch and batch to dump


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
        logging.info('Finish build vocabulary!')

    def chunk_reader(self):
        ''''''
        assert os.path.exists(self.voc_file), OSError('There not exists file {}'.format(self.voc_file))
        with open(self.voc_file, 'r') as v_fin:
            for v_lin in v_fin:
                if self.word_lower:
                    self.counter.update(v_lin.strip().lower().decode('utf-8').encode('utf-8').split())
                else:
                    self.counter.update(v_lin.strip().decode('utf-8').encode('utf-8').split())

    def build_indexs(self):
        ''''''
        for wrd, cnt in self.counter.most_common(self.vocab_size):
            if cnt > self.min_occur - 1:
                self.id2str.append(wrd)

    def save_vocab(self):
        ''''''
        if not os.path.exists(self.vocab_save_path):
            os.mkdir(self.vocab_save_path)
        with codecs.open(os.path.join(self.vocab_save_path, 'vocab.fil'), 'w') as v_fot:
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
        self._analogy_questions = self.read_analogies()

    def all_dt_reset(self):
        ''''''
        self.dataset = self.load_file(self.train_file) \
            if os.path.getsize(self.train_file) > 190240000 else self.fast_load(self.train_file)
        self.minibatchs = self.yield_batch(self.dataset)
        self.eval_set = self.fast_load(self.eval_file)
        self.eval_batchs = self.yield_batch(self.eval_set)

    def load_file(self, filenm=None):
        ''''''
        with codecs.open(filenm, 'r', encoding='utf-8') as tr_fin:
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
        with codecs.open(filenm, 'r', encoding='utf-8') as t_fin:
            chunk_lines = []
            total_words = self.words2idx((t_fin.read().lower() if self.word_lower else t_fin.read()).replace('\n', ' ').split())
            for id in tqdm(range(len(total_words))):
                if id+2*self.window_size+1 < len(total_words):
                    chunk_lines.append(total_words[id:id+2*self.window_size+1])
                if len(chunk_lines) > self.batch_size - 1:
                    yield chunk_lines
                    chunk_lines = []

    def read_analogies(self):
        questions = []
        questions_skipped = 0
        unk_id = self.words2idx('<unk>')
        with codecs.open(self.standard_eval, "rb") as analogy_f:
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
                    
class CorpusProcess(object):
    ''''''
    def __init__(self, model):
        self.gog_ana = './eval_corpus/analogy_google.txt'
        self.msr_ana = './eval_corpus/analogy_msr.txt'
        self.sim_999 = './eval_corpus/sim_999.txt'
        self.ws_353 = './eval_corpus/ws_353.txt'
        self.model = model



    def eval_gog_ana(self):
        questions = []
        questions_skipped = 0
        with codecs.open(self.gog_ana, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                words_ids = self.model.words2idx(words)
                if 0 in words_ids or len(words_ids) != 4:
                    questions_skipped += 1
                    continue
                questions.append(words_ids)
        print "#Eval analogy file: ", self.gog_ana
        print "#Questions: ", len(questions)
        print "#Skipped: ", questions_skipped
        questions = np.array(questions, dtype=np.int32)
        results = self.model.count_dist(ana=questions)
        return results

    def eval_ms_ana(self):
        questions = {}
        questions_skipped = 0
        question_num = 0
        with codecs.open(self.msr_ana, 'rb') as analogy_f:
            for line in analogy_f:
                words = line.strip().lower().split(b' ')
                if len(words) != 5:
                    continue
                words_ids = self.model.words2idx(words)
                if 0 in words_ids[0:3]+words_ids[4:]:
                    questions_skipped += 1
                    continue
                if words[3] not in questions.keys():
                    question_num += 1
                    questions.update({words[3]: [[words_ids[0], words_ids[1], words_ids[2], words_ids[4]]]})
                else:
                    question_num += 1
                    questions[words[3]].append([words_ids[0], words_ids[1], words_ids[2], words_ids[4]])
        print "#Eval analogy file: ", self.msr_ana
        print "#Data has cata: ", len(questions)
        print "#Questions: ", question_num
        print "#Skipped: ", questions_skipped

        results = {}
        for key in questions.keys():
            data = np.array(questions[key], dtype=np.int32)
            results[key] = self.model.count_dist(ana=data)
        return results

    def eval_simlex_999(self):
        ''''''
        questions = {}
        questions_skipped = 0
        question_num = 0
        sem_999_data = pd.read_csv(self.sim_999, sep='\t')
        X = sem_999_data[['word1', 'word2']].values
        Y = sem_999_data['SimLex999'].values
        POS = sem_999_data['POS'].values
        X_ids = [self.model.words2idx(pair) for pair in X.tolist()]
        for pair, tag, score in zip(X_ids, POS, Y):
            if 0 in pair:
                questions_skipped += 1
                continue
            question_num += 1
            if tag in questions.keys():
                questions[tag].append(pair+[score])
            else:
                questions.update({tag:[pair+[score]]})
        print "#Eval similarity file: ", self.sim_999
        print "#Data has cata: ", len(questions)
        print "#Questions: ", question_num
        print "#Skipped: ", questions_skipped
        results={}
        for key in questions.keys():
            data = np.array(questions[key], dtype=np.float32)
            X_dt = np.array(data[:, [0, 1]], dtype=np.int32)
            Y_dt = data[:, 2]
            results[key] = self.model.similarity_op(X_dt, Y_dt) # score, cos_sim
        with open('./eval_results/sim_999.txt', 'w') as sim_out:
            sim_out.write('\t'.join(['word1', 'word2', 'POS', 'EVALUATED'])+'\n')
            for key in results:
                for pair, score in zip(questions[key], results[key][1]):
                    sim_out.write(self.model.idx2words(int(pair[0])) + '\t'+ self.model.idx2words(int(pair[1]))+
                                  '  ' + str(score)+'\n')
        return results

    def eval_ws353(self):
        ''''''
        questions = []
        questions_skipped = 0
        requestion = []

        ws_data = pd.read_csv(self.ws_353, sep='\t')
        X_data = ws_data[['Word 1', 'Word 2']].values
        Y_data = ws_data['Human (mean)']
        X_ids = [self.model.words2idx(pair) for pair in X_data.tolist()]
        for pair, score, str_pair in zip(X_ids, Y_data, X_data.tolist()):
            if 0 in pair:
                questions_skipped += 1
                continue
            questions.append(pair+[score])
            requestion.append(str_pair)
        print "#Eval similarity file: ", self.ws_353
        print "#Questions: ", len(questions)
        print "#Skipped: ", questions_skipped
        data = np.array(questions, dtype=np.float32)
        X_dt = np.array(data[:, [0, 1]], dtype=np.int32)
        Y_dt = data[:, 2]
        results, cos_sim = self.model.similarity_op(X_dt, Y_dt)
        with open('./eval_results/ws_353.txt', 'w') as ws_out:
            ws_out.write('\t'.join(['Word 1', 'Word 2', 'EVALUATED'])+'\n')
            for itm, score in zip(requestion, cos_sim):
                ws_out.write('\t'.join(itm)+str(score)+'\n')
        return results

    def run_all_evaluate(self, enp):
        gog_ana = self.eval_gog_ana() # one score
        ms_ana = self.eval_ms_ana() # dict
        sim_999 = self.eval_simlex_999() # dict
        ws_353 = self.eval_ws353()
        ms_result = "".join(["{}\t{:04.2f}\n".format(k, v) for k, v in ms_ana.items()])
        sim_result = "".join(["{}\t{:04.2f}\n".format(k, v[0]) for k, v in sim_999.items()])
        model_name = str(self.model)
        save_dir = os.path.join('./eval_results', model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file = os.path.join(save_dir, '{}_results.txt'.format(enp))
        with open(save_file, 'w') as rtt:
            rtt.write('########   Google Analogy  Results ##########\n')
            rtt.write(str(gog_ana))
            rtt.write('\n\n\n')
            rtt.write('########   MS Analogy Results ###########\n')
            rtt.write(ms_result)
            rtt.write('\n\n\n')
            rtt.write('########   SimLex 999 Results ###########\n')
            rtt.write(sim_result)
            rtt.write('\n\n\n')
            rtt.write('########   Word_similarity 353 Results ###########\n')
            rtt.write(str(ws_353))


class MErouting(Dataloader):
    def __init__(self):
        ''''''
        super(MErouting, self).__init__()
        self.sample_pool = range(self.vocab_size)
        self.build()
        
    def build(self):
        ''''''
        self.init_placeholder_emb()
        self.compute_graph()
        self.train_op(self.lr, self.optim, self.loss, self.clip)
        self.init_session_op()
        self.analogy_op_graph()
    
    def init_session_op(self):
        ''''''
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session_op(self):
        ''''''
        save_path = os.path.join(self.model_dir, str(self))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.saver.save(self.sess, save_path,)

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

        #self.wemb = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.vector_dim],stddev=0.66, seed=666), name='w_emb')

    def compute_graph(self):
        ''''''
        # embedding looking_up
        context_emb = tf.nn.embedding_lookup(self.emb, self.context_words)
        t_label_emb = tf.nn.embedding_lookup(self.emb, self.true_labels)
        s_label_emb = [tf.nn.embedding_lookup(self.emb, self.samp_labels[:, idx]) for idx in range(self.sample_num)]
        # get element_wise multiply
        t_context_init = tf.reshape(tf.multiply(tf.reshape(t_label_emb, [self.batch_size, 1, self.vector_dim]), context_emb),
                                    shape=[self.batch_size, 2 * self.window_size, 1, self.vector_dim], name='true')
        ts_context_init = [t_context_init]
        ts_context_init.extend([tf.reshape(tf.multiply(tf.reshape(one_samp, [self.batch_size, 1, self.vector_dim]), context_emb),
                       shape=[self.batch_size, 2 * self.window_size, 1, self.vector_dim], name='sample') for one_samp in s_label_emb])
        ts_context_init = tf.concat(ts_context_init, axis=2, name='total_init_input') # [B, 2*windows, 1+S_num, V_dim]

        B_IJ = tf.zeros(shape=[self.batch_size, 2*self.window_size, 1+self.sample_num, 1, 1], name='init_B_IJ')
        routing_out, C_IJ = self.mrouting(ts_context_init, B_IJ)

        self.C_IJ = tf.reshape(C_IJ, shape=[self.batch_size, 2*self.window_size, self.sample_num+1])
        self.logits = tf.reshape(tf.reduce_sum(routing_out, axis=-2), shape=[self.batch_size, -1], name='logits_reshape')
        # shape self.logits [B, 1+S]
        labels = np.array([[1] + [0]*self.sample_num]*self.batch_size, dtype=np.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.t_predict = tf.to_float(tf.greater(tf.sigmoid(self.logits[:, 0]), self.l_margin))
        self.s_predict = tf.to_float(tf.less_equal(tf.sigmoid(self.logits[:, 1:]), self.l_margin))
        self.t_accurate = tf.reduce_sum(self.t_predict)
        self.s_accurate = tf.reduce_sum(tf.reduce_mean(self.s_predict, axis=1))

    def mrouting(self, init_input, B_XX):
        ''''''
        init_input = tf.reshape(init_input, shape=[self.batch_size, 2*self.window_size, 1+self.sample_num, self.vector_dim, 1],
                                name='init_input_reshape')
        input_stop = tf.stop_gradient(init_input, name='stop_gradient')
        if self.routing_iters == 0:
            return tf.reduce_sum(init_input, axis=1, name='no_routing')

        for rt_time in range(self.routing_iters):
            C_IJ = tf.nn.softmax(B_XX, dim=2) # on output dim
            if rt_time < self.routing_iters - 1:
                S_J = tf.multiply(C_IJ, input_stop, name='S_J_internal_mul_iter') # shape [B, 2W, 1+S, V_D, 1], stopped
                S_J = tf.reduce_sum(S_J, axis=1, keep_dims=True, name='S_J_internal_reduce_iter')
                # reduce on input_dim ==> shape [B, 1, 1+S, V_D, 1]
                V_J = self.squash(S_J) if self.use_squash else S_J

                # tiled the V_J
                V_J_tiled = tf.tile(V_J, multiples=[1, 2*self.window_size, 1, 1, 1], name='V_J_tiled')

                # produce the update for B_XX
                U_produced = tf.matmul(input_stop, V_J_tiled, name='U_produced', transpose_a=True) # shape [B, 2W, 1+S, 1, 1]
                B_XX += U_produced

            elif rt_time == self.routing_iters - 1:
                S_J = tf.multiply(C_IJ, init_input, name='S_J_external_mul') # weighting the true input to apply back_prog
                S_J = tf.reduce_sum(S_J, axis=1, keep_dims=True, name='S_J_external_reduce_iter')
                V_J = self.squash(S_J) if self.use_squash else S_J
                return V_J, C_IJ
    
    def squash(self, V_J):
        ''''''
        vector_exp = tf.exp(tf.reduce_sum(V_J, axis=-2, keep_dims=True), name='squash_exp')
        exp_factor = vector_exp/(1+vector_exp)
        vector_squashed = exp_factor * V_J
        return  vector_squashed
    
    

    def train_epoch(self, epoch_num):
        ''''''
        self.logging.info('In epoch {}'.format(epoch_num))
        self.all_dt_reset()
        dump_batch = 0
        for trueL, conT in self.minibatchs:
            sampL = self.negative_sampling(trueL)
            fdict = {}
            fdict[self.context_words] = conT
            fdict[self.true_labels] = trueL
            fdict[self.samp_labels] = sampL
            _, loss, accT, accS, C_IJ = self.sess.run([self.train_op, self.loss, self.t_accurate, self.s_accurate, self.C_IJ], feed_dict=fdict)

            print 'In epoch {}, loss: {:0.4f}\t True_L_acc: {:0.4f}, Sample_L_acc: {:0.4f}\r'.format(epoch_num, loss, float(accT)/self.batch_size, accS/self.batch_size)
            # C_IJ [B, 2*window_size, 1+sample_num]
            if (epoch_num == self.dump_batch[0] - 1) and (dump_batch == self.dump_batch[1] - 1):
                self.dump_C_IJ(conT, trueL, sampL, C_IJ)
        results = self.run_eval()
        return results["f1"]

    def dump_C_IJ(self, contx, true_l, samp_l, C_IJ):
        ''''''
        print C_IJ.shape
        assert C_IJ.shape == (self.batch_size, 2*self.window_size, self.sample_num+1)
        labels = np.concatenate([np.reshape(true_l, newshape=[self.batch_size, 1]), samp_l], axis=1)
        assert labels.shape == (self.batch_size, self.sample_num+1)
        save_path = os.path.join(self.vocab_save_path, str(self))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        _savefile = os.path.join(save_path, 'label_C_ij.txt')
        with open(_savefile, 'w') as tout:
            for sid, (cntx, lblt) in enumerate(zip(contx.tolist(), labels.tolist())):
                cntx = ' '.join(self.idx2words(cntx))
                Lblw = self.idx2words(lblt)
                tout.write('#TW: '+Lblw[0]+'\t#SW: '+' '.join(Lblw[1:])+'\tCNTXW: '+cntx+str(C_IJ[sid])+'\n')
    
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
        msg = " - ".join(["{} {:04.4f}".format(k, v) for k, v in results.items()])
        print msg
        return results

    def train(self):
        ''''''
        self.logging.basicConfig(level=self.logging.INFO)
        best_score = 0
        no_improve = 0
        model_name = str(self)
        try:
            for epoch in range(self.epoch_num):
                f1_value = self.train_epoch(epoch)
                corpus_Eval = CorpusProcess(self)
                self.lr *= self.lr_decay
                self.count_dist()
                corpus_Eval.run_all_evaluate(epoch)
                if f1_value > best_score:
                    best_score = f1_value
                    no_improve = 0
                    self.save_session_op()
                    self.save_emb('epoch_{}_tsne.png'.format(epoch))
                    logging.info('Saved model for new best score')
                else:
                    no_improve += 1
                    if no_improve >= self.early_stop:
                        logging.info('Early stoped training with no improvment in {} epochs'.format(self.early_stop))
                        self.save_emb('epoch_{}_tsne.png'.format(epoch))
                        break
        except KeyboardInterrupt:
            try:
                raw_input('Press <Enter> to start Ipython, <Ctrl-C> to exit')
                _start_shell(locals())
            except:
                print('\r')
                sys.exit(0)
        return best_score


    def save_emb(self, name):
        ''''''
        final_emb = tf.nn.l2_normalize(self.emb , dim=1).eval(session=self.sess)
        dicts = self.id2str
        with open(os.path.join(self.vocab_save_path, 'word_embedding_{}_nor_matrix'.format(self.vector_dim)), 'w') as emb_fot:
            for word, vector in zip(dicts, final_emb):
                emb_fot.write(word+'\t'+str(vector).replace('\n', ' ')+'\n')
        plot_figure(dicts, final_emb, fig_name=os.path.join(self.vocab_save_path, name))

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

    def count_dist(self, ana=None):
        ''''''
        correct = 0
        analogy_questions = self._analogy_questions if ana is None else ana
        try:
            total = analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError('Need to read the analogy data\n')
        start = 0

        while start<total:
            limit = start + 2500
            sub = analogy_questions[start:limit, :]
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
        print("Analogy Eval %4d/%d \t accuracy = %4.2f%%" % (correct, total, correct * 100.0 / total))
        if ana is None:
            with open('./analogy.txt', 'a+') as filt:
                filt.write("Analogy Eval %4d/%d accuracy = %4.2f%%\n" % (correct, total, correct * 100.0 / total))
        return correct * 100.0 / total

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

    def similarity_op(self, x, y):
        '''
        :param x: [wrd1_id, wrd2_id]
        :param y: [man_score]
        :return: [score]
        '''
        nemd = tf.nn.l2_normalize(self.emb, 1)
        wrd1_emb = tf.nn.embedding_lookup(nemd, x[:, 0])
        wrd2_emb = tf.nn.embedding_lookup(nemd, x[:, 1])
        cos_similarity = tf.reduce_sum(tf.multiply(wrd1_emb, wrd2_emb), axis=1)
        cos_sim = self.sess.run(cos_similarity)
        results = scipy.stats.spearmanr(cos_sim, y).correlation
        return results, cos_sim

    def __str__(self):
        return 'multi'
        
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
    ''''''
    param = argparse.ArgumentParser()
    param.add_argument('--model', type=str, default='Single')
    param.add_argument('--routtm', type=int, default=3)
    param.add_argument('--lr', type=float, default=0.001)
    param.add_argument('--method', type=str, default='adam')
    param.add_argument('--lr_decay', type=float, default=0.25)
    args = param.parse_args()
    model = MErouting()
    model.train()
    _start_shell(locals())


