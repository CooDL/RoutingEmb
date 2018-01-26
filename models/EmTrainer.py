#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from utils import Vocab, Dataset, plot_figure
from Configure import configure
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os, sys


class emTrainer(object):
    ''''''

    def __init__(self, configs):
        ''''''
        self.config = configs
        self.training = True
        self.build()

    def __str__(self):
        ''''''
        return 'emtrainer'

    def get_vocabs(self):
        ''''''
        self.vocab = Vocab(self.config.train_file, self.config)

    def get_dataset(self):
        ''''''
        self.train_dataset = Dataset(self.config.train_file, self.vocab, self.config)
        self.dev_dataset = Dataset(self.config.dev_file, self.vocab, self.config)
        self.test_dataset = Dataset(self.config.test_file, self.vocab, self.config)

    def get_placeholder(self):
        ''''''
        self.window_features = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, 2*self.config.window_size,]
                                              , name='Feature_Inputs')
        self.true_target = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, ], name='True_Target')
        self.sample_targets = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, self.config.sample_num],
                                             name='Sample_Targets')

    def get_emeddings(self):
        ''''''
        self.inputs = self.vocab._looking_up(self.window_features, type='EMB')
        self.true_word = self.vocab._looking_up(self.true_target, type='WGT')
        self.sample_words = [self.vocab._looking_up(self.sample_targets[:, idx], type='WGT') for idx in range(self.config.sample_num)]

    def get_init_features(self):
        ''''''
        self.true_word_feature = self.get_element_wise_feature(self.inputs, self.true_word)
        self.sample_words_features = [self.get_element_wise_feature(self.inputs, single_sample) for single_sample in self.sample_words]

    def get_element_wise_feature(self, _inputs, _targets):
        ''''''
        with tf.variable_scope('element_wise_multiply'):
            _inputs = tf.transpose(_inputs, [1, 0, 2])
            element_wise_features = tf.multiply(_inputs, _targets)
            element_wise_features_ = tf.transpose(element_wise_features, [1, 0, 2])
            return element_wise_features_

    def get_allocate_true_samples(self):
        ''''''
        with tf.variable_scope('true_lables', reuse=tf.AUTO_REUSE):
            tr_loss, tr_correct, tr_predicts = self.get_logits_loss(self.true_word_feature, self.true_word, label=1)

        with tf.variable_scope('sample_labels', reuse=tf.AUTO_REUSE):
            sa_loss, sa_correct, sa_predicts = [], [], []
            for single_sample, sample_tar in zip(self.sample_words_features, self.sample_words):
                sloss, scorrect, spredict = self.get_logits_loss(single_sample, sample_tar, label=0)
                sa_loss.append(sloss)
                sa_correct.append(scorrect)
                sa_predicts.append(spredict)
            sa_loss = tf.reduce_mean(sa_loss)
            sa_correct = tf.reduce_mean(sa_correct)

        self.trloss = tr_loss
        self.saloss = sa_loss
        self.trcorrect = tr_correct
        self.sacorrect = sa_correct
        self.predict = tr_predicts
        self.loss = self.trloss + self.config.lamb*self.saloss
        return

    def get_logits_loss(self, init_features, targets, label=1):
        ''''''
        routed_features = self.get_compute_graph(init_features)
        if self.config.use_targeted_attention:
            pass
        #here routed_features should have shape [batch_size, 2*window_size, 200]
        final_features = tf.reduce_sum(routed_features, axis=1, keep_dims=True) # shape [batch_size, vec_dim]
        #final_features ==> [batch_size, 1, vec_dim]
        if self.config.use_reduce_sum:
            logits = tf.reduce_sum(final_features, axis=2, keep_dims=False)
            # logits ==> [batch_size, 1]
        elif self.config.use_matmul:
            targets = tf.reshape(targets, [self.config.batch_size, 1, self.config.vec_dim])
            Weight_tar = tf.reshape(targets, [self.config.batch_size, self.config.vec_dim, 1])
            logits = tf.matmul(final_features, Weight_tar)
        logits_ = tf.sigmoid(logits) # shape [batch_size, 1]
        logits_ = tf.reshape(logits_, [self.config.batch_size, ]) #shape ==> [batch_size, ]
        predict = tf.to_int32(tf.greater(logits_, self.config.margin))

        predict_ = predict if label == 1 else 1 - predict  # shape ==> [batch_size, ]
        correct = tf.reduce_sum(predict_) # shape ==> [int]
        loss = 1 - logits_ if label == 1 else logits_
        loss = tf.reduce_mean(tf.square(loss)) # shape ==> [float]

        return loss, correct, predict

    def margin_loss(self, logits, label=1):
        ''''''
        max_l = tf.square(tf.maximum(0., self.config.m_plus - logits))
        max_r = tf.square(tf.maximum(0., logits - self.config.m_minus))
        assert max_l.get_shape() == [self.config.batch_size,]
        True_label = [label]*self.config.batch_size


    def get_compute_graph(self, inputs):
        ''''''
        #shape of inputs should be [batch_size, 2*windows_size, vec_dim] ==> 2*window_size is regarded as input_cap_num
        B_ij = tf.constant(np.zeros([self.config.batch_size, 2*self.config.window_size, 1], dtype=np.float32))
        feature_routed = self.routing_process(inputs, B_ij)
        return  feature_routed

    def routing_process(self, inputs_, B_xx):
        ''''''
        #shape of inputs should be [batch_size, 2*windows_size, vec_dim], intergation on the input_cap_num dim
        #to compute how much the information from features we use

        inputs_stopped = tf.stop_gradient(inputs_, name='Inut_stop_gradient')
        for routing_iter in range(self.config.routing_times):
            C_ij = tf.nn.softmax(B_xx, dim=1)
            if routing_iter < self.config.routing_times - 1:
                S_j = tf.multiply(C_ij, inputs_stopped)
                V_j = tf.reduce_sum(S_j, axis=2, keep_dims=True)
                if self.config.use_squash:#skip squash
                    V_j = self.squash(V_j, 2)
                B_xx += V_j

            elif routing_iter == self.config.routing_times - 1:
                S_j = tf.multiply(C_ij, inputs_)
                V_j = S_j
                #V_j = tf.reduce_sum(S_j, axis=1, keep_dims=False)
                #comment this because further step for the features
                if self.config.use_squash:
                    V_j = self.squash(V_j, 2)
                return V_j

    def squash(self, inputs, dim=2):
        print 'INFO: Skipped'
        raise NotImplemented

    def build(self):
        self.get_vocabs()
        self.get_dataset()
        self.read_analogies()
        self.get_placeholder()
        self._predict_op()
        self.get_emeddings()
        self.get_init_features()
        self.get_allocate_true_samples()
        self.get_train_op(self.config.lr_method, self.config.lr, self.loss, self.config.clip)
        self.init_session_op()


    def init_session_op(self):
        ''''''
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session_op(self):
        ''''''
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        self.saver.save(self.sess, self.config.model_dir)

    def restore_session(self, modeldir):
        ''''''
        print 'INFO: Reload the model from %s: '%modeldir
        self.saver.restore(self.sess, modeldir)

    def close_session(self):
        ''''''
        self.sess.close()

    def get_train_op(self, lmethod, lrate, loss, clip):
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


    def run_train_epoch(self, epoch_num):
        ''''''
        print 'INFO: start training Epoch: %d'%epoch_num
        for fedict, _ in self.train_dataset.minibatchs():
            feed_dict = {}
            feed_dict[self.window_features] = fedict[self.train_dataset.inputs]
            feed_dict[self.true_target]     = fedict[self.train_dataset.trueTa]
            feed_dict[self.sample_targets]  = fedict[self.train_dataset.sampleTa]
            _, loss, tr_loss, sa_loss, tr_correct, sa_correct = self.sess.run(
                [self.train_op, self.loss, self.trloss, self.saloss, self.trcorrect, self.sacorrect],
                feed_dict=feed_dict)
            #print 'In Epoch %d:'%epoch_num, 'train_loss: %.4f: '%loss
        self._save_emb()
        self.count_dist()
        results = self.run_evaluate(self.dev_dataset)
        msg     = " - ".join(["{} {:04.4f}".format(k, v)
                for k, v in results.items()])
        print "#######################" \
                "####   Dev result   ###" \
                "#######################"
        print msg
        return results["f1"]

    def run_evaluate(self, dataset):
        ''''''
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for fed, _ in dataset.minibatchs():
            feed_dict = {}
            feed_dict[self.window_features] = fed[dataset.inputs]
            feed_dict[self.true_target]     = fed[dataset.trueTa]
            feed_dict[self.sample_targets]  = fed[dataset.sampleTa]
            tr_correct, sa_correct = self.sess.run([self.trcorrect, self.sacorrect], feed_dict=feed_dict)
            correct_preds += tr_correct
            total_preds   += self.config.batch_size
            total_correct += tr_correct + self.config.batch_size - sa_correct
            accs.append((tr_correct + sa_correct)/float(2*self.config.batch_size))
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2.0 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return {'acc': 100*acc, 'f1': 100*f1, 'precision': p, 'recall': r}

    def train(self):
        best_score = 0
        nepoc_no_imprv = 0
        for epoch in range(self.config.nepochs):
            f1_value = self.run_train_epoch(epoch)
            self.config.lr *= self.config.lr_decay
            if f1_value > best_score:
                best_score = f1_value
                nepoc_no_imprv = 0
                self.save_session_op()
                print '\nINFO: new best score'

            else:
                nepoc_no_imprv += 1
                if nepoc_no_imprv >= self.config.early_stop:
                    print 'INFO: early stop with improvement in %d epoch'%nepoc_no_imprv
                    break
        return best_score

    def evaluate(self):
        ''''''
        print 'INFO: Testing on test set\n'
        results = self.run_evaluate(self.test_dataset)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in results.items()])
        print "Dev result\n"
        print msg

    def _predict(self, analogy):
        ''''''
        idx, = self.sess.run([self._ana_pred_idx], {self._ana_a: analogy[:, 0],
                                                       self._ana_b: analogy[:, 1],
                                                       self._ana_c: analogy[:, 2],})
        return idx

    def _predict_op(self):
        ''''''
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)

        nemd = tf.nn.l2_normalize(self.vocab.embvec, 1)

        a_emb = tf.gather(nemd, analogy_a)
        b_emb = tf.gather(nemd, analogy_b)
        c_emb = tf.gather(nemd, analogy_c)

        target = c_emb + (b_emb - a_emb)

        dist = tf.matmul(target, nemd, transpose_b=True)

        _, pred_idx = tf.nn.top_k(dist, 4)

        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemd, nearby_word)

        nearby_dist = tf.matmul(nearby_emb, nemd, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, self.vocab.size))

        self._ana_a = analogy_a
        self._ana_b = analogy_b
        self._ana_c = analogy_c
        self._nearby_word = nearby_word
        self._ana_pred_idx = pred_idx
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

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
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

    def analogy(self, wd0, wd1, wd2):
        '''wd0:wd1 VS wd2:wd3'''
        wid = self.vocab.word2id([wd0, wd1, wd2])
        idx = self._predict(np.array([wid]))
        for c in [self.vocab.id2word(i) for i in idx[0, :]]:
            if c not in [wd0, wd1, wd2]:
                print c
                return
            print('UNKOWN WORDS')

    def nearby(self, words, num=20):
        '''print the nearby words'''

        idxs = np.array([self.vocab.word2id(wd_) for wd_ in words])
        vals, idx = self.sess.run([self._nearby_val, self._nearby_idx], {self._nearby_word: idxs})
        for i in xrange(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self.vocab.id2word(neighbor), distance))

    def _save_emb(self):
        ''''''
        final_emb_vec = self.vocab.embvec.eval(session=self.sess)
        final_wmb_vec = self.vocab.wtvec.eval(session=self.sess)
        #print final_emb_vec.shape
        #print final_wmb_vec.shape
        dicts = self.vocab.id2str
        plot_figure(dicts, final_emb_vec)
        return

    def read_analogies(self):
        questions = []
        questions_skipped = 0
        unk_id = self.vocab.word2id('<unk>')
        with open(self.config.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                ids = self.vocab.word2id(words)
                if unk_id in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self.config.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)


def _start_shell(local_ns=None):
    ''''''
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


if __name__ == '__main__':
    param = configure()
    train = emTrainer(param)
