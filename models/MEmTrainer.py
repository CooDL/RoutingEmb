from EmTrainer import emTrainer
from utils import plot_figure
from Configure import configure
import tensorflow as tf
import numpy as np


class MultiEm(emTrainer):
    ''''''
    def __init__(self, config):
        super(MultiEm, self).__init__(config)
        self.build()

    def __str__(self):
        ''''''
        return 'multi_trainer'

    def build(self):
        ''''''
        self.get_vocabs()
        self.get_dataset()
        self.get_placeholder()
        self.read_analogies()
        self.get_emeddings()
        self._predict_op()
        self.get_init_features()
        self.routing_process_multi()
        self.get_logit_loss()
        self.get_train_op(self.config.lr_method, self.config.lr, self.loss, self.config.clip)
        self.init_session_op()

    def get_init_features(self):
        ''''''

        self.true_word_feature = self.get_element_wise_feature(self.inputs, self.true_word)
        self.sample_words_features = [self.get_element_wise_feature(self.inputs, single_sample) for single_sample in
                                      self.sample_words]
        U_ = [self.true_word_feature] + self.sample_words_features
        self.Uij = tf.concat(U_, axis=2)
        self.W_ = tf.concat([tf.reshape(self.true_word, shape=[self.config.batch_size, 1, self.config.vec_dim, 1])] +
                            [tf.reshape(singlesm, shape=[self.config.batch_size, 1, self.config.vec_dim, 1])
                             for singlesm in self.sample_words], axis=1, name='FOR_ALL_PREDICT')
    def get_element_wise_feature(self, _inputs, _targets):
        ''''''
        _inputs = tf.transpose(_inputs, [1, 0, 2])
        element_wise_features = tf.multiply(_inputs, _targets)
        element_wise_features_ = tf.transpose(element_wise_features, [1, 0, 2])
        element_wise_features_ = tf.reshape(element_wise_features_, shape=[self.config.batch_size,
                                 2*self.config.window_size, 1, self.config.vec_dim, 1], name='FOR_U_HAT')
        return element_wise_features_

    def routing_process_multi(self):
        ''''''
        B_ij = tf.constant(np.zeros([self.config.batch_size, 2 * self.config.window_size,
                                     1+self.config.sample_num, 1, 1], dtype=np.float32))
        # u_shape [B, 2*Window, 1+Sam_num, Vec_dim]
        u_stopped = tf.stop_gradient(self.Uij, name='U_HAT_STOP_GRADIENT')
        for routing_iter in range(self.config.routing_times):
            C_ij = tf.nn.softmax(B_ij, dim=2, name='C_IJ')
            if routing_iter < self.config.routing_times - 1:
                S_J = tf.multiply(C_ij, u_stopped)
                V_J = tf.reduce_sum(S_J, axis=1, keep_dims=True, name='V_J')
                if self.config.use_squash:
                    V_J = self.squash_multi(V_J)
                V_J_tiled = tf.tile(V_J, multiples=[1, 2*self.config.window_size, 1, 1, 1], name='V_J_tiled')
                U_produce_V = tf.matmul(u_stopped, V_J_tiled, transpose_a=True, name='U_PRODUCE_V')
                #shape [B, 2*window_size, sample_num, 1, 1]
                B_ij += U_produce_V

            elif routing_iter == self.config.routing_times - 1:
                S_J = tf.multiply(C_ij, self.Uij, name='OUT_ROUTING')
                V_J_ = tf.reduce_sum(S_J, axis=1, keep_dims=True, name='OUT_ROUTING_REDUCE')
                if self.config.use_squash:
                    V_J_ = self.squash_multi(V_J_)
        self.routing_Vec = V_J_

    def squash_multi(self, V_J_out, dim=-1):
        ''''''
        #shape INPUT: [B, 1, 1+sample_Num, 200]
        vec_exp = tf.exp(tf.reduce_sum(V_J_out, axis=dim, keep_dims=True, name='exp_funct'))
        vec_exp_win = vec_exp / (1 + vec_exp)
        vec_squashed = vec_exp_win * V_J_out

        return vec_squashed

    def get_logit_loss(self):
        ''''''
        #INPUT: self.routing_Vec [B, 1, 6, 200, 1]
        truelabels = [[1] + [0] * self.config.sample_num] * self.config.batch_size
        if self.config.use_reduce_sum:
            logits = tf.reduce_sum(self.routing_Vec, axis=-2)
            logits_ = tf.reshape(logits, [self.config.batch_size, 1+self.config.sample_num], name="OUT_LOGITS_FROM_SUM")
        elif self.config.use_matmul:
            inputs_vec = tf.reshape(self.routing_Vec, shape=[self.config.batch_size, 1+self.config.sample_num,
                                                             self.config.vec_dim, 1], name='FOR_LOGITS_MATMUL')
            logits = tf.matmul(inputs_vec, self.W_, transpose_a=True)
            logits_ = tf.reshape(logits, [self.config.batch_size, 1+self.config.sample_num], name="OUT_LOGITS_FROM_MATMUL")

        sigmoid_logits = tf.sigmoid(logits_, name='USE_FOR_SIGMOID')
        self.predicts = tf.to_int32(tf.greater(sigmoid_logits, self.config.margin),name='PREDICT_LABELS')
        self.accurate = tf.reduce_mean(tf.to_float(tf.equal(self.predicts, truelabels, name='SELF_ACCURATE')), name='degit_of_right')
        self.loss = self.loss_fun(sigmoid_logits)


    def loss_fun(self, logits_):
        ''''''
        truelabels = [[1] + [0] * self.config.sample_num]*self.config.batch_size
        if self.config.lossType == 'MARGIN':
            margin_loss = tf.square(truelabels - logits_)
            margin_loss_ = tf.reduce_mean(margin_loss, axis=1)
            final_margin_loss = tf.reduce_mean(margin_loss_, axis=0, name='margin')
            return tf.sqrt(final_margin_loss)
        if self.config.lossType == 'CROSS':
            cross_loss = tf.nn.softmax_cross_entropy_with_logits(labels=truelabels, logits=logits_, name="cross")
            cross_loss_ = tf.reduce_mean(cross_loss, axis=0, name='cross_final')
            return cross_loss_
        raise NotImplementedError, 'Canot find a loss type named %s'%self.config.lossType

    def run_train_epoch(self, epoch_num):
        print 'INFO: star raining Epoch: %d'%epoch_num
        for fedict, _ in self.train_dataset.minibatchs():
            feed_dict = {}
            feed_dict[self.window_features] = fedict[self.train_dataset.inputs]
            feed_dict[self.true_target]     = fedict[self.train_dataset.trueTa]
            feed_dict[self.sample_targets]  = fedict[self.train_dataset.sampleTa]
            _, loss, predicts, correct = self.sess.run([self.train_op, self.loss, self.predicts, self.accurate],
                                                       feed_dict=feed_dict)
            #print predicts
        results = self.run_evaluate(self.dev_dataset)
        msg = " - ".join(["{} {:04.4f}".format(k, v)
                          for k, v in results.items()])
        print msg
        return results['acc']

    def run_evaluate(self, dataset):
        ''''''
        acc = []
        valoss = []
        val_pred = []
        for fed, _ in dataset.minibatchs():
            feed_dict = {}
            feed_dict[self.window_features] = fed[dataset.inputs]
            feed_dict[self.true_target]     = fed[dataset.trueTa]
            feed_dict[self.sample_targets]  = fed[dataset.sampleTa]
            loss, accuracy, predicts = self.sess.run([self.loss, self.accurate, self.predicts], feed_dict=feed_dict)
            valoss.append(loss)
            val_pred.append(predicts)
            acc.append(accuracy)
        acc = np.mean(acc, dtype=np.float32)
        valoss = np.mean(valoss, dtype=np.float32)

        return {'acc': acc, 'loss': valoss}

    def evaluate(self):
        ''''''
        print 'INFO: Testing on test set\n'
        results = self.run_evaluate(self.test_dataset)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in results.items()])
        print "Dev result"
        print msg