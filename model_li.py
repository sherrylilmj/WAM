import tensorflow as tf
import numpy as np
import time
from utlis_li import get_batch_index
class WAM(object):
    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout
        
        self.word2id = config.word2id
        self.max_sentence_len = config.max_sentence_len
        self.max_entity_len = config.max_entity_len
        self.word2vec = config.word2vec
        self.sess = sess
        
        self.timestamp = str(int(time.time()))
    def build_model(self):
        with tf.name_scope('inputs'):
            self.sentences = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.sentence_lens = tf.placeholder(tf.int32, None)
            self.sentence_types = tf.placeholder(tf.float32, [None, self.max_sentence_len])

            self.sentence_entity_loc_1 = tf.placeholder(tf.int32, None)
            self.sentence_entity_loc_2 = tf.placeholder(tf.int32, None)
            self.sentence_entity1 = tf.placeholder(tf.int32, [None, self.max_entity_len])
            self.sentence_entity2 = tf.placeholder(tf.int32, [None, self.max_entity_len])

            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            
            inputs = tf.nn.embedding_lookup(self.word2vec, self.sentences)
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)

            entity1 = tf.nn.embedding_lookup(self.word2vec, self.sentence_entity1)
            entity1 = tf.cast(entity1, tf.float32)
            entity1 = tf.reduce_mean(entity1, 1)

            entity2 = tf.nn.embedding_lookup(self.word2vec, self.sentence_entity2)
            entity2 = tf.cast(entity2, tf.float32)
            entity2 = tf.reduce_mean(entity2, 1)
        with tf.name_scope('weights'):
            weights = {
                'attention': tf.get_variable(
                    name='W_l',
                    shape=[1, self.n_hidden * 4],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_r',
                    shape=[self.n_hidden * 12, self.n_class],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
        with tf.name_scope('biases'):
            biases = {
                'softmax': tf.get_variable(
                    name='B_r',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
        with tf.name_scope('dynamic_rnn'):
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )
            outputs, state, _ = tf.nn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                sequence_length=self.sentence_lens,
                dtype=tf.float32,
                scope='BiLSTM'
            )
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.max_sentence_len, self.n_hidden * 2])
            batch_size = tf.shape(outputs)[0]

            adj_input = tf.TensorArray(size=batch_size, dtype=tf.float32)

            outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            outputs_iter = outputs_iter.unstack(outputs)

            sentence_entity_loc_1_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            sentence_entity_loc_1_iter = sentence_entity_loc_1_iter.unstack(self.sentence_entity_loc_1)

            sentence_entity_loc_2_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            sentence_entity_loc_2_iter = sentence_entity_loc_2_iter.unstack(self.sentence_entity_loc_2)
            def edge_representation(i,adj_input):
                output = outputs_iter.read(i)
                entity_loc_1 = sentence_entity_loc_1_iter.read(i)
                entity_loc_2 = sentence_entity_loc_2_iter.read(i)
                output_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
                output_iter = output_iter.unstack(output)
                output_iter_ = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
                output_iter_ = output_iter_.unstack(output)
                output_context = tf.TensorArray(size=self.max_sentence_len, dtype=tf.float32)
                output_word = output_iter_.read(0)
                entity_pos_1 = tf.expand_dims(entity_loc_1,-1)
                entity_pos_2 = tf.expand_dims(entity_loc_2,-1)
                entity_pos_1 = tf.tile(entity_pos_1,[self.n_hidden*2])
                entity_pos_2 = tf.tile(entity_pos_2,[self.n_hidden*2])
                #print("entity_pos_1"+str(entity_pos_1.shape))
                entity1=tf.concat([output_word,tf.cast(tf.subtract(entity_pos_1,entity_pos_2),dtype = tf.float32)], axis = 0)
                #print("entity_1"+str(entity1.shape))
                #print("entity_pos_2"+str(entity_pos_2.shape))
                entity2=tf.concat([output_word,tf.cast(tf.subtract(entity_pos_2,entity_pos_1),dtype = tf.float32)], axis = 0)
                #print("entity_2"+str(entity2.shape))
                flag1=0
                flag2=0
                #print(entity2.shape)
                for index in range(0,self.max_sentence_len):
                    output_word=output_iter.read(index)
                    entity1=tf.cond(tf.equal((index),tf.to_int32(entity_loc_1)),lambda:tf.stack([output_word,tf.cast(tf.subtract(entity_pos_1,entity_pos_2),dtype = tf.float32)], axis = 0),lambda:entity1)
                    entity2=tf.cond(tf.equal((index),tf.to_int32(entity_loc_2)),lambda:tf.stack([output_word,tf.cast(tf.subtract(entity_pos_2,entity_pos_1),dtype = tf.float32)], axis = 0),lambda:entity2)
                    flag1=tf.cond(tf.equal((index),tf.to_int32(entity_loc_1)),lambda:1,lambda:0)
                    flag2=tf.cond(tf.equal((index),tf.to_int32(entity_loc_2)),lambda:1,lambda:0)
                    output_context = output_context.write(index,tf.concat([output_word,tf.cast(tf.concat([tf.subtract(entity_pos_1,entity_pos_2)],0),dtype = tf.float32)],0))
                    #if((index+1)==tf.to_int32(entity_loc_1)):
                    #    flag1=1
                    #    entity1=tf.stack([output_word,tf.cast(tf.subtract(entity_pos_1,entity_pos_2),dtype = tf.float32)], axis = 0)
                    #elif((index+1)==tf.to_int32(entity_loc_2)):
                    #    flag2=1
                    #    entity2=tf.stack([output_word,tf.cast(tf.subtract(entity_pos_2,entity_pos_1),dtype = tf.float32)], axis = 0)
                    #else:
                    #    output_context = output_context.write(index - flag1 - flag2,tf.concat([output_word,tf.cast(tf.concat([tf.subtract(entity_pos_1,entity_pos_2)],0),dtype = tf.float32)],0))
                    #output_context = output_context.write(index - flag1 - flag2,tf.concat([output_word,tf.cast(tf.concat([tf.subtract(index,entity_pos_1),tf.subtract(index,entity_pos_2)],0),dtype = tf.float32)],0))
                output_context = output_context.stack()
                print("output_context "+str(output_context))
                output_context = tf.squeeze(output_context)
                context_final = tf.transpose(output_context,perm = [1,0])
                print(context_final.shape)
                u = tf.matmul(weights['attention'],tf.tanh(context_final))
                a = tf.nn.softmax(u)
                context_representation = tf.matmul(context_final,tf.transpose(a,[1,0]))
                context_representation = tf.squeeze(context_representation)
                print("context_representation"+str(context_representation.shape))
                print("entity1"+str(entity1.shape))
                print("entity2"+str(entity2.shape))
                entity_concat=tf.concat([entity1,entity2],axis = 0)
                entity_concat=tf.reshape(entity_concat,[self.n_hidden*8])
                edge = tf.concat([entity_concat,context_representation],axis = 0)
                adj_input = adj_input.write(i,edge) 
                return (i + 1,adj_input)
            def condition(i, adj_input):
                return i < batch_size
            _, input_final = tf.while_loop(cond=condition, body=edge_representation, loop_vars=(0, adj_input))
            self.input_final = tf.reshape(input_final.stack(), [-1, self.n_hidden * 12])
            self.predict = tf.matmul(self.input_final, weights['softmax']) + biases['softmax']
        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.predict, labels = self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            self.predict_label = tf.argmax(self.predict, 1)
            self.correct_pred = tf.equal(self.predict_label, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))
            
        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        _dir = 'logs/' + str(self.timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)
    def train(self, data):
        sentences, sentence_lens, sentence_locs_1, sentence_locs_2, entity1, entity2, labels = data
        print(sentences.shape)
        print(sentence_lens.shape)
        print(sentence_locs_1.shape)
        print(sentence_locs_2.shape)
        print(entity1.shape)
        print(entity2.shape)
        print(labels.shape)
        cost, cnt = 0., 0
        for sample, num in self.get_batch_data(sentences, sentence_lens, sentence_locs_1, sentence_locs_2, entity1, entity2, labels, self.batch_size, True, self.dropout):
            _, loss, step, summary= self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op], feed_dict=sample)
            #print("entity_loc_1 "+str(entity_loc_1))
            #print("entity_loc_2 "+str(entity_loc_2))
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num
        _, train_acc = self.test(data)
        return cost / cnt, train_acc
    def test(self, data):
        sentences, sentence_lens, sentence_locs_1, sentence_locs_2, entity1, entity2, labels = data
        cost, acc, cnt = 0., 0, 0

        for sample, num in self.get_batch_data(sentences, sentence_lens, sentence_locs_1, sentence_locs_2, entity1, entity2, labels, int(len(sentences) / 2) + 1, False, 1.0):
            loss, accuracy, step, summary = self.sess.run([self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt
    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())
        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter' + str(self.timestamp), global_step=i)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; test-loss=%.6f; test-acc=%.6f;' % (str(i), train_loss, train_acc, test_loss, test_acc))
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))
        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
    def get_batch_data(self, sentences, sentence_lens, sentence_locs_1, sentence_locs_2, entity1, entity2, labels, batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(sentences), batch_size, is_shuffle):
            feed_dict = {
                self.sentences: sentences[index],
                self.sentence_lens: sentence_lens[index],
                self.sentence_entity_loc_1: sentence_locs_1[index],
                self.sentence_entity_loc_2: sentence_locs_2[index],
                self.sentence_entity1: entity1[index],
                self.sentence_entity2: entity2[index],
                self.labels: labels[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)