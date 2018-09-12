import tensorflow as tf
import numpy as np
from utlis_li import get_data_info, read_data, load_word_embeddings
from model_li import WAM

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('word2id', 0, 'kernel')
tf.app.flags.DEFINE_integer('id2word', 0, 'kernel')
tf.app.flags.DEFINE_integer('max_sentence_len', 0, 'kernel')
tf.app.flags.DEFINE_integer('max_entity_len', 0, 'kernel')
tf.app.flags.DEFINE_integer('word2vec', 0, 'kernel')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 20, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 4, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_hop', 3, 'number of hop')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_fname', 'data/glove.6B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('train_raw', 'data/raw_train.txt', 'training file name')
tf.app.flags.DEFINE_string('test_raw', 'data/raw_test.txt', 'testing file name')
tf.app.flags.DEFINE_string('data_info', 'data/data_info.txt', 'the file saving data information')
tf.app.flags.DEFINE_string('train_process_dis', 'data/process_train_dis.txt', 'the file saving training data for dis')
tf.app.flags.DEFINE_string('train_process_med', 'data/process_train_med.txt', 'the file saving training data for med')
tf.app.flags.DEFINE_string('test_process_dis', 'data/process_test_dis.txt', 'the file saving testing data for dis')
tf.app.flags.DEFINE_string('test_process_med', 'data/process_test_med.txt', 'the file saving testing data for med')


def main(_):
    print('Loading data info ...')
    FLAGS.word2id, FLAGS.id2word, FLAGS.max_sentence_len, FLAGS.max_entity_len = get_data_info(FLAGS.train_raw, FLAGS.test_raw, FLAGS.data_info, FLAGS.pre_processed)

    print('Loading training data and testing data ...')
    #train_process_d = read_data(FLAGS.train_raw, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_entity_len, FLAGS.train_process_dis, FLAGS.pre_processed, 0)
    #train_process_m = read_data(FLAGS.train_raw, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_entity_len, FLAGS.train_process_med, FLAGS.pre_processed, 1)
    #test_process_d = read_data(FLAGS.test_raw, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_entity_len, FLAGS.test_process_dis, FLAGS.pre_processed, 0)
    #test_process_m = read_data(FLAGS.test_raw, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_entity_len, FLAGS.test_process_med, FLAGS.pre_processed, 1)
    
    sentences, entity1, sentence_lens, sentence_locs, labels, entity_loc1 = read_data(FLAGS.train_raw, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_entity_len, FLAGS.train_process_dis, FLAGS.pre_processed, 0)
    sentences, entity2, sentence_lens, sentence_locs, labels, entity_loc2 = read_data(FLAGS.train_raw, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_entity_len, FLAGS.train_process_med, FLAGS.pre_processed, 1)
    train_data = sentences, sentence_lens, entity_loc1, entity_loc2, entity1, entity2, labels
    print(entity_loc1)
    sentences, entity1, sentence_lens, sentence_locs, labels, entity_loc1 = read_data(FLAGS.test_raw, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_entity_len, FLAGS.test_process_dis, FLAGS.pre_processed, 0)
    sentences, entity2, sentence_lens, sentence_locs, labels, entity_loc2 = read_data(FLAGS.test_raw, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_entity_len, FLAGS.test_process_med, FLAGS.pre_processed, 1)
    test_data = sentences, sentence_lens, entity_loc1, entity_loc2, entity1, entity2, labels

    print('Loading pre-trained word vectors ...')
    FLAGS.word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    with tf.Session() as sess:
        model = WAM(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)

if __name__ == '__main__':
    tf.app.run()
