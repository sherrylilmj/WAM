import os
import re
import numpy as np
from errno import ENOENT
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")
def get_data_info(train_raw, test_raw, data_info, pre_processed):

    word2id, max_sentence_len, max_aspect_len = {}, 0, 0
    word2id['<pad>'] = 0

    if pre_processed:
        if not os.path.isfile(data_info):
            raise IOError(ENOENT, 'Not a file', data_info)
        with open(data_info, 'r',encoding='utf-8') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 3:    # 第一行，记录了句子和实体的最大长度
                    max_sentence_len = int(content[1])
                    max_aspect_len = int(content[2])
                else:
                    word2id[content[0]] = int(content[1])    # 词转成id

    else:
        if not os.path.isfile(train_raw):
            raise IOError(ENOENT, 'Not a file', train_raw)
        if not os.path.isfile(test_raw):
            raise IOError(ENOENT, 'Not a file', test_raw)


        words = []

        with open(train_raw, 'r', encoding = 'utf-8') as f:
            for line in f:
                if(line[0] == '\"'):
                    doc = nlp(line[1: -1])
                    words.extend([sp.text.lower() for sp in doc])
                    if len(doc) > max_sentence_len:
                        max_sentence_len = len(doc)

                elif(line[0] == '('):
                    lines = line[1: -1].split(',')
                    entity1 = nlp(lines[0])    # 病名
                    entity2 = nlp(lines[1])    # 药名
                    # 此时都把他们看作实体
                    if len(entity1) > max_aspect_len:
                        max_aspect_len = len(entity1)
                    if len(entity2) > max_aspect_len:
                        max_aspect_len = len(entity2)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)

        with open(test_raw, 'r', encoding = 'utf-8') as f:
            for line in f:
                if(line[0] == '\"'):
                    doc = nlp(line[1: -1])
                    words.extend([sp.text.lower() for sp in doc])
                    if len(doc) > max_sentence_len:
                        max_sentence_len = len(doc)

                elif(line[0] == '('):
                    lines=line[1: -1].split(',')
                    entity1 = nlp(lines[0])
                    entity2 = nlp(lines[1])
                    if len(entity1) > max_aspect_len:
                        max_aspect_len = len(entity1)
                    if len(entity2) > max_aspect_len:
                        max_aspect_len = len(entity2)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)

        # 写入data_info
        with open(data_info, 'w', encoding = 'utf-8') as f:
            f.write('length %s %s\n' % (max_sentence_len, max_aspect_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))
                
    print('There are %s words in the dataset, the max length of sentence is %s, and the max length of entity is %s' % (len(word2id), max_sentence_len, max_aspect_len))
    return word2id, max_sentence_len, max_aspect_len



def get_loc_info(doc, target):

    match = 0    # 标志位
    target_loc = 0
    cnt = 0
    aspect = []
    for each_word in doc:
        for target_word in target:
            if each_word.text.lower() == target_word.text.lower():
                match = 1
                aspect.append(each_word.i)
                cnt = cnt + 1
                target_loc += each_word.i
    if match == 0:    # null时补0
        aspect.append(0)
    else:
        target_loc = target_loc / cnt
    loc_info = []
    for _i, each in enumerate(doc):
        loc_info.append((_i - aspect[0]) / len(doc))    # 计算相对偏移量   
    return loc_info, target_loc

def read_data(raw, word2id, max_sentence_len, max_aspect_len, process, pre_processed, choice):

    sentences, entity, sentence_lens, sentence_locs, labels, entity_loc = [], [], [], [], [], []
    if pre_processed:
        if not os.path.isfile(process):
            raise IOError(ENOENT, 'Not a file', process)
        lines = open(process, 'r', encoding = "utf-8").readlines()
        for i in range(0, len(lines), 6):
            sentences.append(lines[i][:-1])
            entity.append(lines[i + 1][:-1])
            sentence_lens.append(lines[i + 2][:-1])
            entity_loc.append(lines[i + 3][:-1])
            sentence_locs.append(lines[i + 4][:-1])
            labels.append(lines[i + 5][:-1])
    else:
        if not os.path.isfile(raw):
            raise IOError(ENOENT, 'Not a file', raw)

        f = open(process, 'w', encoding = 'utf-8')

        with open(raw, 'r', encoding='utf-8') as ff:
            for line in ff:
                if(line[0] == '\"'):
                    texts = nlp(line[1: -1])
                    if len(texts.text.strip()) != 0:
                        real_len = len(texts)
                        ids = []
                        for each_word in texts:    # 句子转化成id的集合
                            if each_word.text.lower() in word2id:
                                ids.append(word2id[each_word.text.lower()])

                elif(line[0] == '('):
                    lines=line[1: -1].split(',')
                    if(len(lines) > 2):
                        entityy = nlp(lines[choice])    # choice = 0表示病名，choice = 1表示药名
                        entityy_ids = []
                        for sub_entity in entityy:
                            if sub_entity.text.lower() in word2id:
                                entityy_ids.append(word2id[sub_entity.text.lower()])

                        sentences.append(ids + [0] * (max_sentence_len - len(ids)))    # 内容填充0
                        f.write("%s\n" % sentences[-1])
                        entity.append(entityy_ids + [0] * (max_aspect_len - len(entityy_ids)))    # 位置填充1
                        f.write("%s\n" % entity[-1])
                        sentence_lens.append(len(texts))    # 真实长度
                        f.write("%s\n" % sentence_lens[-1])

                        loc_info,e_loc = get_loc_info(texts, entityy)
                        entity_loc.append(e_loc)
                        f.write("%s\n" % entity_loc[-1])
                        sentence_locs.append(loc_info + [1] * (max_sentence_len - len(loc_info)))
                        f.write("%s\n" % sentence_locs[-1])

                        polarity = lines[2]
                        if(len(polarity) > 1):
                            polarity = polarity[0]
                        if polarity == '1':
                            labels.append([1, 0, 0, 0])
                        elif polarity == '2':
                            labels.append([0, 1, 0, 0])
                        elif polarity == "3":
                            labels.append([0, 0, 1, 0])
                        elif polarity == '0':
                            labels.append([0, 0, 0, 1])
                        else:
                            print("error",lines[2])
                            print("error line",line[1:-2])
                        f.write("%s\n" % labels[-1])
        f.close()
    print("Read %s sentences from %s" % (len(sentences), raw))
    return np.asarray(sentences), np.asarray(entity), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels), np.asarray(entity_loc)



def load_word_embeddings(fname, embedding_dim, word2id):    # fname是glove（300维）

    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.normal(0, 0.05, [len(word2id), embedding_dim])
    oov = len(word2id)

    with open(fname, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            content = line.strip().split()
            if content[0] in word2id:    # 第一个是单词
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))    # 转换成embedding
                oov = oov - 1

    word2vec[word2id['<pad>'], :] = 0

    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec



def get_batch_index(length, batch_size, is_shuffle = True):

    index = list(range(length))

    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size: (i + 1) * batch_size]