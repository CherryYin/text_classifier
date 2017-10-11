#coding:utf-8

import sys, os
import random
import sklearn, gensim
import tensorflow as tf
import numpy as np

from gensim.models.doc2vec import Doc2Vec

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def shuffle_dateset(root_ath, subList, K = 8):
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    i = 0
    for sub in subList:
        sub_path = os.path.join(root_ath, sub)
        text_files = os.listdir(sub_path)
        l = len(text_files)
        texts = []
        for j in range(l):
            with open(os.path.join(sub_path, text_files[j]), 'r') as f1:
                texts.append(f1.read().split(' '))
        #doc2vec_texts.extend(texts)
        split_num = (K -1) * len(text_files) / 8
        train_texts.extend(texts[ : split_num])
        train_sub_labels = np.zeros(shape=split_num)
        train_sub_labels[:] = i  #for knn
        train_labels.extend(list(train_sub_labels))
        test_texts.extend(texts[split_num : ])
        test_sub_labels = np.zeros(shape= l- split_num)
        test_sub_labels[:] = i   for knn
        test_labels.extend(list(test_sub_labels))
        i += 1

    print "train text size is %d, label size is %d"%(len(train_texts), len(train_labels))
    print "test text size is %d, label size is %d" % (len(test_texts), len(test_labels))

    return train_texts, train_labels, test_texts, test_labels


def doc2vec_train(X_train, size=200, epochs = 100):
    print "定义模型，制作词典..."
    model_dm = Doc2Vec(X_train, min_count = 10, window = 3, size = size, sample=1e-3, negative=5, workers=4)

    print "doc2vec训练..."
    model_dm.train(X_train, total_examples=model_dm.corpus_count, epochs=epochs)
    model_dm.save('model/model_sogouC')

    return model_dm

X_train = []
for i, text in enumerate(train_texts):
    documnet = TaggededDocument(text, tags=[i])
    X_train.append(documnet)

print "现在开始doc2vec,请稍等"
model_dm = doc2vec_train(X_train=X_train)
model_dm = Doc2Vec.load("model/model_sogouC")

l = len(test_texts)
correct_count = 0.0
for j in range(l):
    x_test = model_dm.infer_vector(test_texts[j])
    y_test = test_labels[j]
    sims = model_dm.docvecs.most_similar([x_test], topn=5)
    class_map = {}
    for count, sim in sims:
        y_ = train_labels[count]
        if y_ not in class_map:
            class_map[y_] = 0
        class_map[y_] += 1

    class_map_sorted = sorted(class_map.iteritems(), key=lambda d :d[1], reverse=True)
    y = class_map_sorted[0][0]
    if y == y_test:
        correct_count += 1.0

print "knn 的准确率为%f"%(correct_count / l)
