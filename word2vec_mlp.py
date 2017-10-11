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
        split_num = (K -1) * len(text_files) / 8
        train_texts.extend(texts[ : split_num])
        train_sub_labels = np.zeros(shape=[split_num, 9])
        train_sub_labels[:, i] = 1.0              #for mlp
        train_labels.extend(list(train_sub_labels))
        test_texts.extend(texts[split_num : ])
        test_sub_labels = np.zeros(shape=[l - split_num, 9])
        test_sub_labels[:, i] = 1.0       #for mlp
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
    model_dm.save('model/model_sogouC_mlp')

    return model_dm


def mlp(train_texts, train_labels, test_texts, test_labels, model_dm):
    # 输入层cell数量
    in_units = 200
    # 隐含层数量，可以是自定义的，一般比输入层少。
    h1_units = 100
    with tf.Graph().as_default():
        # 输入层W1的shape从[784, 10]改为[784,300],b1也从10改为300
        W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
        b1 = tf.Variable(tf.zeros([h1_units]))
        W2 = tf.Variable(tf.zeros([h1_units, 9]))
        b2 = tf.Variable(tf.zeros([9]))

        x = tf.placeholder(tf.float32, [None, in_units])
        # dropout的保留百分比，用一个placeholder占位，使其可以自由配置。
        # 如果keep_prob被设为0.75,那么随机选择75%的节点信息有效，25%的节点的信息丢弃。
        keep_prob = tf.placeholder(tf.float32)

        # input->hidden采用relu激活。
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        # input->hidden启用dropout
        hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)
        y = tf.nn.softmax(tf.matmul(hidden1_dropout, W2) + b2)
        print y.shape

        y_ = tf.placeholder(tf.float32, [None, 9])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        l = len(train_texts)
        for i in range(1000):
            indexes = [random.randint(0, l-1) for _ in range(40)]
            x_batches, y_batches = [], []
            for index in indexes:
                x_batches.append(model_dm.infer_vector(train_texts[index]))
                y_batches.append(train_labels[index])
            x_batches = np.array(x_batches)
            y_batches = np.array(y_batches)
            sess.run(train_step, {x: x_batches, y_: y_batches, keep_prob: 1.0})

        # 训练时，一般dropout百分比小于1,测试时，一般等于1.
        l = len(test_texts)
        x_test=np.array([model_dm.infer_vector(test_texts[j]) for j in range(l)])
        y_test = np.array(test_labels)
        acc = sess.run(accuracy, {x: x_test, y_: y_test, keep_prob: 1.0})
        print "mlp 准确率为%f"%(acc)


train_texts, train_labels, test_texts, test_labels = shuffle_dateset('SogouC.reduced/splitted',
                                                                                    ['C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024'])
X_train = []
for i, text in enumerate(train_texts):
    documnet = TaggededDocument(text, tags=[i])
    X_train.append(documnet)

j = len(X_train) - 1
for i, text in enumerate(test_texts):
    documnet = TaggededDocument(text, tags=[j+i])
    X_train.append(documnet)
print len(X_train)

print "现在开始doc2vec,请稍等"
model_dm = doc2vec_train(X_train=X_train)
model_dm = Doc2Vec.load("model/model_sogouC_mlp")


print "现在开始mlp,请稍等"
mlp(train_texts, train_labels, test_texts, test_labels, model_dm)

