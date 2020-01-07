import tensorflow as tf
import common
import cv2
import numpy as np

class Vgg16(object):
    def __init__(self, classes_num):
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input-x")
        self.y = tf.placeholder(tf.float32, [None, classes_num], name="input-y")
        self.classnum = classes_num
        self.image_list = None
        self.label_list = None

    def inference_vgg16(self, train=True):
        #卷积1
        conv1 = common.convolutional(self.x, [3, 3, 3, 64], name="conv1", trainable=True)
        conv2 = common.convolutional(conv1, [3,3,64,64], name="conv2", trainable=True)
        #池化1
        pool1 = common.pool(conv2)
        conv3 = common.convolutional(pool1, [3,3,64,128], name="conv3", trainable=True)
        conv4 = common.convolutional(conv3, [3,3,128,128], name="conv4", trainable=True)
        #池化2
        pool2 = common.pool(conv4)
        conv5 = common.convolutional(pool2, [3,3,128,256], name="conv5", trainable=True)
        conv6 = common.convolutional(conv5, [3,3,256,256], name="conv6", trainable=True)
        conv7 = common.convolutional(conv6, [3,3,256,256], name="conv7", trainable=True)
        #池化3
        pool3 = common.pool(conv7)
        conv8 = common.convolutional(pool3, [3,3,256, 512], name="conv8", trainable=True)
        conv9 = common.convolutional(conv8, [3,3,512,512], name="conv9", trainable=True)
        conv10 = common.convolutional(conv9, [3,3,512,512], name="conv10", trainable=True)
        #池化4
        pool4 = common.pool(conv10)
        conv11 = common.convolutional(pool4, [3,3,512,512], name="conv11", trainable=True)
        conv12 = common.convolutional(conv11, [3, 3, 512, 512], name="conv12", trainable=True)
        conv13 = common.convolutional(conv12, [3, 3, 512, 512], name="conv13", trainable=True)
        pool5 = common.pool(conv13)
        pool_shape = pool5.get_shape().as_list()
        #5次池化 有填充卷积，输入维度[224，224，3] pool5输出为[7 ,7, 512]
        input_node = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool5, [-1, input_node])
        fc1 = common.fullconnect(reshaped, 4096, "fc1", train)
        fc2 = common.fullconnect(fc1, 4096, "fc2", train)
        fc3 = common.fullconnect(fc2, self.classnum, "fc3", train)
        return fc3

    def softmax_or_loss(self, input, train=True):
        if train:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=input)
            loss = tf.reduce_mean(cross_entropy)
            return loss
        else:
            return tf.nn.softmax(input, axis=self.classnum)

    def get_data_from_file(self, batch_size, i):
        size = len(self.image_list)
        times = size//batch_size
        print(i, '/', times)
        if i + 1 > times:
            raise StopIteration ('超出')
        point = i*batch_size
        img_l = self.image_list[point:point + batch_size]
        img = []
        for g in img_l:
            image = cv2.imread(g,1)
            image = cv2.resize(image,(224,224))
            img.append(image)
        img = np.array(img)
        label_list = self.label_list[point:point + batch_size]
        label = np.zeros([batch_size, self.classnum])
        for j in range(len(label_list)):
            label[j][int(label_list[j])] = 1
        label = np.array(label)
        #print(img.shape, label.shape)
        return img, label

    def setfilereadlist(self, image_list, label):
        self.image_list = image_list
        self.label_list = label



    def train(self, bitchsize, epoch, learn_rate, save_path):
        output = self.inference_vgg16(True)
        cross_entropy = self.softmax_or_loss(output, True)
        tf.add_to_collection("loss", cross_entropy)
        loss = tf.add_n(tf.get_collection("loss"))
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        #dataset = dataset
        write = tf.summary.FileWriter("./log/", tf.get_default_graph())
        write.close()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                print('第%d代' % int(i + 1))
                t = 0
                while True:
                    try:
                        #print('第%d个bitch' % t)
                        x,y = self.get_data_from_file(bitchsize, t)
                        t = t+1
                    except StopIteration:
                        break
                    l, _ = sess.run([loss, train_step], feed_dict={self.x:x, self.y:y})
                if i%100 == 0:
                    print("在训练%d 代后，损失为 %f" % (i, l))
            saver.save(sess, save_path)
            print("训练完成")


