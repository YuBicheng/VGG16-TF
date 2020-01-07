import vggClass
import os
import sys
import tensorflow as tf

data_path = "./tfrecord/save.tfrecords"
save_path = './model/model.ckpt'
train_list = 'shuffle.txt'
learn_rate = 0.001
batch_size = 8
epoch = 100


def _parse_record(example_proto):
    features = {
        'img_data': tf.FixedLenFeature([], tf.string),
        'img_label': tf.FixedLenFeature([], tf.int64),
        'img_shape': tf.FixedLenFeature([3], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features=features)
    return parsed_features


def get_dataset(listfile):
    with open(train_list, 'r') as f:
        list = f.readlines()
    img_path_list = []
    label_list = []
    for l in list:
        l.strip('\n')
        img, label = l.split(' ')
        img_path_list.append(img)
        label_list.append(label)

    size = len(list)
    #print(list[-1]) 查看最后一个数据是否是空
    return img_path_list, label_list, size


if __name__=="__main__":
    #tfrecord方式读入数据
    # dataset = tf.data.TFRecordDataset(data_path)
    # dataset = dataset.map(_parse_record)
    # dataset = dataset.shuffle(2)
    # dataset = dataset.batch(batch_size,True)
    # iter = dataset.make_one_shot_iterator()
    img_list, label_list, size = get_dataset(train_list)

    #构建网络
    vgg = vggClass.Vgg16(5)
    vgg.setfilereadlist(img_list, label_list)
    vgg.train(batch_size, epoch, learn_rate, save_path)
