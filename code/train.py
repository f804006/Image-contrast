import random

from model import siamese
import csv
from PIL import Image
import numpy as np
from keras.optimizers import SGD, Adam
from utils import cvtColor, preprocess_input
from utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize

input_shape = [105, 105]

resize_crop = RandomResizedCrop(input_shape)
policy = ImageNetPolicy()
resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
center_crop = CenterCrop(input_shape)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def data_autoAugment(image, random):
    if not random:
        image = resize(image)
        image = center_crop(image)
        return image

    image = resize_crop(image)

    # 随机翻转图像
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 随机增强
    image = policy(image)
    return image


def image_process(image_url):
    # 打开图片
    image = Image.open(image_url)
    # 都转为三通道彩色图像
    image = cvtColor(image)

    # 数据增强
    image = data_autoAugment(image, True)

    image = preprocess_input(np.array(image).astype(np.float32))
    return image


def load_dataset(dataset_path, val_ratio):
    # 存放待对比的图片1, shape: n*[128*128*3]
    image_data1 = []
    # 存放待对比的图片2, shape: n*[128*128*3]
    image_data2 = []
    # 存放两个图片的相似性 0不相似，1相似
    label = []
    firstLine = True
    with open(dataset_path + "/annos.csv", 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if firstLine:
                firstLine = False
                continue
            label.append(float(line[1]))
            image1_url = dataset_path + "/data/" + line[0] + '/a.jpg'
            image2_url = dataset_path + "/data/" + line[0] + '/b.jpg'
            img1 = image_process(image1_url)
            img2 = image_process(image2_url)
            image_data1.append(img1)
            image_data2.append(img2)

    random.seed(1)
    shuffle_index = np.arange(len(label), dtype=np.int32)
    random.shuffle(shuffle_index)
    random.seed(None)

    image_data1 = np.array(image_data1, dtype=np.object)
    image_data2 = np.array(image_data1, dtype=np.object)
    label = np.array(label)
    image_data1 = image_data1[shuffle_index]
    image_data2 = image_data2[shuffle_index]
    label = label[shuffle_index]

    num_train = int(len(label) * (1 - val_ratio))
    train_data1 = image_data1[:num_train]
    train_data2 = image_data2[:num_train]
    train_label = label[:num_train]

    val_data1 = image_data1[num_train:]
    val_data2 = image_data2[num_train:]
    val_label = label[num_train:]

    return train_data1, train_data2, train_label, val_data1, val_data2, val_label


def getTrainConf():
    pass


if __name__ == "__main__":
    dataset_path = "../init_data/toUser/train"

    val_ratio = 0.2
    train_data1, train_data2, train_label, val_data1, val_data2, val_label = load_dataset(dataset_path, val_ratio)
    model_path = "model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model = siamese(input_shape=[input_shape[0], input_shape[1], 3])
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # Adam学习率
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    momentum = 0.9
    batch_size = 32

    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=Init_lr, momentum=momentum, nesterov=True),
                  metrics=["binary_accuracy"])
    model.summary()

    model.fit(x=[train_data1, train_data2], y=train_label, batch_size=32, epochs=50)
