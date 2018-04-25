import cv2, os, re, glob
import numpy as np
import tensorflow as tf
from libs import data_input as data_in

COMMON_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CAPTCHA_PICS_DIR_NAME = "pics"
CAPTCHA_PICS_DIR_PATH = os.path.join(COMMON_DIR_PATH, CAPTCHA_PICS_DIR_NAME)

def makeBiasVal(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

def makeWeightVal(shape):
    # initialize weight by gauusian
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)

def makeConvLayer(ch_in, ch_out, size):
    # define stride size
    UNUSED_PARAM      = 1
    shift_span       = [UNUSED_PARAM, 1, 1, UNUSED_PARAM]
    conv_window_size = [size, size, ch_in, ch_out]
    # define size of w and d
    w = makeWeightVal(conv_window_size)
    b = makeBiasVal([ch_out])
    return lambda x: tf.nn.relu( b + tf.nn.conv2d(x, w, shift_span, padding='SAME') )

def makePoolLayer(size):
    UNUSED_PARAM      = 1
    pool_window_size = [UNUSED_PARAM, size, size, UNUSED_PARAMj]
    shift_span       = [UNUSED_PARAM, size, size, UNUSED_PARAM]
    return lambda x: tf.nn.max_pool(x, ksize=pool_window_size, strides=shift_span, padding='SAME')


def makeFullyConnectedLayer(ch_in, ch_out, size, f):
    ch_in_flatten = size * ch_in
    w = makeWeightVal([ch_in_flatten, ch_out])
    b = makeBiasVal([ch_out])
    return lambda x: f( b + tf.matmal(tf.reshape(x, [-1, ch_in_flatten]), w) )

def makeFullyConnectedLayerWithRelu(ch_in, ch_out, size):
    return makeFullyConnectedLayer(ch_in, ch_out, size, tf.nn.relu)

def makeFullyConnectedLayerWithSoftmax(ch_in, ch_out, size):
    return makeFullyConnectedLayer(ch_in, ch_out, size, tf.nn.softmax)

def makeDropoutLayer(dr_ratio):
    return lambda x: tf.dropout(x, dr_datio)


# Build CNN Model
def buildModel(input_xs, input_ys):
    # get size of each pic
    w = input_xs[0].shape[0]
    h = input_xs[0].shape[1]
    if(w != h):
        print("error! : data shape is not square.")
    # get size of flatten x and label
    pic_flat_size = w * h
    color_ch = input_xs[0].shape[3]
    img_total_pixcel = pic_flat_size * color_ch
    label_num        = len( input_ys )

    # define place holder
    x = tf.placeholder(tf.float32, shape=[None, img_total_pixcel])
    l = tf.placeholder(tf.float32, shape=[None, label_num])

    # reshape input x
    x_image = tf.reshape(x, [-1, w, h, color_ch])
    # 1st conv-pool layer set
    h1_out_ch = 32
    h1_w_size = 5
    h1_c = makeConvLayer(color_ch, h1_out_ch, h1_w_size,)(x)
    h1_p_size = 2
    h1_p = makePoolLayer(h1_p_size)(h1_c)
    # 2nd conv-pool layer set
    h2_out_ch = 64
    h2_w_size = 5
    h2_c = makeConvLayer(h1_out_ch, h2_out_ch, h2_w_size)(h1_p)
    h2_p_size = 2
    h2_p = makePoolLayer(h2_p_size)(h2_c)
    # full connected layer
    fc1_out_ch = 1024
    fc1_input_size = pic_flat_size / (h1_p_size * h2_p_size)
    fc1_out = makeFullyConnectedLayerWithRelu(h2_out_ch, fc1_out_ch, fc_input_size)(h2_p)
    # dropout layer
    drop_rate = 0.1
    fc1_rest = makeDropoutLayer(drop_rate)(fc1_out)
    # full connected layer
    fc2_out_ch = label_num
    fc2_out = makeFullyConnectedLayer(fc1_out_ch, fc2_out_ch, fc_input_size)(fc1_rest)


def getInputDatas():
    # input dataset and their labels
    ds, ls = data_in.getDataset(CAPTCHA_PICS_DIR_PATH)

    # make label set
    lSet = set()
    enUnique = lambda seq: [ x for x in seq if x not in lSet and not lSet.add(x) ]
    lList=enUnique(ls)
    lList.sort()
    # make one-hot vector
    makeOneHot = lambda x: [ 1 if y==x else 0  for y in lList ]
    ls = list( map(makeOneHot,ls) )
    # ready to input
    data_l = np.array(ls)
    # make data set
    data_x = np.array(ds, dtype="float") / 255.0

    return data_x, data_l


def main():
    input_xs, input_ys = getInputDatas()
    buildModel(input_xs, input_ys)

    graph_cnn = tf.Graph()
    with graph_cnn.as_default():
        return 0
    return 0


if __name__ == '__main__':
    main()

