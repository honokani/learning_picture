import cv2, os, re, glob, math
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from libs import data_input as data_in

COMMON_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CAPTCHA_PICS_DIR_NAME = "pics"
CAPTCHA_PICS_DIR_PATH = os.path.join(COMMON_DIR_PATH, CAPTCHA_PICS_DIR_NAME)

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

    # Build CNN Model
    def buildModel(ph_x, drop_ratio, x_params, label_num):
        def makeBiasVal(shape):
            b = tf.constant(0.1, shape=shape)
            return tf.Variable(b)

        def makeWeightVal(shape):
            # initialize weight by gauusian
            w = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(w)

        def makeConvLayer(ch_in, ch_out, sz):
            # define stride size
            UNUSED_PARAM      = 1
            shift_span       = [UNUSED_PARAM, 1, 1, UNUSED_PARAM]
            conv_window_size = [sz, sz, ch_in, ch_out]
            # define size of w and d
            w = makeWeightVal(conv_window_size)
            b = makeBiasVal([ch_out])
            return lambda x: tf.nn.relu( b + tf.nn.conv2d(x, w, shift_span, padding='SAME') )

        def makePoolLayer(sz):
            UNUSED_PARAM      = 1
            pool_window_size = [UNUSED_PARAM, sz, sz, UNUSED_PARAM]
            shift_span       = [UNUSED_PARAM, sz, sz, UNUSED_PARAM]
            return lambda x: tf.nn.max_pool(x, ksize=pool_window_size, strides=shift_span, padding='SAME')

        def makeFullyConnectedLayer(in_sz, out_sz, f):
            w = makeWeightVal([in_sz, out_sz])
            b = makeBiasVal([out_sz])
            return lambda x: f( b + tf.matmul(tf.reshape(x, [-1, in_sz]), w) )

        def makeFullyConnectedLayerWithRelu(in_sz, out_sz):
            return makeFullyConnectedLayer(in_sz, out_sz, tf.nn.relu)

        def makeFullyConnectedLayerWithSoftmax(in_sz, out_sz):
            return makeFullyConnectedLayer(in_sz, out_sz, tf.nn.softmax)

        def makeDropoutLayer(dr_ratio):
            return lambda x: tf.nn.dropout(x, 1-dr_ratio)

        # get size of flatten x and label
        x_w, x_h, x_ch = x_params
        pic_flat_size = x_w * x_h
        img_total_pixcel = pic_flat_size * x_ch

        # reshape input x
        x_in = tf.reshape(ph_x, [-1, x_w, x_h, x_ch])
        # 1st conv-pool layer set
        h1_out_ch = 32
        h1_w_size = 5

        with tf.name_scope('conv1') as scope:
            h1_c = makeConvLayer(x_ch, h1_out_ch, h1_w_size)(x_in)
        h1_p_size = 2
        with tf.name_scope('pool1') as scope:
            h1_p = makePoolLayer(h1_p_size)(h1_c)
        # 2nd conv-pool layer set
        h2_out_ch = 64
        h2_w_size = 5
        with tf.name_scope('conv2') as scope:
            h2_c = makeConvLayer(h1_out_ch, h2_out_ch, h2_w_size)(h1_p)
        h2_p_size = 2
        with tf.name_scope('pool2') as scope:
            h2_p = makePoolLayer(h2_p_size)(h2_c)
        # full connected layer and drop layer
        fc1_pic_size = int(pic_flat_size / math.pow(h1_p_size * h2_p_size, 2))
        fc1_pic_flatten = fc1_pic_size * h2_out_ch
        fc1_layer_len = 1024
        with tf.name_scope('fullyconn') as scope:
            fc1_out = makeFullyConnectedLayerWithRelu(fc1_pic_flatten, fc1_layer_len)(h2_p)
        with tf.name_scope('drop') as scope:
            fc1_rest = makeDropoutLayer(drop_ratio)(fc1_out)
        # normalization
        fc2_layer_len = label_num
        with tf.name_scope('softmax') as scope:
            fc2_out = makeFullyConnectedLayerWithSoftmax(fc1_layer_len, fc2_layer_len)(fc1_rest)
        # 

        return fc2_out

    def defineLoss(result, correct):
        # culc cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=result)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss_value", loss)
        return tf.reduce_mean(loss)

    def minimizeLoss(loss):
        train_ratio = 0.0000002
        return tf.train.AdamOptimizer( train_ratio ).minimize( loss )

    def checkAccuracy(result, correct):
        prediction = tf.equal(tf.argmax(result, 1), tf.argmax(correct, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def runLearning(ops, itors, inputs, valid_ratio):
        e = np.array([])
        npSplit=lambda n,t:(e,t)if(len(t)<1 or n<1)else((t,e)if(len(t)<n)else(t[:n],t[n:]))

        # expand inputs
        training_step, check_step = ops[0], ops[1]
        epoch_sz, minibatch_sz = itors[0], itors[1]

        # count train data size for accuracy test
        xs, ys = inputs[0], inputs[1]
        input_sz = len(ys)
        test_sz  = int(input_sz * valid_ratio)
        if(test_sz<1):
            test_sz = 1
        elif(input_sz<test_sz):
            test_sz = input_sz -1

        # split test_data and train_data
        trains_num = np.random.permutation(input_sz)
        tgt_to_accuracy, tgt_to_train = npSplit(test_sz, trains_num)
        test_x   = xs[tgt_to_accuracy]
        test_y   = ys[tgt_to_accuracy]
        train_x  = xs[tgt_to_train]
        train_y  = ys[tgt_to_train]
        train_sz = input_sz - test_sz

        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(COMMON_DIR_PATH, sess.graph)

        # run epoch-size time
        for ep in range(epoch_sz):
            rest_tgt = np.random.permutation(train_sz)
            # rest_tgt = np.array(list(range(train_sz)))
            while(0 < rest_tgt.size):
                tgt_to_pick, rest_tgt = npSplit(minibatch_sz, rest_tgt)
                batch_x = train_x[tgt_to_pick]
                batch_y = train_y[tgt_to_pick]
                sess.run(training_step, feed_dict={ph_x:batch_x, ph_y:batch_y, drop_ratio:0.1})
                if(len(rest_tgt)<minibatch_sz):
                    break
            if(ep%1 == 0):
                noDrop = 0.0
                trn=sess.run(check_step,feed_dict={ph_x:batch_x, ph_y:batch_y, drop_ratio:noDrop})
                acc=sess.run(check_step,feed_dict={ph_x:test_x, ph_y:test_y, drop_ratio:noDrop})
                sums=sess.run(summary_op,feed_dict={ph_x:test_x, ph_y:test_y, drop_ratio:noDrop})
                summary_writer.add_summary(sums,ep)
                print("ep: {}, train: {}, accuracy: {}".format(ep, trn, acc))


    # get input datas
    in_xs, in_ys = getInputDatas()
    # get size of each pic and labels
    x_w  = in_xs[0].shape[0]
    x_h  = in_xs[0].shape[1]
    if(x_w != x_h):
        print("error! : data shape is not square.")
        return(-1)
    x_ch = in_xs[0].shape[3]
    label_num = in_ys[0].shape[0]

    graph_cnn = tf.Graph()
    with graph_cnn.as_default():
        # define place holder
        ph_x = tf.placeholder(tf.float32, shape=[None, x_w, x_h, 1, x_ch])
        ph_y = tf.placeholder(tf.float32, shape=[None, label_num])
        # (1-keep ratio) placeholder
        drop_ratio = tf.placeholder(tf.float32)

        # build cnn model, get placeholder, and result logits
        result_logits = buildModel(ph_x, drop_ratio, (x_w,x_h,x_ch), label_num)
        # get loss from diff between "input label" and "result through cnn model"
        loss_val = defineLoss(result_logits, ph_y)
        # training model step
        training_step = minimizeLoss(loss_val)
        # check accuracy
        check_step = checkAccuracy(result_logits, ph_y)

        # ready saver
        saver = tf.train.Saver()
        # initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # do learn
        epoch_sz = 1000
        minibatch_sz = 100
        runLearning( (training_step, check_step)
                   , (epoch_sz, minibatch_sz)
                   , (in_xs, in_ys)
                   , 0.1
                   )

        # save model
        model_name = "model.ckpt"
        model_path = os.path.join(COMMON_DIR_PATH, model_name)
        save_path = saver.save(sess, model_path)


if __name__ == '__main__':
    print("start!")
    main()
    print("end!")

