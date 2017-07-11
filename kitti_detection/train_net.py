"""
Script to train the network
"""

import os
import random
import time

import kitti_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

VERBOSE = True
DATA_CONST = kitti_data.LabeledImageData


class Profiler:
    """
    Simple profiler
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.time_enter = time.time()

    def __exit__(self, dummy, dummmy, dummmmy):
        time_exit = time.time()
        print('Running time for {} is {}'.format(self.name, time_exit - self.time_enter))


def flip_dim2(ndarray):
    """
    Flip x coordinate for training data
    :param ndarray: 4d array
    :return: flipped 4d array
    """
    return ndarray[:, :, ::-1, ...]


def convolution_layer(net_input, filter_shape, layer_name):
    """
    TensorFlow convolution layer
    :param net_input: Tensor
    :param filter_shape: conv2d filter shape
    :param layer_name: Name of the layer
    :return: Output tensor
    """
    MEAN = 0.
    # Very important
    STDDEV = 0.05
    assert len(filter_shape) == 4
    filter_x, filter_y, filter_z, num_filter = filter_shape

    weight = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=STDDEV, shape=filter_shape),
                         name='{}_W'.format(layer_name))
    tf.add_to_collection('vars', weight)

    bias = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=STDDEV, shape=[num_filter]),
                       name='{}_b'.format(layer_name))
    tf.add_to_collection('vars', bias)

    conv = tf.nn.conv2d(input=net_input, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    conv = tf.nn.relu(conv)

    return conv


def pooling_layer(net_input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    """
    TensorFlow pooling layer
    :param net_input: Input tensor
    :param ksize: kernel size of pooling
    :param strides: stride of pooling
    :return: Tensor after pooling
    """
    return tf.nn.max_pool(net_input, ksize=ksize, strides=strides, padding='VALID')


def fully_connected_layer(net_input, num_output, layer_name):
    """
    TensorFlow fully connected layer
    Implement in sliding window style
    :param net_input: Input tensor
    :param num_output: num of output elements
    :param layer_name: name of the layer
    :return: Output tensor
    """

    MEAN = 0.
    # Very important
    STDDEV = 0.05
    fc_filter_shape = (1, 1, int(net_input.get_shape()[3]), num_output)

    weight = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=STDDEV, shape=fc_filter_shape),
                         name='{}_W'.format(layer_name))
    tf.add_to_collection('vars', weight)
    bias = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=STDDEV, shape=[num_output]),
                       name='{}_b'.format(layer_name))
    tf.add_to_collection('vars', bias)

    conv = tf.nn.conv2d(input=net_input, filter=weight, strides=(1, 1, 1, 1), padding='SAME') + bias
    conv = tf.nn.relu(conv)

    return conv


def output_layer(net_input, num_output_channels, layer_name):
    """
    Output layer
    Basically, fully_connected_layer without relu
    :param net_input: Input tensor
    :param num_output_channels: num of output elements
    :param layer_name: name of the layer
    :return: Output tensor
    """
    MEAN = 0.
    # Very important
    STDDEV = 0.05

    W_output = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=STDDEV,
                                               shape=(1, 1, int(net_input.get_shape()[3]), num_output_channels)),
                           name='{}_W'.format(layer_name))
    tf.add_to_collection('vars', W_output)

    b_output = tf.Variable(tf.zeros(num_output_channels), name='{}_W'.format(layer_name))
    tf.add_to_collection('vars', b_output)

    y_predict = tf.add(tf.nn.conv2d(input=net_input, filter=W_output, strides=(1, 1, 1, 1), padding='SAME'), b_output,
                       name='{}_output'.format(layer_name))
    return y_predict


def Net():
    """
    Define the net structure. Right now, it is a shallow "deep neural network"
    If you got a cluster, design a metric and find the best net.
    :return:
        heatmap_output: Tensor represents classification output
        regression_output: Tensor represents regression output
        net_input: Input to this net
        keep_prob: dropout probability
    """
    # Input to classifier
    net_input = tf.placeholder(tf.float32, (None, kitti_data.IMAGE_SIZE[0], kitti_data.IMAGE_SIZE[1], 3),
                               name='net_input')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    conv1 = convolution_layer(net_input, filter_shape=(5, 5, 3, 16), layer_name='conv1')
    conv1_pool = pooling_layer(conv1)

    conv2 = convolution_layer(conv1_pool, filter_shape=(5, 5, 16, 32), layer_name='conv2')
    conv2_pool = pooling_layer(conv2)

    conv3 = convolution_layer(conv2_pool, filter_shape=(5, 5, 32, 64), layer_name='conv3')
    conv3_pool = pooling_layer(conv3)

    conv4 = convolution_layer(conv3_pool, filter_shape=(5, 5, 64, 128), layer_name='conv4')
    conv4_pool = pooling_layer(conv4)

    conv5 = convolution_layer(conv4_pool, filter_shape=(5, 5, 128, 256), layer_name='conv5')

    fc1 = fully_connected_layer(conv5, num_output=1000, layer_name='fc1')
    fc1_dropped = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc2 = fully_connected_layer(fc1_dropped, num_output=1000, layer_name='fc2')
    fc2_dropped = tf.nn.dropout(fc2, keep_prob=keep_prob)

    heatmap_output = output_layer(fc2_dropped, num_output_channels=DATA_CONST.NUM_HEATMAP_CHANNELS,
                                  layer_name='heatmap')
    regression_output = output_layer(fc2_dropped, num_output_channels=DATA_CONST.NUM_REGRESSION_CHANNELS,
                                     layer_name='regression')
    return heatmap_output, regression_output, net_input, keep_prob


def net_loss(heatmap_out, regression_out, final_mat_size):
    """
    Define the training loss
    :param heatmap_out: Classification output of net
    :param regression_out: Regression output of net
    :param final_mat_size: Size of net output tensor
    :return: Training operation, few loss, placeholders for training data
    """
    # Define the loss
    heatmap_training_mat = tf.placeholder(tf.float32, (None, final_mat_size[0], final_mat_size[1],
                                                       DATA_CONST.NUM_HEATMAP_CHANNELS))
    heatmap_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(heatmap_out, heatmap_training_mat)

    ignore_region_mask = tf.placeholder(tf.float32, (None, final_mat_size[0], final_mat_size[1]))
    cross_entropy_masked = tf.multiply(heatmap_cross_entropy, ignore_region_mask)
    heatmap_loss = tf.reduce_mean(cross_entropy_masked)

    regression_training_mat = tf.placeholder(tf.float32, (None, final_mat_size[0], final_mat_size[1],
                                                          DATA_CONST.NUM_REGRESSION_CHANNELS))
    regression_loss = tf.nn.l2_loss(regression_out - regression_training_mat)

    bbox_region_mask = tf.placeholder(tf.float32, (None, final_mat_size[0], final_mat_size[1]))
    bbox_loss_masked = tf.multiply(tf.multiply(regression_loss, bbox_region_mask), ignore_region_mask)

    REGRESSION_LOSS_WIEGHT = 1e-2
    regression_loss = REGRESSION_LOSS_WIEGHT * tf.reduce_mean(bbox_loss_masked)

    total_loss = heatmap_loss + regression_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    training_operation = optimizer.minimize(total_loss)

    return training_operation, total_loss, regression_loss, heatmap_loss, \
           heatmap_training_mat, ignore_region_mask, regression_training_mat, bbox_region_mask


def get_training_data(image_batch, final_mat_size, flip_data):
    """
    Format training data to TensorFlow style
    :param image_batch: A list of ImageTrainingData
    :param final_mat_size: Size of net output
    :param flip_data: whether to flip data
    :return:
        X: Image
        Y_heatmap: Training one-hot encoding
        Y_ignore: Ignore mask
        Y_regression: Regression encoded data
        Y_regression_mask: Regression mask
    """
    X = np.concatenate([image_data.get_new_input() for image_data in image_batch])
    Y_heatmap = np.concatenate([image_data.get_classification_output(final_mat_size)
                                for image_data in image_batch])
    Y_ignore = np.concatenate([image_data.get_training_output_mask(final_mat_size)
                               for image_data in image_batch])
    Y_regression = np.concatenate([image_data.get_regression_output(final_mat_size)
                                   for image_data in image_batch])
    Y_regression_mask = np.concatenate([image_data.get_regression_mask(final_mat_size)
                                        for image_data in image_batch])

    # TODO: hacky... Should directly flip kitti_data.ImageTrainingData
    if flip_data:
        X = flip_dim2(X)
        Y_heatmap = flip_dim2(Y_heatmap)
        Y_ignore = flip_dim2(Y_ignore)
        Y_regression = flip_dim2(Y_regression)
        Y_regression_mask = flip_dim2(Y_regression_mask)

    # Label sum to one
    assert np.sum(Y_heatmap[0, :, :, :]) == Y_heatmap.shape[1] * Y_heatmap.shape[2]
    # Only mask regression region
    assert np.sum(np.logical_not(Y_regression_mask) * Y_regression[:, :, :, 0]) == 0
    assert len(X) == len(Y_heatmap)
    assert len(X.shape) == 4
    assert len(Y_heatmap.shape) == 4

    return X, Y_heatmap, Y_ignore, Y_regression, Y_regression_mask


def train():
    """
    Training the net
    """
    TRAINING_DATA_DERECTORY = os.path.join(kitti_data.DATA_DIRECTORY, 'training')
    image_database = kitti_data.KittiImageTrainingDatabase(TRAINING_DATA_DERECTORY)

    # Define the net
    heatmap_out, regression_out, net_input, keep_prob = Net()
    net_out_put_size = (heatmap_out.get_shape()[1], heatmap_out.get_shape()[2])

    # Get the training operation, and placeholders for training data
    training_operation, total_loss, bbox_loss, heatmap_loss, heatmap_training_mat, ignore_region_mask, \
    bbox_training_mat, bbox_region_mask = net_loss(heatmap_out, regression_out, net_out_put_size)

    saver = tf.train.Saver()

    EPOCHS = 100
    # TODO: Change it if you are using TITAN X
    BATCH_SIZE = 1

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        num_examples = len(image_database.training_images)
        training_images = image_database.training_images

        for i in range(EPOCHS):
            # SGD
            random.shuffle(training_images)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                image_batch = training_images[offset:end]

                # I shouldn't put data augmentation here, but it is super easy to implement
                # TODO: This implementation is prone to error. Careful with flipping regression data
                X, Y_heatmap, Y_ignore, Y_regression, Y_regression_mask = \
                    get_training_data(image_batch, net_out_put_size, flip_data=random.choice([True, False]))

                heatmap_loss_val, bbox_loss_val, _ = sess.run([heatmap_loss, bbox_loss, training_operation],
                                                              feed_dict={net_input: X,
                                                                         heatmap_training_mat: Y_heatmap,
                                                                         ignore_region_mask: Y_ignore,
                                                                         bbox_training_mat: Y_regression,
                                                                         bbox_region_mask: Y_regression_mask,
                                                                         keep_prob: 0.5})

                # TODO: Put into a function. A python function should less than 50 lines
                if VERBOSE and offset % 100 == 0:
                    ax1.cla()
                    ax1.set_title('input X')
                    ax2.cla()
                    ax2.set_title('Y_heatmap_1')
                    ax3.cla()
                    ax3.set_title('y_pred_1')
                    ax4.cla()
                    ax4.set_title('y_pred_2')
                    ax5.cla()
                    ax5.set_title('bbox_predict_0')
                    ax6.cla()
                    ax6.set_title('Y_regression_0')
                    ax7.cla()
                    ax7.set_title('Y_regression_1')
                    ax8.cla()
                    ax8.set_title('Y_ignore')
                    ax9.cla()
                    ax9.set_title('Y_regression_mask')

                    ax1.imshow(kitti_data.ImageNormalizer().un_normalize(X[0, :, :, :]).astype('uint8'))
                    ax2.imshow(Y_heatmap[0, :, :, 2])
                    y_pred = heatmap_out.eval(feed_dict={net_input: X[np.newaxis, 0, :, :, :], keep_prob: 1.0})
                    ax3.imshow(y_pred[0, :, :, 1])
                    ax4.imshow(y_pred[0, :, :, 2])
                    bbox_predict = regression_out.eval(feed_dict={net_input: X[np.newaxis, 0, :, :, :], keep_prob: 1.0})
                    ax5.imshow(bbox_predict[0, :, :, 0])
                    ax6.imshow(Y_regression[0, :, :, 0])
                    ax7.imshow(Y_regression[0, :, :, 1])
                    ax8.imshow(Y_ignore[0, :, :])
                    ax9.imshow(Y_regression_mask[0, :, :])
                    # print(Y_mask)
                    f.canvas.draw()
                    f.show()
                    plt.pause(0.01)
                    print('heatmap_loss_val:{}  bbox_loss_val:{}   at batch:{}'
                          .format(heatmap_loss_val, bbox_loss_val, offset))

            print('EPOCH {} ...'.format(i + 1))
            if i % 1 == 0:
                MODEL_SAVE_PATH = './model_{}'.format(i)
                saver.save(sess, MODEL_SAVE_PATH)
                print('Model saved')


if __name__ == '__main__':
    train()
