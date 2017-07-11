"""
Visualize network result
"""

import os

import kitti_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Convert longitudinal pixel coordinate to meters. It is a random guess :)
PIXEL_TO_METER = 0.2
DATA_CONST = kitti_data.LabeledImageData


def plot_mats(X, heatmap_value, regression_value):
    """
    Plot network output
    """
    predict_labels = np.argmax(heatmap_value[0, ...], axis=2)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(kitti_data.ImageNormalizer().un_normalize(X[0, :, :, :]).astype('uint8'))
    ax1.set_title('input image')
    ax2.imshow(predict_labels == DATA_CONST.CAR_CHANNEL_IDX)
    ax2.set_title('car pixel')
    ax3.imshow(predict_labels == DATA_CONST.PED_CHANNEL_IDX)
    ax3.set_title('pedestrain pixel')
    ax4.imshow(heatmap_value[0, :, :, DATA_CONST.CAR_CHANNEL_IDX])
    ax4.set_title('car heatmap')
    # TODO: regression channel is suspicious
    ax5.imshow(regression_value[0, :, :, DATA_CONST.CAR_DISTANCE_IDX])
    ax5.set_title('car distance')
    ax6.imshow(regression_value[0, :, :, DATA_CONST.PED_DISTANCE_IDX])
    ax6.set_title('pedestrian distance')

    f.canvas.draw()


def decode_regression_result(target_classification, target_distances):
    """
    Output longitudinal and lateral coordinate of targets if a pixel location if classified as true
    :param target_classification: Bool mat
    :param target_distances: Distance mat
    :return: List of target longitudinal coordinates, list of target lateral coordinates
    """
    target_indice = [car_index for car_index, is_car in np.ndenumerate(target_classification) if is_car]
    target_locations = [(x, target_distances[y, x]) for y, x in target_indice]

    target_longitudinal = [PIXEL_TO_METER * pixel_idx for pixel_idx, dist in target_locations]
    target_lateral = [dist for pixel_idx, dist in target_locations]

    return target_longitudinal, target_lateral


def bird_eye_plot(heatmap_value, regression_value):
    """
    Plot detections in bird-eye view
    :param heatmap_value: Net heatmap output
    :param regression_value: Net regression output
    """
    predict_labels = np.argmax(heatmap_value[0, ...], axis=2)

    car_longitudinal, car_lateral = decode_regression_result(predict_labels == DATA_CONST.CAR_CHANNEL_IDX,
                                                             regression_value[0, :, :, DATA_CONST.CAR_DISTANCE_IDX])

    ped_longitudinal, ped_lateral = decode_regression_result(predict_labels == DATA_CONST.PED_CHANNEL_IDX,
                                                             regression_value[0, :, :, DATA_CONST.PED_DISTANCE_IDX])

    plt.figure()
    plt.scatter(car_longitudinal, car_lateral)
    plt.scatter(ped_longitudinal, ped_lateral, color='r')

    MAX_PIXEL_IDX = 76
    MAX_RANGE = 40
    axes = plt.gca()
    axes.set_xlim([0, MAX_PIXEL_IDX * PIXEL_TO_METER])
    axes.set_ylim([0, MAX_RANGE])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_title('Bird eye objects location\nBlue: cars   Red: pedestrian')
    plt.xlabel('lateral distance')
    plt.ylabel('longitudinal distance')


def check_net(net_path):
    """
    Visualize network result
    :param net_path: path to tensorflow meta data
    """
    TEST_DATA_DIRECTORY = os.path.join(kitti_data.DATA_DIRECTORY, 'testing')
    image_database = kitti_data.KittiImageTestingDatabase(TEST_DATA_DIRECTORY)
    images = image_database.images

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(net_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(TESTING_NET_CHECKPOINT_DIR))

        graph = tf.get_default_graph()
        # net_input and keep_prob are placeholders which defined in train_net.py
        # TO be honest, I don't know what :0 means...
        net_input = graph.get_tensor_by_name("net_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        for image in images:
            X = image.get_new_input()

            feed_dict = {net_input: X, keep_prob: 1.0}

            # y_heatmap and y_regression are network outputs which defined in train_net.py
            y_heatmap = graph.get_tensor_by_name("heatmap_output:0")
            y_regression = graph.get_tensor_by_name("regression_output:0")

            heatmap_value, regression_value = sess.run([y_heatmap, y_regression], feed_dict)

            plot_mats(X, heatmap_value, regression_value)
            bird_eye_plot(heatmap_value, regression_value)

            plt.show()


if __name__ == '__main__':
    # Just hard coding to make this program push bottom
    TESTING_NET_CHECKPOINT_DIR = 'trained_net'
    TESTING_NET_PATH = 'trained_net/model_99.meta'
    check_net(TESTING_NET_PATH)
