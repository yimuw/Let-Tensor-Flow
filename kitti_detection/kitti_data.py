"""
Utils to manage Kitti dataset
"""

import pandas
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import collections
import itertools

# TODO: download kitti data set, and change it to './data'
DATA_DIRECTORY = './data_small'
# TODO: kitti image size not consistent?
IMAGE_SIZE = (360, 1220)


class ImageNormalizer:
    """
    Common image normalizer to ensure consistency
    """

    def __init__(self):
        self.MEAN = np.array([100., 100., 100.], dtype='float32')

    def normalize(self, image):
        image_copy = image.copy()
        image_copy -= self.MEAN
        return image_copy

    def un_normalize(self, image):
        image_copy = image.copy()
        image_copy += self.MEAN
        return image_copy


def to_4d(array3d):
    """
    Convert np array from 3d to 4d because tensorflow need 4d input
    :param array3d: input array
    :return: 4d np array
    """
    assert len(array3d.shape) == 3
    return array3d[np.newaxis]


def resize_3d_array(array, new_size):
    """
    Resize a 3d np array
    :param array: np array
    :param new_size: new array size
    :return: a np array of new_size
    """
    assert len(array.shape) == 3
    num_channels = array.shape[2]
    # Ugly code to handle 1d case
    mat_resized = np.zeros(shape=(new_size[0], new_size[1], num_channels))
    # If image shape is (c,r,1), cv2.resize will change it to (c,r)
    for c in range(num_channels):
        # cv2.resize take (cols, rows)
        mat_resized[:, :, c] = cv2.resize(array[:, :, c], (new_size[1], new_size[0]),
                                          interpolation=cv2.INTER_NEAREST)
    return mat_resized


class ImageData:
    """
    Represent a image file
    """

    def __init__(self, image_path):
        self.image_path = image_path

    def get_new_input(self):
        """
        Load image from file, and do preprocessing
        """
        # openCv format: BGR
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        # TODO: why images in kitti dataset have different size????
        # crop to same size
        image = image[:IMAGE_SIZE[0], :IMAGE_SIZE[1], :]
        # Normalize
        image = ImageNormalizer().normalize(image)

        return to_4d(image)


class LabeledImageData(ImageData):
    """
    Represent image training data file
    The ideal is using a database to manage labeled data.
     Benefits:
               1) Easily manage training and testing data.
               2) Handle data imbalance (even best detector has a problem with side of cars).
               3) Show off.
    """

    # Constants
    NUM_HEATMAP_CHANNELS = 3
    BACK_GROUND_IDX = 0
    CAR_CHANNEL_IDX = 1
    PED_CHANNEL_IDX = 2

    NUM_REGRESSION_CHANNELS = 2
    CAR_DISTANCE_IDX = 0
    PED_DISTANCE_IDX = 1

    def __init__(self, image_path, label_file_path):
        super().__init__(image_path)
        self.label_file_path = label_file_path

        # Check kitti data set description
        self.label_fields = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                             'bbox_right', 'bbox_bottom', 'dim_height', 'dim_width', 'dim_length',
                             'location_x', 'location_y', 'location_z', 'rotation_y']

        self.labels = pandas.read_csv(label_file_path, sep=' ', names=self.label_fields)

        assert len(self.labels) > 0

    def get_classification_output(self, y_size=None):
        """
        Generate one-hot training data for classification
        :param y_size: size of output np array
        :return: one-hot encoded np array
        """
        # Y is inited to be background
        Y_heatmap = np.zeros(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.NUM_HEATMAP_CHANNELS))
        Y_heatmap[:, :, self.BACK_GROUND_IDX] = np.ones(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1]))

        def set_heatmap(labels, encoding):
            for label in labels:
                Y_heatmap[int(label.bbox_top): int(label.bbox_bottom),
                int(label.bbox_left): int(label.bbox_right), :] \
                    = encoding

        set_heatmap(labels=self.__get_vehicle_labels('TRAINING'), encoding=np.array([0, 1, 0]))
        set_heatmap(labels=self.__get_pedestrian_labels('TRAINING'), encoding=np.array([0, 0, 1]))

        if y_size is not None:
            Y_heatmap = resize_3d_array(Y_heatmap, y_size)

        return to_4d(Y_heatmap)

    def get_regression_output(self, y_size=None):
        """
        Generate regression training data. you can put ANY number to regression channel as long as it make sense.
        :param y_size: size of output np array
        :return: regression output
        """
        Y_bbox = np.zeros(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.NUM_REGRESSION_CHANNELS))

        def set_distance_channel(labels, channel_idx):
            """
            TODO: Any other channels ?
            """
            # process far label first
            all_labels = sorted(labels,
                                key=lambda l: - np.sqrt(l.location_x ** 2 +
                                                        l.location_y ** 2 + l.location_z ** 2))
            for label in all_labels:
                distance = np.linalg.norm([label.location_x, label.location_y, label.location_z])
                Y_bbox[int(label.bbox_top): int(label.bbox_bottom),
                int(label.bbox_left): int(label.bbox_right), channel_idx] \
                    = distance

        set_distance_channel(labels=list(self.__get_vehicle_labels('TRAINING')), channel_idx=self.CAR_DISTANCE_IDX)
        set_distance_channel(labels=list(self.__get_pedestrian_labels('TRAINING')), channel_idx=self.PED_DISTANCE_IDX)

        if y_size is not None:
            Y_bbox = resize_3d_array(Y_bbox, y_size)

        return to_4d(Y_bbox)

    def get_regression_mask(self, y_size=None):
        """
        Get a mask to mask out not interesting region
        :param y_size: size of mask
        :return: np array of 0. or 1.
        """
        # mask is inited to be background
        mask = np.zeros(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1]))

        for label in itertools.chain(self.__get_vehicle_labels('TRAINING'),
                                     self.__get_pedestrian_labels('TRAINING')):
            mask[int(label.bbox_top): int(label.bbox_bottom),
            int(label.bbox_left): int(label.bbox_right)] = 1.

        if y_size is not None:
            mask = cv2.resize(mask, (y_size[1], y_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Add batch dimension
        return mask[np.newaxis]

    def get_training_output_mask(self, y_size=None):
        """
        Get a mask to mask out don't care region
        :param y_size: size of mask
        :return: np array of 0. or 1.
        """
        # mask is inited to be 1
        mask = np.ones(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1]))

        # TODO: itertools.chain?
        dont_care_labels = self.labels[np.logical_or(np.logical_or(self.labels.type == 'DontCare',
                                                                   self.labels.type == 'Truck'),
                                                     self.labels.type == 'Tram')]
        # Set ignore bbox to 0.
        for index, row in dont_care_labels.iterrows():
            mask[int(row.bbox_top): int(row.bbox_bottom),
            int(row.bbox_left): int(row.bbox_right)] = 0.

        # Set ignore bbox to 0.
        for row in itertools.chain(self.__get_vehicle_labels('DONT_CARE'),
                                   self.__get_pedestrian_labels('DONT_CARE')):
            mask[int(row.bbox_top): int(row.bbox_bottom),
            int(row.bbox_left): int(row.bbox_right)] = 0.

        # Set training  bbox to 1, since ignore cars may overlap with training cars
        for row in itertools.chain(self.__get_vehicle_labels('TRAINING'),
                                   self.__get_pedestrian_labels('TRAINING')):
            mask[int(row.bbox_top): int(row.bbox_bottom),
            int(row.bbox_left): int(row.bbox_right)] = 1.

        if y_size is not None:
            mask = cv2.resize(mask, (y_size[1], y_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Add batch dimension
        return mask[np.newaxis]

    def __get_vehicle_labels(self, output_type):
        """
        Return vehicle bbox
        Copy from kitti description:

                #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                             truncated refers to the object leaving image boundaries
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.
        :param output_type: If is 'DONT_CARE', returns list of don't care vehicle bboxs.
                            If is 'TRAINING', returns list of training vehicle bboxs
        :return: List of bboxs
        """
        # For car and van
        vehicle_labels = self.labels[np.logical_or(self.labels.type == 'Car', self.labels.type == 'Van')]
        # TODO: This control flow is weird...
        for index, row in vehicle_labels.iterrows():
            is_dont_care = False

            # Define don't care here
            if (-row.bbox_top + row.bbox_bottom) < 20:
                is_dont_care = True
            if int(row.occluded) == 2 or int(row.occluded) == 3:
                is_dont_care = True
            if bool(row.truncated) is True:
                is_dont_care = True

            if is_dont_care and output_type == 'DONT_CARE':
                yield row
            elif not is_dont_care and output_type == 'TRAINING':
                yield row

    def __get_pedestrian_labels(self, output_type):
        """
        Get pedestrian bboxs
        :param output_type: If is 'DONT_CARE', returns list of don't care vehicle bboxs.
                            If is 'TRAINING', returns list of training vehicle bboxs
        :return: List of bboxs
        """
        # For car and van
        ped_labels = self.labels[np.logical_or(np.logical_or(self.labels.type == 'Pedestrian',
                                                             self.labels.type == 'Person_sitting'),
                                               self.labels.type == 'Cyclist')]
        # TODO: This control flow is weird...
        for index, row in ped_labels.iterrows():
            is_dont_care = False

            if (-row.bbox_top + row.bbox_bottom) < 20:
                is_dont_care = True
            if int(row.occluded) == 2 or int(row.occluded) == 3:
                is_dont_care = True
            # Lidar should detect
            if bool(row.truncated) is True:
                is_dont_care = True

            if is_dont_care and output_type == 'DONT_CARE':
                yield row
            elif not is_dont_care and output_type == 'TRAINING':
                yield row


class KittiImageTestingDatabase:
    """
    Testing database for kitti (without label!)
    """

    def __init__(self, kitti_training_data_path):
        self.images = []

        for image_path in glob.iglob(os.path.join(kitti_training_data_path, 'image_2', '*.png')):
            self.images.append(ImageData(image_path))


class KittiImageTrainingDatabase:
    """
    Training database for kitti
    I didn't do cross-validation (or testing) for the result because it is not interesting.
    However, if you want to improve something, a metric is a must. A miserable intern can do the job
    """

    def __init__(self, kitti_training_data_path):
        self.training_images = []
        # Assume kitti detection dataset default layout
        for image_path in glob.iglob(os.path.join(kitti_training_data_path, 'image_2', '*.png')):
            label_path = self.__image_path_to_label_path(image_path)
            self.training_images.append(LabeledImageData(image_path, label_path))

    @staticmethod
    def __image_path_to_label_path(image_path):
        """
        internal hack
        """
        image_path_split = image_path.split(os.sep)
        image_path_split[-1] = os.path.splitext(image_path_split[-1])[0] + '.txt'
        image_path_split[-2] = 'label_2'

        return os.path.join(*image_path_split)
