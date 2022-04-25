import cv2
import cpm_utils
import numpy as np
import math
import tensorflow as tf
import time
import random
import os


tfr_file = 'omc_dataset_train.tfrecords'
path = r'C:\Users\matth\OneDrive\Documents\Storage\CSCI_5561\cpm-tf\utils'
dataset_dir = os.path.join(path, 'dataset')

SHOW_INFO = False
box_size = 32
num_of_joints = 17
gaussian_radius = 2


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int32_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create writer
tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0
t1 = time.time()
# Loop each dir
# for person_dir in os.listdir(dataset_dir):
person_dir = 'train'
# if not os.path.isdir(dataset_dir) or person_dir=='test': continue

gt_file = os.path.join(dataset_dir, person_dir, 'labels.txt')
gt_content = open(gt_file, 'r').readlines()

for idx, line in enumerate(gt_content):
    line = line.split()
    # Check if it is a valid img file
    if not line[0].endswith(('jpg', 'png')):
        continue
    cur_img_path = os.path.join(dataset_dir, person_dir, line[0])
    cur_img = cv2.imread(cur_img_path)

    # Read in bbox and joints coords
    tmp = [float(x) for x in line[1:5]]
    cur_hand_bbox = [min([tmp[0], tmp[2]]),
                        min([tmp[1], tmp[3]]),
                        max([tmp[0], tmp[2]]),
                        max([tmp[1], tmp[3]])
                        ]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]

    cur_hand_joints_x = [float(i) for i in line[5:5+num_of_joints*2:2]]
    cur_hand_joints_y = [float(i) for i in line[6:6+num_of_joints*2:2]]

    # Crop image and adjust joint coords
    # cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),
    #             int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])),
    #             :]
    # cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
    # cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]

    # output_image = np.ones(shape=(box_size, box_size, 3)) * 128
    # output_heatmaps = np.zeros((box_size, box_size, num_of_joints))

    # # Resize and pad image to fit output image size
    # if cur_img.shape[0] > cur_img.shape[1]:
    #     scale = box_size / (cur_img.shape[0] * 1.0)

    #     # Relocalize points
    #     cur_hand_joints_x = map(lambda x: x * scale, cur_hand_joints_x)
    #     cur_hand_joints_y = map(lambda x: x * scale, cur_hand_joints_y)

    #     # Resize image 
    #     image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    #     offset = image.shape[1] % 2

    #     output_image[:, int(box_size / 2 - math.floor(image.shape[1] / 2)): int(
    #         box_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
    #     cur_hand_joints_x = map(lambda x: x + (box_size / 2 - math.floor(image.shape[1] / 2)),
    #                             cur_hand_joints_x)

    #     cur_hand_joints_x = np.asarray(list(cur_hand_joints_x))
    #     cur_hand_joints_y = np.asarray(list(cur_hand_joints_y))


    # else:
    #     scale = box_size / (cur_img.shape[1] * 1.0)

    #     # Relocalize points
    #     cur_hand_joints_x = map(lambda x: x * scale, cur_hand_joints_x)
    #     cur_hand_joints_y = map(lambda x: x * scale, cur_hand_joints_y)

    #     # Resize image
    #     image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    #     offset = image.shape[0] % 2

    #     output_image[int(box_size / 2 - math.floor(image.shape[0] / 2)): int(
    #         box_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
    #     cur_hand_joints_y = map(lambda x: x + (box_size / 2 - math.floor(image.shape[0] / 2)),
    #                             cur_hand_joints_y)

    #     cur_hand_joints_x = np.asarray(list(cur_hand_joints_x))
    #     cur_hand_joints_y = np.asarray(list(cur_hand_joints_y))

    # # Create background map
    # output_background_map = np.ones((box_size, box_size)) - np.amax(output_heatmaps, axis=2)
    coords_set = np.concatenate((np.reshape(cur_hand_joints_x, (num_of_joints, 1)),
                                    np.reshape(cur_hand_joints_y, (num_of_joints, 1))),
                                axis=1)
    output_image = cv2.resize(cur_img, (256, 256))

    output_image_raw = output_image.astype(np.uint8).tostring()
    output_coords_raw = coords_set.flatten().tolist()

    raw_sample = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(output_image_raw),
        'joint': _float32_feature(output_coords_raw)
    }))

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1
    if img_count % 50 == 0:
        print('Processed %d images, took %f seconds' % (img_count, time.time() - t1))
        t1 = time.time()

tfr_writer.close()
