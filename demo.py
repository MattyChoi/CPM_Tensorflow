import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import os
# import imageio
import tensorflow as tf
from models.nets import cpm_body
from config import FLAGS

"""Parameters
"""
flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           # default_value='test_imgs/roger.png',
                           default_value=r'C:\Users\matth\OneDrive\Documents\Storage\CSCI_5561\Final_Project\cpm-tf\test_imgs\test_0000035.jpg',
                           # default_value='SINGLE',
                           docstring='path to image')
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/weights/cpm_body.pkl',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=256,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=32,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('joints',
                            default_value=17,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=3,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_bool('KALMAN_ON',
                         default_value=False,
                         docstring='enalbe kalman filter')
tf.app.flags.DEFINE_integer('kalman_noise',
                            default_value=3e-2,
                            docstring='Kalman filter noise value')
tf.app.flags.DEFINE_string('color_channel',
                           default_value='RGB',
                           docstring='')

limbs = [[0, 1],
         [2, 3],
         [3, 4],
         [5, 6],
         [6, 7],
         [8, 9],
         [9, 10],
         [11, 12],
         [12, 13]]

PYTHON_VERSION = 3


def mgray(test_img_resize, test_img):
    test_img_resize = np.dot(test_img_resize[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (FLAGS.input_size, FLAGS.input_size, 1))
    cv2.imshow('color', test_img.astype(np.uint8))
    cv2.imshow('gray', test_img_resize.astype(np.uint8))
    cv2.waitKey(1)
    return test_img_resize


def main(argv):
    tf_device = '/gpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                        name='input_image')

        model = cpm_body.CPM_Model(flags.stages, flags.joints + 1, 5)
        model.build_model()

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    
    model_path_suffix = os.path.join('cpm_body',
                                     'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                     'joints_{}'.format(FLAGS.num_of_joints),
                                     'stages_{}'.format(FLAGS.cpm_stages),
                                     'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                      FLAGS.lr_decay_step)
                                     )
    model_save_dir = os.path.join('models',
                                  'weights',
                                  model_path_suffix)
    saver.restore(sess, model_save_dir + '\\' + FLAGS.network_def.split('.py')[0] + '-10000')

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))

    # Create kalman filters
    kalman_filter_array = None

    # iamge processing
    with tf.device(tf_device):
        while True:
            test_img_t = time.time()
            test_img = cpm_utils.read_image(flags.DEMO_TYPE, [], FLAGS.input_size, 'IMAGE')
            test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
            # test_img_resize = tf.image.resize_images(test_img, [FLAGS.input_size, FLAGS.input_size])
            print('img read time %f' % (time.time() - test_img_t))

            test_img_input = test_img_resize  - 128.0
            test_img_input = np.expand_dims(test_img_input, axis=0)

            # Inference
            fps_t = time.time()
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                            model.stage_heatmap, ],
                                                            feed_dict={'input_placeholder:0': test_img_input})

            print('fps: %.2f' % (1 / (time.time() - fps_t)))
            # Show visualized image
            demo_img = visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
            cv2.imshow('demo_img', demo_img.astype(np.uint8))
            if cv2.waitKey(0) == ord('q'): break
            print('fps: %.2f' % (1 / (time.time() - fps_t)))



def visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array):
    hm_t = time.time()

    last_heatmap = stage_heatmap_np[-1][0, :, :, 0:flags.joints].reshape(
        (flags.hmap_size, flags.hmap_size, flags.joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('hm resize time %f' % (time.time() - hm_t))

    joint_t = time.time()
    joint_coord_set = np.zeros((flags.joints, 2))
    
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    for joint_num in range(flags.joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                        (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num % 6 ]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=5, color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num % 6]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=5, color=joint_color, thickness=-1)
        print(joint_coord, joint_num)

    print('plot joint time %f' % (time.time() - joint_t))

    return test_img


if __name__ == '__main__':
    tf.app.run()
