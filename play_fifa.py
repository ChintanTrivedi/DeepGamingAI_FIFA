# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[70]:


import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from direct_keys import *
from display_controller import get_controller_image
from get_keys import key_check
from grab_screen import grab_screen
from utils import label_map_util
from utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# What model to download.
MODEL_NAME = 'fifa_graph2'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 3


# ## Load a (frozen) Tensorflow model into memory.

def get_model_movement():
    # Network building
    net = tflearn.input_data(shape=[None, 10, 128], name='net1_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net1_layer2')
    net = tflearn.dropout(net, 0.6, name='net1_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net1_layer4')
    net = tflearn.dropout(net, 0.6, name='net1_layer5')
    net = tflearn.fully_connected(net, 5, activation='softmax', name='net1_layer6')
    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             name='net1_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)


def get_model_action():
    # Network building
    net = tflearn.input_data(shape=[None, 10, 128], name='net2_layer1')
    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net2_layer2')
    net = tflearn.dropout(net, 0.6, name='net2_layer3')
    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net2_layer4')
    net = tflearn.dropout(net, 0.6, name='net2_layer5')
    net = tflearn.fully_connected(net, 5, activation='softmax', name='net2_layer6')
    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             name='net2_layer7')
    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)


def take_action(movement_index, action_index):
    # movement = [[uparrow], [downarrow], [leftarrow], [rightarrow], []]
    movement_custom_b = [[U, E], [J, E], [H, E], [L, E], []]
    action = [[spacebar], [W], [Q], [F], []]

    # print('movement: ' + str(movement_index) + ' and action: ' + str(action_index))

    for index in movement_custom_b[movement_index]:
        PressKey(index)
    for index in action[action_index]:
        PressKey(index)
    time.sleep(0.2)
    for index in movement_custom_b[movement_index]:
        ReleaseKey(index)
    for index in action[action_index]:
        ReleaseKey(index)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    model_movement = get_model_movement()
    model_movement.load('./fifa_models/model_movement')

with g2.as_default():
    model_action = get_model_action()
    model_action.load('./fifa_models/model_action')

steps_of_history = 10
input_window = np.zeros(shape=(steps_of_history, 128))

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        feature_vector = detection_graph.get_tensor_by_name(
            "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")
        for i in range(0, steps_of_history):
            screen = grab_screen(region=None)
            screen = screen[20:1000, :1910]
            image_np = cv2.resize(screen, (900, 400))
            image_np_expanded = np.expand_dims(image_np, axis=0)

            rep = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
            input_window[i, :] = np.array(rep).reshape(-1, 128)

print('starting to play...')

visualise = 1
play = 1

last_time = time.time()
frames_count = 0

with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    feature_vector = detection_graph.get_tensor_by_name(
        "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")
    paused = True
    while True:
        if not paused:
            screen = grab_screen(region=None)
            screen = screen[20:1000, :1910]
            image_np = cv2.resize(screen, (900, 400))
            image_np_expanded = np.expand_dims(image_np, axis=0)

            if visualise == 1:
                with detection_graph.as_default():
                    (boxes, scores, classes, num, rep) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections, feature_vector],
                        feed_dict={image_tensor: image_np_expanded})
                    input_window[:-1, :] = input_window[1:, :]
                    input_window[-1, :] = np.array(rep).reshape(-1, 128)

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=3)

                    cv2.imshow('game', image_np)
                    if cv2.waitKey(25) & 0xff == ord('o'):
                        cv2.destroyAllWindows()
                        break
            else:
                with detection_graph.as_default():
                    (rep) = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
                    input_window[:-1, :] = input_window[1:, :]
                    input_window[-1, :] = np.array(rep).reshape(-1, 128)

            with g1.as_default():
                Y_movement = model_movement.predict(input_window.reshape(-1, 10, 128))
                movement_index = np.argmax(Y_movement)

            with g2.as_default():
                Y_action = model_action.predict(input_window.reshape(-1, 10, 128))
                action_index = np.argmax(Y_action)

            if play == 1:
                take_action(movement_index, action_index)

            if visualise == 1:
                image_controller = get_controller_image(movement_index, action_index)
                # image_controller = get_controller_image(np.random.randint(0, 5), np.random.randint(0, 5))
                cv2.imshow('controller', image_controller)
                if cv2.waitKey(25) & 0xff == ord('i'):
                    cv2.destroyAllWindows()
                    break

            current_time = time.time()
            if current_time - last_time >= 1:
                print('{} frames per second'.format(frames_count))
                last_time = current_time
                frames_count = 0
            else:
                frames_count = frames_count + 1

        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                cv2.destroyAllWindows()
                time.sleep(1)
        elif 'O' in keys:
            print('Quitting!')
            cv2.destroyAllWindows()
            break
