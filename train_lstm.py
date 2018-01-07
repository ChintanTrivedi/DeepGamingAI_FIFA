import numpy as np
import tensorflow as tf
import tflearn
import os

steps_of_history = 10


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


def reshape_for_lstm(data):
    trainX = []
    trainY_movement = []
    trainY_action = []

    for i in range(0, len(data) - steps_of_history):
        window = data[i:i + steps_of_history]

        sampleX = []
        for row in window:
            sampleX.append(row[0])
        sampleY_movement = np.array(window[-1][1]).reshape(-1)
        sampleY_action = np.array(window[-1][2]).reshape(-1)

        trainX.append(np.array(sampleX).reshape(steps_of_history, -1))
        trainY_movement.append(sampleY_movement)
        trainY_action.append(sampleY_action)

    print(np.array(trainX).shape)
    print(np.array(trainY_movement).shape)
    print(np.array(trainY_action).shape)

    return trainX, list(trainY_movement), list(trainY_action)


def get_list():
    list = []
    n_samples = 10000
    for i in range(0, n_samples):
        feature_vector = np.random.rand(128, 1)
        output_movement = np.zeros((5, 1))
        output_movement[np.random.randint(0, 4), 0] = 1
        output_action = np.zeros((5, 1))
        output_action[np.random.randint(0, 4), 0] = 1
        list.append([feature_vector, output_movement, output_action])
    return list


def main():
    # data = get_list()
    filename = 'rnn/training_data1511723591.npy'
    data = list(np.load(filename))
    print(np.shape(data))

    train = 1
    test = 0

    if train == 1:
        # prepare training data
        trainX, trainY_movement, trainY_action = reshape_for_lstm(data)

        with tf.Graph().as_default():
            model_movement = get_model_movement()
            model_movement.fit(trainX, trainY_movement, n_epoch=500, validation_set=0.1)
            model_movement.save('fifa_models/model_movement')

        with tf.Graph().as_default():
            model_action = get_model_action()
            model_action.fit(trainX, trainY_action, n_epoch=500, validation_set=0.1)
            model_action.save('fifa_models/model_action')

    if test == 1:
        trainX, _, _ = reshape_for_lstm(data)

        g1 = tf.Graph()
        g2 = tf.Graph()

        with g1.as_default():
            model_movement = get_model_movement()
            model_movement.load('./fifa_models/model_movement')

        with g2.as_default():
            model_action = get_model_action()
            model_action.load('./fifa_models/model_action')

        with g1.as_default():
            Y_movement = model_movement.predict(trainX)
            print('prediciton 1')
            print(np.shape(Y_movement))
            print(Y_movement[100])

        with g2.as_default():
            Y_action = model_action.predict(trainX)
            print('prediciton 2')
            print(np.shape(Y_action))
            print(Y_action[100])


def main_all():
    training_all = np.zeros(shape=(0, 3))
    for filename in os.listdir('rnn'):
        filename = 'rnn/' + filename
        d = np.load(filename)
        training_all = np.concatenate((training_all, d))

    data = list(training_all)
    trainX, trainY_movement, trainY_action = reshape_for_lstm(data)
    with tf.Graph().as_default():
        model_movement = get_model_movement()
        # model_movement.load('./fifa_models2/model_movement')
        model_movement.fit(trainX, trainY_movement, n_epoch=400, validation_set=0.1)
        model_movement.save('fifa_models2/model_movement')

    with tf.Graph().as_default():
        model_action = get_model_action()
        # model_action.load('./fifa_models2/model_action')
        model_action.fit(trainX, trainY_action, n_epoch=300, validation_set=0.1)
        model_action.save('fifa_models2/model_action')

    return


main_all()
