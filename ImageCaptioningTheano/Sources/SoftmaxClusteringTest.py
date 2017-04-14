from LayerProvider import *
from MainLoop import SGDRMainLoop
from NeuralNet import ConvNeuralNet
from Trainer import Trainer
import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import glob
from Utils import LoadList
def loadMNIST(data_path):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(data_path)
    if data_dir == "" and not os.path.isfile(data_path):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            data_path
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    with gzip.open(data_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def CreateOneHot(label_vector, max_val):
    """
    Generate one-hot matrix from an index vector of length N (num sample)
    The one-hot matrix will have the size of N x (max_val+1)

    :type label_vector: 1D numpy vector/python list
    :param label_vector: Index vector, values are ranged from 0 to max_val

    :type max_val: int
    :param max_val: max value of the label vector
    :return:
    """
    one_hot = np.eye(max_val+1, max_val+1, dtype=theano.config.floatX)
    one_hot = one_hot[label_vector]
    expected_output = theano.shared(
        np.reshape(
            one_hot,(
                len(label_vector), max_val+1, 1, 1)
        )
    )
    return expected_output


if __name__ == '__main__':
    trainer = Trainer()

    # Setting training params
    trainer.opts['batch_size'] = 10
    trainer.opts['save'] = True
    trainer.opts['save_freq'] = 10
    trainer.opts['validation'] = False
    # Prepare data & label
    data_path = '../data/mnist.pkl.gz'
    datasets = loadMNIST(data_path)
    train_X, train_Y = datasets[0]
    valid_X, valid_Y = datasets[1]
    test_X, test_Y = datasets[2]

    trainer.opts['num_sample'] = train_X.get_value(borrow=True).shape[0]
    trainer.opts['num_val_sample'] = valid_X.get_value(borrow=True).shape[0]
    num_test = test_X.get_value(borrow=True).shape[0]

    train_X = train_X.reshape((trainer.opts['num_sample'], 1, 28, 28))
    train_X = theano.shared(train_X[0:300,:,:,:].eval())
    trainer.opts['num_sample'] = 300

    train_Y = train_Y[0:trainer.opts['num_sample']]

    valid_X = valid_X.eval()[0:300,:]
    trainer.opts['num_val_sample'] = 300
    valid_X = theano.shared(valid_X.reshape((trainer.opts['num_val_sample'], 1, 28, 28)))
    valid_Y = valid_Y[0:trainer.opts['num_val_sample']]

    test_X = test_X.eval()[0:300,:]
    num_test = 300
    test_X = theano.shared(test_X.reshape((num_test, 1, 28, 28)))
    test_Y = test_Y[0:num_test]

    train_Y = CreateOneHot(train_Y.eval(), 9)
    valid_Y = CreateOneHot(valid_Y.eval(), 9)
    test_Y = CreateOneHot(test_Y.eval(), 9)
    # Create a CNN for debugging by fixing a set of real input
    # net = ConvNeuralNet(train_X[1:16,:,:,:].eval())


    e = -1

    # Create a CNN
    net = ConvNeuralNet()
    net.name = 'mnist_cnn_classification'

    trained_path = '/home/kien/data/trained_model/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')

    if (len(snapshot_list) == 0):

        net.net_opts['l1_learning_rate'] = np.asarray(0.01, dtype=theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.0001, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

        # Construct the CNN
        net.layer_opts['updatable'] = False
        net.layer_opts['filter_shape'] = (8, 1, 3, 3)
        net.content['l1'] = ConvLayer(net, net.content['input'])

        net.layer_opts['filter_shape'] = (16, 8, 5, 5)
        net.content['l2'] = ConvLayer(net, net.content['l1'])

        net.content['l3'] = ELULayer(net, net.content['l2'])

        net.content['l4'] = Pool2DLayer(net, net.content['l3'])

        net.content['DO1'] = DropOut(net, net.content['l4'], 0.4)

        net.layer_opts['filter_shape'] = (32, 16, 5, 5)
        net.content['l5'] = ConvLayer(net, net.content['DO1'])

        net.content['l6'] = ELULayer(net, net.content['l5'])

        net.layer_opts['bn_shape'] = net.GetOutputShape(train_X.eval().shape)
        net.content['BN'] = BatchNormLayer(net, net.content['l6'])

        last_output_shape = net.GetOutputShape(train_X.eval().shape)

        net.layer_opts['num_fc_node'] = 10
        net.content['l7'] = FCLayer(net, net.content['BN'], last_output_shape)

        last_output_shape = net.GetOutputShape(train_X.eval().shape)

        net.layer_opts['bn_shape'] = (trainer.opts['batch_size'], last_output_shape[1],
                                      last_output_shape[2], last_output_shape[3])
        net.content['l8'] = BatchNormLayer(net, net.content['l7'])

        net.layer_opts['softmax_norm_dim'] = 1
        net.content['l9'] = SoftmaxLayer(net, net.content['l8'])

        net.content['cost'] = CategoricalCrossEntropy(net, net.content['l9'])

        # net.content['cost2'] = L2Objective(net, net.content['l10'])

        # net.content['cost'] = AggregateLossLayer([net.content['cost1'] , net.content['cost2'] ])

        net.simpleprint()

        # Initialize learning rate for each updatable layer
        net.InitLR(0.5)

        # Create params list, grad list, momentum list for the theano function to update
        trainer.InitParams(net)

        # Update rule
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['DO1', 'l9']
        net.InitTrainFunction(train_update_rule, train_X, train_Y, additional_output)
        net.InitValFunction(valid_X, valid_Y)
        e = 0
    else:
        snapshot_list = sorted(snapshot_list)
        print('Loading latest snapshot at %s' % snapshot_list[-1])
        net, trainer, e = LoadList(snapshot_list[-1])

        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['DO1', 'l9']
        net.InitTrainFunction(train_update_rule, train_X, train_Y, additional_output)
        net.InitValFunction(valid_X, valid_Y, additional_output)

    main_loop = SGDRMainLoop(net, trained_path)
    main_loop.run(net, trainer, e)

    a = 2
