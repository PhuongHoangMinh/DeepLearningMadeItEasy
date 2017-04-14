from LayerProvider import *
from MainLoop import SGDRMainLoop
from NeuralNet import *
from Trainer import Trainer
import os
import gzip
import six.moves.cPickle as pickle
import glob
import numpy as np
from Utils import LoadList
def npsigmoid(X):
    return 1.0/(1.0 + np.exp(-X))


def CreateOneHot(label_vector, max_val):
    """
    Generate one-hot matrix from an index vector of length N (num sample)
    The one-hot matrix will have the size of N x (max_val+1)
    Modified for sentence input

    :type label_vector: 2D numpy array
    :param label_vector: Index array of shape (batch_size, T), values are ranged from 0 to max_val
            T is sentence length

    :type max_val: int
    :param max_val: max value of the label vector
    :return:
    """
    one_hot = np.eye(max_val+1, max_val+1, dtype=theano.config.floatX)
    one_hot = one_hot[label_vector]
    expected_output = theano.shared(
        np.reshape(
            one_hot,(
                label_vector.shape[0], label_vector.shape[1], max_val+1, 1)
        )
    )
    return expected_output

def CreateData(x_dim, num_class, trainer):
    train_batch_X = np.arange(0, 1, 1 / float(trainer.opts['train_sentence_length'])).reshape(
        (1, trainer.opts['train_sentence_length'], 1, 1))
    valid_batch_X = np.arange(0, 1, 1 / float(trainer.opts['train_sentence_length'])).reshape(
        (1, trainer.opts['train_sentence_length'], 1, 1))
    test_batch_X = np.arange(0, 1, 1 / float(trainer.opts['train_sentence_length'])).reshape(
        (1, trainer.opts['train_sentence_length'], 1, 1))

    noise = 0.01 * np.random.randn(trainer.opts['num_sample'], trainer.opts['train_sentence_length'], x_dim, 1) + 0.1
    train_batch_X = np.repeat(train_batch_X, x_dim, 2)
    train_batch_X = np.repeat(train_batch_X, trainer.opts['num_sample'], 0) + noise

    noise = 0.01 * np.random.randn(trainer.opts['num_val_sample'], trainer.opts['train_sentence_length'], x_dim,
                                   1) + 0.1
    valid_batch_X = np.repeat(valid_batch_X, x_dim, 2)
    valid_batch_X = np.repeat(valid_batch_X, trainer.opts['num_val_sample'], 0) + noise

    noise = 0.01 * np.random.randn(trainer.opts['num_test_sample'], trainer.opts['train_sentence_length'], x_dim,
                                   1) + 0.1
    test_batch_X = np.repeat(test_batch_X, x_dim, 2)
    test_batch_X = np.repeat(test_batch_X, trainer.opts['num_test_sample'], 0) + noise

    label_batch = np.arange(4, 4 + trainer.opts['train_sentence_length'])

    noise = np.random.randint(-2, 2, (trainer.opts['num_sample'], 1))
    noise = np.repeat(noise, trainer.opts['train_sentence_length'], 1)
    train_label_batch = label_batch.reshape(1, trainer.opts['train_sentence_length'])
    train_label_batch = np.repeat(train_label_batch, trainer.opts['num_sample'], 0) + noise

    noise = np.random.randint(-2, 2, (trainer.opts['num_val_sample'], 1))
    noise = np.repeat(noise, trainer.opts['train_sentence_length'], 1)
    valid_label_batch = label_batch.reshape(1, trainer.opts['train_sentence_length'])
    valid_label_batch = np.repeat(valid_label_batch, trainer.opts['num_val_sample'], 0) + noise

    noise = np.random.randint(-2, 2, (trainer.opts['num_test_sample'], 1))
    noise = np.repeat(noise, trainer.opts['train_sentence_length'], 1)
    test_label_batch = label_batch.reshape(1, trainer.opts['train_sentence_length'])
    test_label_batch = np.repeat(test_label_batch, trainer.opts['num_test_sample'], 0) + noise

    train_X = theano.shared(
        np.asarray(

            train_batch_X,
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    valid_X = theano.shared(
        np.asarray(
            valid_batch_X,
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    test_X = theano.shared(
        np.asarray(
            test_batch_X,
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    train_Y = theano.shared(
        np.asarray(
            train_label_batch
        ),
        borrow=True
    )

    valid_Y = theano.shared(
        np.asarray(
            valid_label_batch,
        ),
        borrow=True
    )

    test_Y = theano.shared(
        np.asarray(
            test_label_batch,
        ),
        borrow=True
    )

    train_Y = CreateOneHot(train_Y.eval(), num_class - 1)
    valid_Y = CreateOneHot(valid_Y.eval(), num_class - 1)
    test_Y = CreateOneHot(test_Y.eval(), num_class - 1)
    return (train_X, valid_X, test_X, train_Y, valid_Y, test_Y)

def train():
    # theano.config.optimizer='fast_compile'

    trainer = Trainer()

    # Setting training params
    trainer.opts['batch_size'] = 100
    trainer.opts['save'] = True
    trainer.opts['save_freq'] = 100
    trainer.opts['num_sample'] = 300000
    trainer.opts['num_epoch'] = 5000
    trainer.opts['train_sentence_length'] = 11
    trainer.opts['test_setence_length'] = 15
    trainer.opts['num_val_sample'] = 1
    trainer.opts['num_test_sample'] = 1
    # Generate data
    num_class = 16
    np.random.seed(13111991)

    x_dim = 32

    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = CreateData(x_dim, num_class, trainer)

    # Create a CNN for debugging by fixing a set of real input
    # net = ConvNeuralNet(train_X[1:16,:,:,:].eval())

    # Create a CNN



    net = ShowTellNet()
    net.name = 'lstm_test'
    trained_path = '../../data/trained_model/'
    #trained_path = '/home/kien/data/trained_model/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
    e = -1
    if (len(snapshot_list) == 0):


        net.net_opts['l1_learning_rate'] = np.asarray(0.0001, dtype=theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.00001, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

        net.layer_opts['num_fc_node'] = 32
        net.content['img_emb'] = FCLayer(net, net.content['input_img'], (1,trainer.opts['train_sentence_length'],x_dim,1))
        net.content['img_emb_swap'] = SwapDim(net, net.content['img_emb'], 1, 2)
        # Construct the network

        net.layer_opts['num_emb'] = 32
        net.content['word_emb'] = WordEmbLayer(net, net.content['input_sen'],
                                (trainer.opts['batch_size'], trainer.opts['train_sentence_length']-1,
                                                      num_class, 1))

        net.content['cat'] = Concat(net, net.content['img_emb_swap'], net.content['word_emb'], 1)

        net.layer_opts['num_lstm_node'] = num_class
        net.content['lstm'] = LSTM(net, net.content['cat'],
                                   (trainer.opts['batch_size'], trainer.opts['train_sentence_length']-1,
                                    net.layer_opts['num_emb'], 1))

        net.content['lstm_r'] = LSTMRemove(net, net.content['lstm'], 0)

        #################### DEBUG #######################
        # X = np.reshape(train_X[0:2, :, :, :].eval(), (2, 10, x_dim, 1))
        # Y = np.reshape(train_Y[0:2, :, :, :].eval(), (2, 10, num_class, 1))
        # h_dummy = np.zeros((1, 1, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        # c_dummy = np.zeros((1, 1, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        h_dummy5 = np.zeros((1, 5, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        c_dummy5 = np.zeros((1, 5, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        # cat = net.content['cat'].output.eval({net.input[0]:X , net.input[1]: Y})
        # cat = np.reshape(cat, (2, 11, x_dim))
        # cat0 = np.reshape(cat[1,0,:], (1,1,x_dim))
        # cat1 = np.reshape(cat[1,1,:], (1,1,x_dim))
        # cat2 = np.reshape(cat[1,2,:], (1,1,x_dim))
        #
        # x0 = cat[0,0,:].reshape(1,1,x_dim)
        # x1 = cat[0,1,:].reshape(1,1,x_dim)
        # x2 = cat[0,2,:].reshape(1,1,x_dim)
        # x3 = cat[0,3,:].reshape(1,1,x_dim)
        # Wi = net.content['lstm'].W['i'].eval()
        # Wf = net.content['lstm'].W['f'].eval()
        # Wc = net.content['lstm'].W['c'].eval()
        # Wo = net.content['lstm'].W['o'].eval()
        #
        # Ui = net.content['lstm'].U['i'].eval()
        # Uf = net.content['lstm'].U['f'].eval()
        # Uc = net.content['lstm'].U['c'].eval()
        # Uo = net.content['lstm'].U['o'].eval()
        #
        # bi = net.content['lstm'].b['i'].eval()
        # bf = net.content['lstm'].b['f'].eval()
        # bc = net.content['lstm'].b['c'].eval()
        # bo = net.content['lstm'].b['o'].eval()
        #
        # hm1 = h_dummy
        # cm1 = c_dummy
        #
        # # First iteration
        # i0 = npsigmoid(np.dot(x0, Wi) + np.dot(hm1, Ui) + bi)
        # f0 = npsigmoid(np.dot(x0, Wf) + np.dot(hm1, Uf) + bf)
        # o0 = npsigmoid(np.dot(x0, Wo) + np.dot(hm1, Uo) + bo)
        # c0 = f0*cm1 + i0*np.tanh(np.dot(x0, Wc) + np.dot(hm1, Uc) + bc)
        # h0 = o0*c0
        #
        # # 2nd iteration
        # i1 = npsigmoid(np.dot(x1, Wi) + np.dot(h0, Ui) + bi)
        # f1 = npsigmoid(np.dot(x1, Wf) + np.dot(h0, Uf) + bf)
        # o1 = npsigmoid(np.dot(x1, Wo) + np.dot(h0, Uo) + bo)
        # c1 = f1 * c0 + i1 * np.tanh(np.dot(x1, Wc) + np.dot(h0, Uc) + bc)
        # h1 = o1 * c1
        #
        # # 3rd iteration
        # i2 = npsigmoid(np.dot(x2, Wi) + np.dot(h1, Ui) + bi)
        # f2 = npsigmoid(np.dot(x2, Wf) + np.dot(h1, Uf) + bf)
        # o2 = npsigmoid(np.dot(x2, Wo) + np.dot(h1, Uo) + bo)
        # c2 = f2 * c1 + i2 * np.tanh(np.dot(x2, Wc) + np.dot(h1, Uc) + bc)
        # h2 = o2 * c2
        #
        # # 4th iteration
        # i3 = npsigmoid(np.dot(x3, Wi) + np.dot(h2, Ui) + bi)
        # f3 = npsigmoid(np.dot(x3, Wf) + np.dot(h2, Uf) + bf)
        # o3 = npsigmoid(np.dot(x3, Wo) + np.dot(h2, Uo) + bo)
        # c3 = f3 * c2 + i3 * np.tanh(np.dot(x3, Wc) + np.dot(h2, Uc) + bc)
        # h3 = o3 * c3
        # bp = 1
        #
        #
        # lstm = net.content['lstm'].output.eval({net.input[0]:X, net.input[1]:Y,
        #                                         net.content['lstm'].h_m1_sym: h_dummy2,
        #                                         net.content['lstm'].c_m1_sym: c_dummy2})

        ####################END DEBUG#####################

        net.layer_opts['softmax_norm_dim'] = 2
        net.content['softmax'] = SoftmaxLayer(net, net.content['lstm_r'])



        net.content['cost'] = CategoricalCrossEntropy(net, net.content['softmax'])

        # net.simpleprint()

        net.InitLR(0.01)

        # Create params list, grad list, momentum list for the theano function to update
        trainer.InitParams(net)

        # Update rule
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_img', 'word_emb','softmax']
        # Clip train_Y before

        net.InitTrainFunction(train_update_rule, [train_X, train_Y[:,:-1,:,:]], train_Y[:,1:,:,:], additional_output)
        net.InitValFunction([valid_X, valid_Y[:,:-1,:,:]], valid_Y[:,1:,:,:], additional_output)
    else:
        snapshot_list = sorted(snapshot_list)
        print('Loading latest snapshot at %s' % snapshot_list[-1])
        net, trainer, e = LoadList(snapshot_list[-1])

        # trainer = Trainer()

        # Setting training params
        # trainer.opts['batch_size'] = 100
        # trainer.opts['save'] = True
        # trainer.opts['save_freq'] = 50
        # trainer.opts['num_sample'] = 1000
        # trainer.opts['num_epoch'] = 5000
        # trainer.opts['train_sentence_length'] = 10
        # trainer.opts['test_setence_length'] = 15
        # trainer.opts['num_val_sample'] = 1
        # trainer.opts['num_test_sample'] = 1
        #
        #

        net.net_opts['l1_learning_rate'] = np.asarray(0.0001, dtype=theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.00001, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']
        net.InitLR(100)
        trainer.InitParams(net)
        # Create params list, grad list, momentum list for the theano function to update
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_img', 'word_emb','softmax']


        ###########################
        # net = ShowTellNet()
        # net.name = 'lstm_test'
        #
        # net.net_opts['l1_learning_rate'] = np.asarray(0.0001, dtype=theano.config.floatX)
        # net.reset_opts['min_lr'] = np.asarray(0.00001, dtype=theano.config.floatX)
        # net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']
        #
        # net.layer_opts['num_fc_node'] = 16
        # net.content['img_emb'] = FCLayer(net, net.content['input_img'], (1, 10, x_dim, 1))
        # net.content['img_emb_swap'] = SwapDim(net, net.content['img_emb'], 1, 2)
        # # Construct the network
        #
        # net.layer_opts['num_emb'] = 16
        # net.content['word_emb'] = WordEmbLayer(net, net.content['input_sen'],
        #                                        (trainer.opts['batch_size'], trainer.opts['train_sentence_length'],
        #                                         num_class, 1))
        #
        # net.content['cat'] = Concat(net, net.content['img_emb_swap'], net.content['word_emb'], 1)
        #
        # net.layer_opts['num_lstm_node'] = num_class
        # net.content['lstm'] = LSTM(net, net.content['cat'],
        #                            (trainer.opts['batch_size'], trainer.opts['train_sentence_length'],
        #                             net.layer_opts['num_emb'], 1))
        #
        # net.content['lstm_r'] = LSTMRemove(net, net.content['lstm'], 0, 1)
        #
        # net.layer_opts['softmax_norm_dim'] = 2
        # net.content['softmax'] = SoftmaxLayer(net, net.content['lstm_r'])
        #
        # net.content['cost'] = CategoricalCrossEntropy(net, net.content['softmax'])
        # net.InitLR(100)
        # trainer.InitParams(net)
        # train_update_rule = trainer.InitUpdateRule(net)
        # additional_output = ['input_img', 'word_emb', 'softmax']

        #######################3



        # Create params list, grad list, momentum list for the theano function to update



        # net.train_function = theano.function(
        #     [net.index],
        #     outputs=[net.content['cost'].output] + [net.output[0][net.index, :, :, :]],
        #     updates=None,
        #     givens={
        #         net.input[0]: train_X[net.index, :, :, :],
        #         net.input[1]: train_Y[net.index, :, :, :],
        #         net.output[0]: train_X[net.index, :, :, :],
        #         net.content['lstm'].h_m1_sym: T.zeros((1, net.index.shape[0], net.content['lstm'].W_shape[1]),
        #                                               dtype=theano.config.floatX),
        #         net.content['lstm'].c_m1_sym: T.zeros((1, net.index.shape[0], net.content['lstm'].W_shape[1]),
        #                                               dtype=theano.config.floatX)
        #
        #     }
        #
        # )
        net.InitTrainFunction(train_update_rule, [train_X, train_Y], train_Y, additional_output)
        net.InitValFunction([valid_X, valid_Y], valid_Y, additional_output)

    main_loop = SGDRMainLoop(net, trained_path)
    main_loop.run(net, trainer, e)



    a = 2

def infer():
    trainer = Trainer()

    # Setting training params
    trainer.opts['batch_size'] = 100
    trainer.opts['save'] = True
    trainer.opts['save_freq'] = 5000
    trainer.opts['num_sample'] = 10000
    trainer.opts['num_epoch'] = 5000
    trainer.opts['train_sentence_length'] = 11
    trainer.opts['test_setence_length'] = 15
    trainer.opts['num_val_sample'] = 1
    trainer.opts['num_test_sample'] = 1
    # Generate data
    num_class = 16
    np.random.seed(13111991)

    x_dim = 32

    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = CreateData(x_dim, num_class, trainer)

    net = ShowTellNet()
    net.name = 'lstm_test'
    trained_path = '../../data/trained_model/'
    # trained_path = '/home/kien/data/trained_model/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')

    assert len(snapshot_list) != 0, ('Can\'t find net data at %s' % trained_path)
    snapshot_list = sorted(snapshot_list)
    print('Loading latest snapshot at %s' % snapshot_list[-1])
    net, trainer, e = LoadList(snapshot_list[-1])

    # Creating small batches for debugging
    X = np.reshape(train_X[0, :, :, :].eval(), (1, 11, x_dim, 1))
    Y = np.reshape(train_Y[0, :, :, :].eval(), (1, 11, num_class, 1))

    X_cat = net.content['img_emb_swap'].output.eval({net.input[0]: X})
    beam_search = BeamSearch(net, X_cat, net.content['lstm'], net.content['word_emb'], 11, 8, 4)

    bp = 1


if __name__ == '__main__':
    train()
    # infer()