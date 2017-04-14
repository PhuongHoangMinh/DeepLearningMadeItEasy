from LayerProvider import *
from NeuralNet import *
import numpy as np
from Utils import LoadList
from Trainer import Trainer
from MainLoop import *
import matplotlib.image as mpimg
import glob
import scipy
def npsigmoid(X):
    return 1.0/(1.0 + np.exp(-X))

def LoadVGG(net):
    data_path = '../../data/pretrained/vgg16.npy'
    #data_path = '/home/kien/PycharmProjects/data/vgg16.npy'
    data_dict = np.load(data_path).item()
    net.layer_opts['updatable'] = False
    net.layer_opts['border_mode'] = 1

    W = data_dict['conv1_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv1_1'][1]
    b = b.reshape(1, 64, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv1_1'] = ConvLayer(net, net.content['input_img'])
    net.content['conv1_1'].W.set_value(W)
    net.content['conv1_1'].b.set_value(b)

    net.content['relu1_1'] = ReLULayer(net, net.content['conv1_1'])

    W = data_dict['conv1_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv1_2'][1]
    b = b.reshape(1, 64, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv1_2'] = ConvLayer(net, net.content['relu1_1'])
    net.content['conv1_2'].W.set_value(W)
    net.content['conv1_2'].b.set_value(b)

    net.content['relu1_2'] = ReLULayer(net, net.content['conv1_2'])

    net.layer_opts['pool_mode'] = 'max'
    net.content['pool1'] = Pool2DLayer(net, net.content['relu1_2'])

    W = data_dict['conv2_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv2_1'][1]
    b = b.reshape(1, 128, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv2_1'] = ConvLayer(net, net.content['pool1'])
    net.content['conv2_1'].W.set_value(W)
    net.content['conv2_1'].b.set_value(b)

    net.content['relu2_1'] = ReLULayer(net, net.content['conv2_1'])

    W = data_dict['conv2_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv2_2'][1]
    b = b.reshape(1, 128, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv2_2'] = ConvLayer(net, net.content['relu2_1'])
    net.content['conv2_2'].W.set_value(W)
    net.content['conv2_2'].b.set_value(b)

    net.content['relu2_2'] = ReLULayer(net, net.content['conv2_2'])

    net.content['pool2'] = Pool2DLayer(net, net.content['relu2_2'])

    W = data_dict['conv3_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_1'][1]
    b = b.reshape(1, 256, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_1'] = ConvLayer(net, net.content['pool2'])
    net.content['conv3_1'].W.set_value(W)
    net.content['conv3_1'].b.set_value(b)

    net.content['relu3_1'] = ReLULayer(net, net.content['conv3_1'])

    W = data_dict['conv3_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_2'][1]
    b = b.reshape(1, 256, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_2'] = ConvLayer(net, net.content['relu3_1'])
    net.content['conv3_2'].W.set_value(W)
    net.content['conv3_2'].b.set_value(b)

    net.content['relu3_2'] = ReLULayer(net, net.content['conv3_2'])

    W = data_dict['conv3_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_3'][1]
    b = b.reshape(1, 256, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_3'] = ConvLayer(net, net.content['relu3_2'])
    net.content['conv3_3'].W.set_value(W)
    net.content['conv3_3'].b.set_value(b)

    net.content['relu3_3'] = ReLULayer(net, net.content['conv3_3'])

    net.content['pool3'] = Pool2DLayer(net, net.content['relu3_3'])

    W = data_dict['conv4_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_1'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_1'] = ConvLayer(net, net.content['pool3'])
    net.content['conv4_1'].W.set_value(W)
    net.content['conv4_1'].b.set_value(b)

    net.content['relu4_1'] = ReLULayer(net, net.content['conv4_1'])

    W = data_dict['conv4_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_2'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_2'] = ConvLayer(net, net.content['relu4_1'])
    net.content['conv4_2'].W.set_value(W)
    net.content['conv4_2'].b.set_value(b)

    net.content['relu4_2'] = ReLULayer(net, net.content['conv4_2'])

    W = data_dict['conv4_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_3'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_3'] = ConvLayer(net, net.content['relu4_2'])
    net.content['conv4_3'].W.set_value(W)
    net.content['conv4_3'].b.set_value(b)

    net.content['relu4_3'] = ReLULayer(net, net.content['conv4_3'])

    net.content['pool4'] = Pool2DLayer(net, net.content['relu4_3'])

    W = data_dict['conv5_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_1'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_1'] = ConvLayer(net, net.content['pool4'])
    net.content['conv5_1'].W.set_value(W)
    net.content['conv5_1'].b.set_value(b)

    net.content['relu5_1'] = ReLULayer(net, net.content['conv5_1'])

    W = data_dict['conv5_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_2'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_2'] = ConvLayer(net, net.content['relu5_1'])
    net.content['conv5_2'].W.set_value(W)
    net.content['conv5_2'].b.set_value(b)

    net.content['relu5_2'] = ReLULayer(net, net.content['conv5_2'])

    W = data_dict['conv5_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_3'][1]
    b = b.reshape(1, 512, 1, 1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_3'] = ConvLayer(net, net.content['relu5_2'])
    net.content['conv5_3'].W.set_value(W)
    net.content['conv5_3'].b.set_value(b)

    net.content['relu5_3'] = ReLULayer(net, net.content['conv5_3'])

    net.content['pool5'] = Pool2DLayer(net, net.content['relu5_3'])

    return net

def VGG_preprocess(data):
    VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = np.reshape(VGG_MEAN, (1, 3, 1, 1))
    return np.asarray(data-VGG_MEAN, dtype=theano.config.floatX)

def CreateData(n_word):
    # Load flickr8k data
    data_path = '../../data/Flickr8k_processed/'
    
    #data_path = '/home/kien/PycharmProjects/data/Flickr8k_processed/'
    img_data = LoadList(data_path + 'Flickr8k_imgdata.dat')
    labels = LoadList(data_path + ('Flickr8k_label_%d.dat' % n_word))
    vocab = LoadList(data_path + ('Flickr8k_vocab_%d.dat' % n_word))

    train_X = np.transpose(img_data[0], (0,3,1,2))
    val_X = np.transpose(img_data[1], (0,3,1,2))
    test_X = np.transpose(img_data[2], (0,3,1,2))
   
    # Change RGB to BGR
    train_X = train_X[:, [2, 1, 0], :, :]
    val_X = val_X[:, [2, 1, 0], :, :]
    test_X = test_X[:, [2, 1, 0], :, :]

    train_X = VGG_preprocess(train_X)
    val_X = VGG_preprocess(val_X)
    test_X = VGG_preprocess(test_X)

    del test_X
    del img_data

    # X = np.reshape(train_X[0:2,:,:,:], (2, 3, 128, 128))
    # X = np.reshape(train_X[0:2, :, :, :], (2, 3, 64, 64))

    train_Y = np.asarray(labels[0], dtype=theano.config.floatX)
    train_Y = np.reshape(train_Y, (6000, 40, n_word, 1))
    val_Y = np.asarray(labels[1], dtype=theano.config.floatX)
    val_Y = np.reshape(val_Y, (1000, 40, n_word, 1))
    test_Y = np.asarray(labels[2], dtype=theano.config.floatX)
    test_Y = np.reshape(test_Y, (1000, 40, n_word, 1))
    Y = train_Y[0:2,:,:]
    Y = np.reshape(Y, (2, 40, n_word, 1))
    
    # For debug purpose
    #train_X = train_X[0:32,:,:,:]
    #train_Y = train_Y[0:32,:,:,:]


    words = np.argmax(train_Y[:,1:,:,:], axis=2)
    remove_ind = words == 17
    train_weight = np.ones_like(words, dtype=theano.config.floatX)
    train_weight[remove_ind] = 0
    train_weight = train_weight.reshape((6000, 39, 1, 1))
    train_weight = np.repeat(train_weight, n_word, 2)
    train_weight = theano.shared(train_weight)

    words = np.argmax(val_Y[:,1:,:,:], axis=2)
    remove_ind = words == 17
    val_weight = np.ones_like(words, dtype=theano.config.floatX)
    val_weight[remove_ind] = 0
    val_weight = val_weight.reshape((1000, 39, 1, 1))
    val_weight = np.repeat(val_weight, n_word, 2)
    val_weight = theano.shared(val_weight)

    train_X = theano.shared(train_X)
    train_Y = theano.shared(train_Y)
    val_X = theano.shared(val_X)
    val_Y = theano.shared(val_Y)

    return (train_X, train_Y, train_weight, val_X, val_Y, val_weight)


def train():
    trained_path = '../../data/trained_model/'
    # LSTM params
    n_word = 2000
    max_len = 40

    # Create net
    net = ShowTellNet()
    net.name = 'ShowTellCheck'
    #net.name = 'abc'
    # Find latest snapshot
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')

    if (len(snapshot_list) == 0):
        train_X, train_Y, train_weight, val_X, val_Y, val_weight = CreateData(n_word)
        #train_X = theano.shared(train_X.eval()[0:200,:,:,:])
        #train_Y = theano.shared(train_Y.eval()[0:200,:,:,:])
        # Trainer params
        trainer = Trainer()
        trainer.opts['batch_size'] = 32
        trainer.opts['save'] = False
        trainer.opts['save_freq'] = 20
        trainer.opts['num_sample'] = 200
        trainer.opts['num_val_sample'] = 1000
        trainer.opts['validation'] = False
        trainer.opts['num_epoch'] = 10000
        trainer.opts['dzdw_norm_thres'] = 1
        trainer.opts['dzdb_norm_thres'] = 0.01
        # Load VGG
        net = LoadVGG(net)
        net.layer_opts['updatable'] = True

        # Setting params
        net.net_opts['l1_learning_rate'] = np.asarray(0.005, theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.005, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

        # Construct the network

        net.layer_opts['num_fc_node'] = 512
        # net.layer_opts['num_fc_node'] = 128
        # net.content['fc6'] = FCLayer(net, net.content['pool5'], (1, 512, 2, 2))
        net.content['fc6'] = FCLayer(net, net.content['pool5'], (1, 512, 4, 4))

        net.content['fc6_swap'] = SwapDim(net, net.content['fc6'], 1, 2)

        net.layer_opts['num_emb'] = 512
        # net.layer_opts['num_emb'] = 128
        net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
                                         (trainer.opts['batch_size'], max_len-1, n_word, 1))

        net.content['cat'] = Concat(net, net.content['fc6_swap'], net.content['we'], 1)

        net.layer_opts['num_lstm_node'] = n_word
        net.content['lstm'] = LSTM(net, net.content['cat'],
                                   (trainer.opts['batch_size'], max_len-1, net.layer_opts['num_emb'], 1))

        ################
        # TESTING LSTM #
        ################

        # h_dummy = np.zeros((1, 1, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        # c_dummy = np.zeros((1, 1, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        # h_dummy2 = np.zeros((1, 2, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        # c_dummy2 = np.zeros((1, 2, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        h_dummy5 = np.zeros((1, 5, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        c_dummy5 = np.zeros((1, 5, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)


        # cat = net.content['cat'].output.eval({net.input[0]:X , net.input[1]: Y})
        # cat = np.reshape(cat, (2, 41, 128))
        # cat0 = np.reshape(cat[1,0,:], (1,1,128))
        # cat1 = np.reshape(cat[1,1,:], (1,1,128))
        # cat2 = np.reshape(cat[1,2,:], (1,1,128))
        #
        # x0 = cat[0,0,:].reshape(1,1,128)
        # x1 = cat[0,1,:].reshape(1,1,128)
        # x2 = cat[0,2,:].reshape(1,1,128)

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
        # i2 = npsigmoid(np.dot(x2, Wi) + np.dot(h1, Ui) + bi)
        # f2 = npsigmoid(np.dot(x2, Wf) + np.dot(h1, Uf) + bf)
        # o2 = npsigmoid(np.dot(x2, Wo) + np.dot(h1, Uo) + bo)
        # c2 = f2 * c1 + i2 * np.tanh(np.dot(x2, Wc) + np.dot(h1, Uc) + bc)
        # h3 = o2 * c2
        # bp = 1
        #
        # h1, c1 = onestep(cat0, h_dummy, c_dummy, net.content['lstm'].W['i'], net.content['lstm'].W['f'],
        #                  net.content['lstm'].W['c'], net.content['lstm'].W['o'],
        #                  net.content['lstm'].U['i'], net.content['lstm'].U['f'], net.content['lstm'].U['c'],
        #                  net.content['lstm'].U['o'],
        #                  net.content['lstm'].b['i'], net.content['lstm'].b['f'], net.content['lstm'].b['c'],
        #                  net.content['lstm'].b['o'])
        #
        # h1 = h1.eval()
        # c1 = c1.eval()
        #
        # h2, c2 = onestep(cat1, h1, c1, net.content['lstm'].W['i'], net.content['lstm'].W['f'],
        #                  net.content['lstm'].W['c'], net.content['lstm'].W['o'],
        #                  net.content['lstm'].U['i'], net.content['lstm'].U['f'], net.content['lstm'].U['c'],
        #                  net.content['lstm'].U['o'],
        #                  net.content['lstm'].b['i'], net.content['lstm'].b['f'], net.content['lstm'].b['c'],
        #                  net.content['lstm'].b['o'])
        #
        # h2 = h2.eval()
        # c2 = c2.eval()
        #
        # h3, c3 = onestep(cat2, h2, c2, net.content['lstm'].W['i'], net.content['lstm'].W['f'],
        #                  net.content['lstm'].W['c'], net.content['lstm'].W['o'],
        #                  net.content['lstm'].U['i'], net.content['lstm'].U['f'], net.content['lstm'].U['c'],
        #                  net.content['lstm'].U['o'],
        #                  net.content['lstm'].b['i'], net.content['lstm'].b['f'], net.content['lstm'].b['c'],
        #                  net.content['lstm'].b['o'])
        #
        # h3 = h3.eval()
        # c3 = c3.eval()
        #
        # lstm = net.content['lstm'].output.eval({net.input[0]:X, net.input[1]:Y,
        #                                         net.content['lstm'].h_m1_sym: h_dummy2,
        #                                         net.content['lstm'].c_m1_sym: c_dummy2})

        # Remove the first 'word' because it was just image priorcat knowledge, has nothing to do with the actual sentence
        net.content['lstm_r'] = LSTMRemove(net, net.content['lstm'], 1)
        #a = net.content['lstm_r'].output.eval({net.input[1]: train_Y[0:5,0:-1,:,:].eval(),
        #    net.input[0]: train_X[0:5,:,:,:].eval(),
        #        net.content['lstm'].h_m1_sym: h_dummy5,
        #        net.content['lstm'].c_m1_sym: c_dummy5
        #        })
        #print('lstm_r shape:')
        #print(a.shape)
        net.layer_opts['softmax_norm_dim'] = 2
        net.content['softmax'] = SoftmaxLayer(net, net.content['lstm_r'])

        net.content['cost'] = CategoricalCrossEntropy(net, net.content['softmax'])

        net.InitLR(0.2) 
        trainer.InitParams(net)
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_sen', 'lstm_r', 'softmax']
        net.InitTrainFunction(train_update_rule, [train_X, train_Y[:,:-1,:,:]], train_Y[:,1:,:,:], additional_output, train_weight)
        net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], additional_output, val_weight)
        e = 0
    else:
        snapshot_list = sorted(snapshot_list)
        print('Loading latest snapshot at %s' % snapshot_list[-1])
        net, trainer, e = LoadList(snapshot_list[-1])
        trainer.opts['save_freq'] = 10
        print('Finished loading snapshot') 
        
        train_X, train_Y, train_weight, val_X, val_Y, val_weight = CreateData(n_word)
        net.net_opts['l1_learning_rate'] = np.asarray(0.00008, theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.00008, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']
        net.InitLR(1000)
        trainer.InitParams(net)
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_sen', 'lstm_r', 'softmax']
        
        net.InitTrainFunction(train_update_rule, [train_X, train_Y[:,:-1,:,:]], train_Y[:,1:,:,:], additional_output, train_weight)
        net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], additional_output, val_weight)

    main_loop = SGDRMainLoop(net, trained_path)
    main_loop.run(net, trainer, e)

def Infer(img_path):
    data_path = '../../data/Flickr8k_processed/'
    trained_path = '../../data/trained_model/'
    n_word = 2000
    max_len = 40
    vocab = LoadList(data_path + ('Flickr8k_vocab_%d.dat' % n_word))
    vocab = np.asarray(vocab)
    start_token = 0
    stop_token = 16
    w = 128
    h = 128
    I = mpimg.imread(img_path)
    I = scipy.misc.imresize(I, [h, w], 'bicubic')
    I = np.transpose(I, (2, 0, 1))
    I = np.reshape(I, (1, 3, h, w))
    I = VGG_preprocess(I)
    I = I[:,[2,1,0],:,:]

    net = ShowTellNet()
    net.name = 'ShowTell'
    # trained_path = '/home/kien/data/trained_model/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')

    assert len(snapshot_list) != 0, ('Can\'t find net data at %s' % trained_path)
    snapshot_list = sorted(snapshot_list)
    print('Loading latest snapshot at %s' % snapshot_list[-1])
    net, trainer, e = LoadList(snapshot_list[-1])

    fc6_out = net.content['fc6_swap'].output.eval({net.input[0]: I})
    beam_search = BeamSearch(net, fc6_out, net.content['lstm'], net.content['we'], max_len, 20,
            start_token, stop_token)
    print(beam_search.output)
    print(vocab[0,beam_search.output])

if __name__=='__main__':

    train()
    #Infer('../../data/Flickr8k_Dataset/Flicker8k_Dataset/3526431764_056d2c61dc.jpg')
    #Infer('../../data/random_test_data/MSCOCO_5.jpg')#Flickr8k_Dataset/Flicker8k_Dataset/MSCOCO_2.jpg.jpg
    #Infer('../../data/random_test_data/MSCOCO_6.jpg')
    bp = 1
