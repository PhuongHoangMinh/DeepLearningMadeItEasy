from LayerProvider import *
from NeuralNet import *
import numpy as np
import matplotlib.image as mpimg
import scipy.misc
from Utils import LoadList
from Trainer import Trainer
from MainLoop import *
import glob
import scipy
from coco_utils import *
from PrepareCOCOData import VGG_preprocess
import pdb

def LoadVGG():
    data_path = '../../data/pretrained/vgg16.npy'
    #data_path = '/home/kien/PycharmProjects/data/vgg16.npy'
    data_dict = np.load(data_path).item()

    net = ShowTellNet()
    
    net.net_opts['rng_seed'] = 123
    net.net_opts['rng'] = np.random.RandomState(net.net_opts['rng_seed'])
    
    net.layer_opts['updatable'] = False
    net.layer_opts['border_mode'] = 1
    
    W = data_dict['conv1_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv1_1'][1]
    b = b.reshape(1,64,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv1_1'] = ConvLayer(net, net.content['input_img'])
    net.content['conv1_1'].W.set_value(W)
    net.content['conv1_1'].b.set_value(b)
    
    net.content['relu1_1'] = ReLULayer(net, net.content['conv1_1'])
    
    W = data_dict['conv1_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv1_2'][1]
    b = b.reshape(1,64,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv1_2'] = ConvLayer(net, net.content['relu1_1'])
    net.content['conv1_2'].W .set_value(W)
    net.content['conv1_2'].b.set_value(b)
    
    net.content['relu1_2'] = ReLULayer(net, net.content['conv1_2'])
    
    net.layer_opts['pool_mode'] = 'max'
    net.content['pool1'] = Pool2DLayer(net, net.content['relu1_2'])
    
    W = data_dict['conv2_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv2_1'][1]
    b = b.reshape(1,128,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv2_1'] = ConvLayer(net, net.content['pool1'])
    net.content['conv2_1'].W.set_value(W)
    net.content['conv2_1'].b.set_value(b)
    
    net.content['relu2_1'] = ReLULayer(net, net.content['conv2_1'])
    
    W = data_dict['conv2_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv2_2'][1]
    b = b.reshape(1,128,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv2_2'] = ConvLayer(net, net.content['relu2_1'])
    net.content['conv2_2'].W.set_value(W)
    net.content['conv2_2'].b.set_value(b)
    
    net.content['relu2_2'] = ReLULayer(net, net.content['conv2_2'])
    
    net.content['pool2'] = Pool2DLayer(net, net.content['relu2_2'])
    
    W = data_dict['conv3_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_1'][1]
    b = b.reshape(1,256,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_1'] = ConvLayer(net, net.content['pool2'])
    net.content['conv3_1'].W.set_value(W)
    net.content['conv3_1'].b.set_value(b)
    
    net.content['relu3_1'] = ReLULayer(net, net.content['conv3_1'])
    
    W = data_dict['conv3_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_2'][1]
    b = b.reshape(1,256,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_2'] = ConvLayer(net, net.content['relu3_1'])
    net.content['conv3_2'].W.set_value(W)
    net.content['conv3_2'].b.set_value(b)
    
    net.content['relu3_2'] = ReLULayer(net, net.content['conv3_2'])
    
    W = data_dict['conv3_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv3_3'][1]
    b = b.reshape(1,256,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv3_3'] = ConvLayer(net, net.content['relu3_2'])
    net.content['conv3_3'].W.set_value(W)
    net.content['conv3_3'].b.set_value(b)
    
    net.content['relu3_3'] = ReLULayer(net, net.content['conv3_3'])
    
    net.content['pool3'] = Pool2DLayer(net, net.content['relu3_3'])
    
    W = data_dict['conv4_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_1'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_1'] = ConvLayer(net, net.content['pool3'])
    net.content['conv4_1'].W.set_value(W)
    net.content['conv4_1'].b.set_value(b)
    
    net.content['relu4_1'] = ReLULayer(net, net.content['conv4_1'])
    
    W = data_dict['conv4_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_2'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_2'] = ConvLayer(net, net.content['relu4_1'])
    net.content['conv4_2'].W.set_value(W)
    net.content['conv4_2'].b.set_value(b)
    
    net.content['relu4_2'] = ReLULayer(net, net.content['conv4_2'])
    
    W = data_dict['conv4_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv4_3'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv4_3'] = ConvLayer(net, net.content['relu4_2'])
    net.content['conv4_3'].W .set_value(W)
    net.content['conv4_3'].b.set_value(b)
    
    net.content['relu4_3'] = ReLULayer(net, net.content['conv4_3'])
    
    net.content['pool4'] = Pool2DLayer(net, net.content['relu4_3'])
    
    W = data_dict['conv5_1'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_1'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_1'] = ConvLayer(net, net.content['pool4'])
    net.content['conv5_1'].W .set_value(W)
    net.content['conv5_1'].b.set_value(b)
    
    net.content['relu5_1'] = ReLULayer(net, net.content['conv5_1'])
    
    W = data_dict['conv5_2'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_2'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_2'] = ConvLayer(net, net.content['relu5_1'])
    net.content['conv5_2'].W .set_value(W)
    net.content['conv5_2'].b.set_value(b)
    
    net.content['relu5_2'] = ReLULayer(net, net.content['conv5_2'])
    
    W = data_dict['conv5_3'][0]
    W = np.transpose(W, (3, 2, 0, 1))
    b = data_dict['conv5_3'][1]
    b = b.reshape(1,512,1,1)
    net.layer_opts['filter_shape'] = W.shape
    net.content['conv5_3'] = ConvLayer(net, net.content['relu5_2'])
    net.content['conv5_3'].W .set_value(W)
    net.content['conv5_3'].b.set_value(b)
    
    net.content['relu5_3'] = ReLULayer(net, net.content['conv5_3'])
    
    net.content['pool5'] = Pool2DLayer(net, net.content['relu5_3'])
    
    net.layer_opts['num_fc_node'] = 4096
    net.content['fc6'] = FCLayer(net, net.content['pool5'], (1, 512, 7, 7))
    W = data_dict['fc6'][0]
    W = np.reshape(W,(7,7,512,4096))
    W = np.transpose(W,(2,0,1,3))
    W = np.reshape(W,(7*7*512,4096))
    # W = np.transpose(W)
    # W = np.reshape(W, (4096, 25088, 1, 1))
    b = data_dict['fc6'][1]
    b = b.reshape(1,4096)
    net.content['fc6'].W.set_value(W)
    net.content['fc6'].b.set_value(b)
    
    net.content['fc7'] = FCLayer(net, net.content['fc6'], (1, 4096, 1, 1))
    W = data_dict['fc7'][0]
    # W = np.transpose(W)
    # W = np.reshape(W, (4096, 4096, 1, 1))
    b = data_dict['fc7'][1]
    b = b.reshape(1,4096)
    net.content['fc7'].W.set_value(W)
    net.content['fc7'].b.set_value(b)
    
    net.layer_opts['num_fc_node'] = 1000
    net.content['fc8'] = FCLayer(net, net.content['fc7'], (1, 4096, 1, 1))
    W = data_dict['fc8'][0]
    # W = np.transpose(W)
    # W = np.reshape(W, (1000, 4096, 1, 1))
    b = data_dict['fc8'][1]
    b = b.reshape(1,1000)
    net.content['fc8'].W.set_value(W)
    net.content['fc8'].b.set_value(b)

    return net

def npsigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def TrimCaptions(captions, img_idx):
    # There are multiple captions for one image
    # Choose one of them as train labels
    """
    :param captions: the list of captions
    :param img_idx: corresponding image of each caption
        for example captions[55] is for image with img_idx[55]
    :return: trimmed captions
    """
    num_samples = np.max(img_idx)+1
    max_len = captions.shape[1]
    new_captions = np.zeros((num_samples, max_len), dtype=theano.config.floatX)
    for i in range(0, len(img_idx)):
        idx = img_idx[i]
        new_captions[idx] = captions[i]
    return new_captions

def CreateData(n_word=None, null_word=0):
    # Get data from h5 file
    data_path = '../../data/MSCOCO/coco_captioning/'

    data = load_co_co(data_path)

    train_X = data['train_features']
    val_X = data['val_features']
    train_Y = data['train_captions']
    val_Y = data['val_captions']
    vocab = data['idx_to_word']
    train_idx = data['train_image_idxs']
    val_idx = data['val_image_idxs']
    train_urls = data['train_urls']
    val_urls = data['val_urls']
    max_len = train_Y.shape[1]
    
    if (n_word==None):
        n_word = len(vocab)

    num_sample = train_X.shape[0]
    num_val_sample = val_X.shape[0]
    num_cnn_features = train_X.shape[1]

    # Shape the data into 4D tensor
    train_X = np.reshape(train_X, (num_sample, num_cnn_features, 1, 1))
    val_X = np.reshape(val_X, (num_val_sample, num_cnn_features, 1, 1))

    # Covert captions to one-of-k vectors
    I = np.eye(n_word)
    I = np.asarray(I, dtype=theano.config.floatX)
    train_Y = TrimCaptions(train_Y, train_idx)
    val_Y = TrimCaptions(val_Y, val_idx)
    train_Y = I[np.asarray(train_Y, dtype=np.uint32)]
    val_Y = I[np.asarray(val_Y, dtype=np.uint32)]
    train_Y = np.reshape(train_Y, (num_sample, max_len, n_word, 1))
    val_Y = np.reshape(val_Y, (num_val_sample, max_len, n_word, 1))

    words = np.argmax(train_Y[:, 1:, :, :], axis=2)
    remove_ind = words == null_word
    train_weight = np.ones_like(words, dtype=theano.config.floatX)
    train_weight[remove_ind] = 0
    train_weight = train_weight.reshape((num_sample, max_len-1, 1, 1))
    train_weight = np.repeat(train_weight, n_word, 2)
    #train_weight = theano.shared(train_weight)

    words = np.argmax(val_Y[:, 1:, :, :], axis=2)
    remove_ind = words == null_word
    val_weight = np.ones_like(words, dtype=theano.config.floatX)
    val_weight[remove_ind] = 0
    val_weight = val_weight.reshape((num_val_sample, max_len-1, 1, 1))
    val_weight = np.repeat(val_weight, n_word, 2)

    # train_X = theano.shared(train_X)
    # train_Y = theano.shared(train_Y)
    # val_X = theano.shared(val_X)
    # val_Y = theano.shared(val_Y)

    return (train_X, train_Y, train_weight, train_urls, num_sample,
            val_X, val_Y, val_weight, val_urls, num_val_sample,
            vocab, n_word, max_len, num_cnn_features)



def train():
    trained_path = '../../data/trained_model/'
    # LSTM params
    n_word = 1004
    max_len = 40

    big_batch_size = np.asarray([10000], dtype=theano.config.floatX)

    # Create net
    net = ShowTellNet()
    net.name = 'ShowTellCOCO6'

    memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
    print('Memory: %.2f avail before loading numpy data' % (memory[0]/1024./1024/1024))

    train_X, train_Y, train_weight, train_urls, num_sample, \
    val_X, val_Y, val_weight, val_urls, num_val_sample, \
    vocab, n_word, max_len, num_cnn_features = CreateData(n_word)

    num_big_batch = np.ceil(np.asarray(num_sample, dtype=theano.config.floatX)/big_batch_size)
    num_big_val_batch = np.ceil(np.asarray(num_val_sample, dtype=theano.config.floatX)/big_batch_size) #num_val_sample
    num_big_epoch = 50


    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
    if (len(snapshot_list) == 0):

        # Trainer params
        trainer = Trainer()
        trainer.opts['batch_size'] = 400
        trainer.opts['save'] = True
        trainer.opts['save_freq'] = 2
        trainer.opts['num_sample'] = num_sample
        trainer.opts['num_val_sample'] = num_val_sample
        trainer.opts['validation'] = False
        trainer.opts['num_epoch'] = 1
        #trainer.opts['dzdw_norm_thres'] = 0.25
        #trainer.opts['dzdb_norm_thres'] = 0.025
        memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        print('Memory: %.2f avail before creating network' % (memory[0]/1024./1024/1024))

        # Setting params
        net.layer_opts['updatable'] = True
        net.net_opts['l1_learning_rate'] = np.asarray(0.005, theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.005, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

        # Construct the network

        net.layer_opts['num_emb'] = 512
        net.content['dim_swap'] = SwapDim(net, net.content['input_img'], 1, 2)
        net.content['iwe'] = WordEmbLayer(net, net.content['dim_swap'],
                                          (trainer.opts['batch_size'], 1, num_cnn_features, 1))

        net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
                                         (trainer.opts['batch_size'], max_len - 1, n_word, 1))

        net.content['cat'] = Concat(net, net.content['iwe'], net.content['we'], 1)

        net.layer_opts['num_lstm_node'] = 1024

        net.content['lstm'] = LSTM(net, net.content['cat'],
                                   (trainer.opts['batch_size'], max_len - 1, net.layer_opts['num_emb'], 1))

        net.layer_opts['num_affine_node'] = n_word
        net.content['affine'] = AffineLayer(net, net.content['lstm'],
                                               (trainer.opts['batch_size'],
                                                max_len - 1,
                                                net.layer_opts['num_lstm_node'],
                                                1))
        
        #X1 = train_X[0:2,:,:,:]
        #Y1 = train_Y[0:2,:-1,:,:]
        #h_m1=np.zeros((1, 2, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)
        #c_m1=np.zeros((1, 2, net.layer_opts['num_lstm_node']), dtype=theano.config.floatX)

        #lstm_out = net.content['affine'].output.eval({net.input[0]:X1, net.input[1]:Y1, 
        #    net.content['lstm'].h_m1_sym:h_m1,
        #    net.content['lstm'].c_m1_sym:c_m1})
        #print(lstm_out.shape)

        net.content['lstm_r'] = LSTMRemove(net, net.content['affine'], 1)
        #net.content['lstm_r'] = LSTMRemove(net, net.content['lstm'], 1)

        net.layer_opts['softmax_norm_dim'] = 2
        #net.content['softmax'] = SoftmaxLayer(net, net.content['lstm_r'])
        
        net.layer_opts['l2_term'] = 0.0125
        #net.content['l2'] = L2WeightDecay(net, net.content['lstm_r'])
        net.content['cost'] = SoftmaxLogLoss(net, net.content['lstm_r'])
        
        #net.content['cost'] = AggregateSumLoss([net.content['l2'], net.content['smloss']]) 

        memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        print('Memory: %.2f avail after creating network' % (memory[0]/1024./1024/1024))


        net.InitLR(0.2)
        trainer.InitParams(net)
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_sen', 'lstm_r']

        e = 0
        last_big_e = 0
    else:
        snapshot_list = sorted(snapshot_list)
        print('Loading latest snapshot at %s' % snapshot_list[-1])
        net, trainer, last_big_e = LoadList(snapshot_list[-1])
        trainer.opts['save_freq'] = 3
        trainer.opts['validation'] = True
        print('Finished loading snapshot')
        #trainer.opts['dzdw_norm_thres'] = 4
        #trainer.opts['dzdb_norm_thres'] = 0.04

        net.net_opts['l1_learning_rate'] = np.asarray(0.0008, theano.config.floatX)
        net.reset_opts['min_lr'] = np.asarray(0.0008, dtype=theano.config.floatX)
        net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']
        net.InitLR(0.2)
        trainer.InitParams(net)
        train_update_rule = trainer.InitUpdateRule(net)
        additional_output = ['input_sen', 'lstm_r']


    for big_e in range(last_big_e, num_big_epoch):
        for j in range(0, num_big_batch):
            big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
            if ((j+1)*big_batch_size > num_sample):
                big_batch_range = np.arange(j * big_batch_size, num_sample)
            trainer.opts['num_sample'] = big_batch_range.shape[0]
            big_batch_range = np.asarray(big_batch_range, dtype=np.uint32)            
            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
            print('Memory: %.2f avail before putting train data to shared' % (memory[0]/1024./1024/1024))
            train_Xj = theano.shared(train_X[big_batch_range, :, :, :])
            train_Yj = theano.shared(train_Y[big_batch_range, :, :, :])

            # Calculate hash value to avoid reading dupplicated data
            hash_weight = np.asarray([1.3**t for t in range(max_len)])
            hash_value = np.sum(np.argmax(train_Yj[0,:,:,0].eval(), axis=1)*hash_weight)
            print("Hash value: %f" % hash_value)

            train_weightj = theano.shared(train_weight[big_batch_range, :, :, :])
            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
            print('Memory: %.2f avail after' % (memory[0]/1024./1024/1024))

            net.InitTrainFunction(train_update_rule, [train_Xj, train_Yj[:, :-1, :, :]], train_Yj[:, 1:, :, :],
                additional_output, train_weightj)

            trainer.opts['validation'] = False
            trainer.opts['train'] = True
            main_loop = SGDRMainLoop(net, trained_path)
            main_loop.run(net, trainer, 0)
            train_Xj = None
            train_Yj = None
            train_weightj = None
            net.train_function = None
            print('Finished iteration %d of big epoch %d' % (j, big_e))

        trainer.opts['validation'] = True
        if (trainer.opts['validation']): #((big_e+1)%10 == 0) and 
            print('Starting validation...')
            for j in range(0, num_big_val_batch):
                big_batch_range = np.arange(j * big_batch_size, (j + 1) * big_batch_size)
                if ((j + 1) * big_batch_size > num_val_sample):
                    big_batch_range = np.arange(j * big_batch_size, num_val_sample)

                big_batch_range = np.asarray(big_batch_range, dtype=np.uint32)
                trainer.opts['num_val_sample'] = big_batch_range.shape[0]
                val_Xj = theano.shared(val_X[big_batch_range, :, :, :])
                val_Yj = theano.shared(val_Y[big_batch_range, :, :, :])
                val_weightj = theano.shared(val_weight[big_batch_range, :, :, :])
                net.InitValFunction([val_Xj, val_Yj[:, :-1, :, :]], val_Yj[:, 1:, :, :],
                                    additional_output, val_weightj)
                trainer.opts['validation'] = True
                trainer.opts['train'] = False
                main_loop = SGDRMainLoop(net, trained_path)
                main_loop.run(net, trainer, 0)
                val_Xj = None
                val_Yj = None
                val_weightj = None
                net.val_function = None
                print("Finished validation iteration %d of big epoch %d" % (big_e, j)) 
        # Epoch is done
        main_loop.LRRestart(net)
        
        if (big_e%trainer.opts['save_freq'] == 0):
            net1 = net.NNCopy()
            SaveList([net1, trainer, big_e], '../../data/trained_model/%s_e-%05d.dat' % (net.name, big_e))


def InferCOCO(idx, beam_size, train=True):
    """Infer caption from given COCO data
    :type idx: int
    :param idx: index of COCO image in either train or val data

    :type train: bool
    :param train: indicate whether the sample will be taken from train or validation data
    """
    net = ShowTellNet()
    net.name = 'ShowTellCOCO6'
    
    # Load train and test data
    train_X, train_Y, train_weight, train_urls, num_sample, \
        val_X, val_Y, val_weight, val_urls, num_val_sample, \
        vocab, n_word, max_len, num_cnn_features = CreateData()
    vocab = np.asarray(vocab)
   
    # Take a sample out of the data for inference
    if (train):
        X = train_X[idx,:,:,:]
        Y = train_Y[idx, :, :, :]
        url = train_urls[idx]
    else:
        X = val_X[idx,:,:,:]
        Y = val_Y[idx, :, :, :]
        url = val_urls[idx]

    # pdb.set_trace()
    #X = X.reshape((X.shape[0], X.shape[1], X.shape[2], -1))
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
    Y = np.reshape(Y, (1, Y.shape[0], Y.shape[1], Y.shape[2]))
    # pdb.set_trace()


    # get some memory back
    del train_X
    del train_Y
    del train_weight
    del val_weight
    del val_X
    
    # Put sample into GPU mem
    #val_sample = theano.shared(val_sample)
    #val_label = theano.shared(val_label)

    start_token = 1
    stop_token = 2

    trained_path = '../../data/trained_model/ShowTellCOCO/model_weight/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
    assert len(snapshot_list) != 0, ('Can\'t find net data at %s' % trained_path)
    snapshot_list = sorted(snapshot_list)
    print('Loading latest snapshot at %s' % snapshot_list[-1])
    net, trainer, e = LoadList(snapshot_list[-1])

    iwe_out = net.content['iwe'].output.eval({net.input[0]: X})
    beam_search = BeamSearch(net, iwe_out, net.content['lstm'], net.content['we'], net.content['affine'], max_len, beam_size, start_token, stop_token)
    print('Numerical output:\n %s' % beam_search.output)
    print('Caption output:\n %s' % " ".join(list(vocab[np.asarray(beam_search.output, dtype=np.uint16)])))
    # print('Actual numerical output: \n %s' % Y)
    # print('Actual caption:\n %s' % " ".join(list(vocab[np.asarray(Y, dtype=np.uint16)])))
    print('Image URL (load for visualization):\n %s' % url)

def InferImage(img_path):
    net = ShowTellNet()
    net.name = 'ShowTellCOCO6'
    
    # Load train and test data
    train_X, train_Y, train_weight, train_urls, num_sample, \
        val_X, val_Y, val_weight, val_urls, num_val_sample, \
        vocab, n_word, max_len, num_cnn_features = CreateData()
    vocab = np.asarray(vocab)  
    
	# get some memory back
    del train_X
    del train_Y
    del train_weight
    del val_weight
    del val_X

    start_token = 1
    stop_token = 2

    trained_path = '../../data/trained_model/ShowTellCOCO/model_weight/'
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
    assert len(snapshot_list) != 0, ('Can\'t find net data at %s' % trained_path)
    snapshot_list = sorted(snapshot_list)
    print('Loading latest snapshot at %s' % snapshot_list[-1])
    net, trainer, e = LoadList(snapshot_list[-1])

    VGGNet = LoadVGG()
    # X = mpimg.imread(img_path)
    # X = scipy.misc.imresize

    if(type(img_path) == list):
    	for path in img_path:
    		X = mpimg.imread(path)
    		Infer(X, VGGNet, net, vocab, beam_size = 20, max_len = max_len, start_token = start_token, stop_token = stop_token)
    else:
	    # Preprocess image
	    X = mpimg.imread(img_path)
	    Infer(X, VGGNet, net, vocab, beam_size = 20, max_len = max_len, start_token = start_token, stop_token = stop_token)

def Infer(X, VGGNet, net, vocab, **kwargs):

	beam_size = kwargs.pop('beam_size', 20)
	max_len   = kwargs.pop('max_len', 40)
	start_token = kwargs.pop('start_token', 1)
	stop_token = kwargs.pop('stop_token', 2)

	X = scipy.misc.imresize(X, [224, 224], 'bicubic')
	X = np.reshape(X, (1,224,224,3))

	# Change RGB to BGR
	X = X[:,:,:,[2,1,0]]

	X = np.transpose(X, (0, 3,2,1))
	X = VGG_preprocess(X)
	X = X.astype(theano.config.floatX)
	X = VGGNet.content['fc7'].output.eval({VGGNet.input[0]: X})

	iwe_out = net.content['iwe'].output.eval({net.input[0]: X})
	beam_search = BeamSearch(net, iwe_out, net.content['lstm'], net.content['we'], net.content['affine'], max_len, beam_size, start_token, stop_token)
	print('Numerical output:\n %s' % beam_search.output)
	print('Caption output:\n %s' % " ".join(list(vocab[np.asarray(beam_search.output, dtype=np.uint16)]))) 	 

if __name__ == '__main__':
    #train()
    beam_size = 20
  
    # InferImage('../../data/random_test_data/dog.jpg')
    # InferImage('../../data/random_test_data/IMG_1313.JPG')
    # InferImage('../../data/random_test_data/IMG_3073.JPG')
    # InferImage('../../data/random_test_data/IMG_8719.JPG')

    list_image_path = ['../../data/random_test_data/Family_2.jpg', 
    				   '../../data/random_test_data/Family_3.jpg', 
    				   '../../data/random_test_data/Family_4.jpg',
    				   '../../data/random_test_data/Korea.jpg']

    InferImage(list_image_path)

    #InferImage('../../data/random_test_data/grad1.jpg')
    #InferImage('../../data/random_test_data/IMG_3151.JPG')
    #InferCOCO(5, beam_size, False)
    #InferCOCO(6, beam_size, False)
    # InferCOCO(50, beam_size, False)
    #InferCOCO(60, beam_size, False)
    # InferCOCO(987, beam_size, False)   #987: a big church with a clock in front of the church
    # InferCOCO(2511, beam_size, False)  #2511: an elephant is walking behind a wall at the zoo
    #InferCOCO(777, beam_size, False)   #164: a large group of people in a desert on horses   #777: a pepperoni pizza with UNK and it END  
    #InferCOCO(300, beam_size, False)  #6384: two white sheep in a grassy field  #300: a train is parked on a train track
    #InferCOCO(6384, beam_size, False)

    # for i in np.random.randint(2000, 6000, size = 10):
    # 	InferCOCO(i, beam_size, False)
    #bp = 1
