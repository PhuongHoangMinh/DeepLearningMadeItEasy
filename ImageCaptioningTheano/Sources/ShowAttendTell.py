from LayerProvider import *
from NeuralNet import *
import numpy as np
from Utils import LoadList, LoadH5
from Trainer import Trainer
from MainLoop import *
import glob
import scipy
import pdb
def LoadVGG_Attend(net):
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

	#after max pooling for 4th convolution layer -> 14*14*256 image-feature-region with respect to 224*224 input image
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

	return net

def VGG_preprocess_Flickr(data):
	VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
	VGG_MEAN = np.reshape(VGG_MEAN, (1, 3, 1, 1))
	return np.asarray(data-VGG_MEAN, dtype=theano.config.floatX)

def CreateDataFlick224(n_word = None):
	data_path = '../../data/Flickr8k_processed/'

	img_data_dict = LoadH5(data_path + 'Flickr8k_224_imgdata.dat')
	img_data = []
	for data in img_data_dict.values():
		img_data.append(data)
		print(len(data))

	labels = LoadList(data_path + ('Flickr8k_label_%d.dat'%n_word))
	vocab  = LoadList(data_path + ('Flickr8k_vocab_%d.dat'%n_word))

	#transpose (N, h, w, 3) to (N, 3, h, w)
	train_X = np.transpose(img_data[2], (0,3,1,2))
	val_X   = np.transpose(img_data[1], (0,3,1,2))
	test_X  = np.transpose(img_data[0], (0,3,1,2))

	#change RGB to BGR
	train_X = train_X[:,[2,1,0], :, :]
	val_X   = val_X[:, [2,1,0], :, :]
	test_X  = test_X[:, [2,1,0],: ,:]

	train_X = VGG_preprocess_Flickr(train_X)
	val_X   = VGG_preprocess_Flickr(val_X)
	test_X  = VGG_preprocess_Flickr(test_X)

	del test_X
	del img_data

	train_Y = np.asarray(labels[0], dtype=theano.config.floatX)
	train_Y = np.reshape(train_Y, (6000, 40, n_word, 1))
	val_Y = np.asarray(labels[1], dtype=theano.config.floatX)
	val_Y = np.reshape(val_Y, (1000, 40, n_word, 1))
	test_Y = np.asarray(labels[2], dtype=theano.config.floatX)
	test_Y = np.reshape(test_Y, (1000, 40, n_word, 1))

	words = np.argmax(train_Y[:,1:,:,:], axis=2)
	remove_ind = words == 17
	train_weight = np.ones_like(words, dtype=theano.config.floatX)
	train_weight[remove_ind] = 0
	train_weight = train_weight.reshape((6000, 39, 1, 1))
	train_weight = np.repeat(train_weight, n_word, 2)
	# train_weight = theano.shared(train_weight)

	words = np.argmax(val_Y[:,1:,:,:], axis=2)
	remove_ind = words == 17
	val_weight = np.ones_like(words, dtype=theano.config.floatX)
	val_weight[remove_ind] = 0
	val_weight = val_weight.reshape((1000, 39, 1, 1))
	val_weight = np.repeat(val_weight, n_word, 2)

	return (train_X, train_Y, train_weight, val_X, val_Y, val_weight)

def CreateDataFlick(n_word = None):
# Load flickr8k data
    data_path = '../../data/Flickr8k_processed/'
    
    #data_path = '/home/kien/PycharmProjects/data/Flickr8k_processed/'
    img_data = LoadList(data_path + 'Flickr8k_imgdata.dat')

    labels = LoadList(data_path + ('Flickr8k_label_%d.dat' % n_word))
    vocab = LoadList(data_path + ('Flickr8k_vocab_%d.dat' % n_word))

    train_X = np.transpose(img_data[2], (0,3,1,2))
    val_X = np.transpose(img_data[1], (0,3,1,2))
    test_X = np.transpose(img_data[0], (0,3,1,2))
   
    # Change RGB to BGR
    train_X = train_X[:, [2, 1, 0], :, :]
    val_X = val_X[:, [2, 1, 0], :, :]
    test_X = test_X[:, [2, 1, 0], :, :]

    train_X = VGG_preprocess_Flickr(train_X)
    val_X = VGG_preprocess_Flickr(val_X)
    test_X = VGG_preprocess_Flickr(test_X)
    pdb.set_trace()
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

def train_Attend():
	trained_path = '../../data/trained_model/'
	# LSTM params
	n_word = 2000
	max_len = 40
	
	train_X, train_Y, train_weight, val_X, val_Y, val_weight = CreateDataFlick(n_word)
        pdb.set_trace()
	#create net
	net = ShowTellNet()
	net.name = "ShowAttendTell"
	snapshot_list = glob.glob(trained_path + net.name + '*.dat')

	X = train_X[0:2,:,:,:]
	Y = train_Y[0:2,:,:,:]
	input_Y = train_Y[:,:-1,:,:]
	expected_Y = train_Y[:,1:,:,:]
	weight = train_weight[0:2,:, :, :]	

	if(len(snapshot_list) == 0):
		
		# Trainer params
		trainer = Trainer()
		trainer.opts['batch_size'] = 8
		trainer.opts['save'] = False
		trainer.opts['save_freq'] = 20
		trainer.opts['num_sample'] = 6000
		trainer.opts['num_val_sample'] = 1000
		trainer.opts['validation'] = False
		trainer.opts['num_epoch'] = 2000
		trainer.opts['dzdw_norm_thres'] = 1
		trainer.opts['dzdb_norm_thres'] = 0.01

		net = LoadVGG_Attend(net)
		net.layer_opts['updatable'] = True

		# Setting params
		net.net_opts['l1_learning_rate'] = np.asarray(0.005, theano.config.floatX)
		net.reset_opts['min_lr'] = np.asarray(0.005, dtype=theano.config.floatX)
		net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

		#Constructing LSTM_ATTEND network from image_feature_region step-by-step
		# step 1: net.content['pool4'] reshape to (N, 196, 512) tensor - image_feature_region
		# step 2: using (N, 196, 512) image_feature_region tensor as input to compute h0, c0 - initial state memory of LSTM_ATTEND
		# step 3: construct LSTM_ATTEND from h0, c0 (kwargs) and (N, 196, 512) image_feature_region tensor
		# step 4: construct DeepOutLayer from h_t, z_t output from LSTM_ATTEND layer
		# step 5: using DeepOutLayer output to compute output vector (instead of h_t), then negative log likelihood calculated by SoftMaxLogLoss Layer

		feature_shape = net.content['relu5_3'].output.shape
		new_shape = (feature_shape[0], feature_shape[1], T.prod(feature_shape[2:]))
		net.content['4th_convol_feature_region'] = ReshapeLayer(net, net.content['relu5_3'], new_shape)#net.content['pool4'].output.reshape()
		
		# Done
		# pdb.set_trace()
		# convol_out = net.content['4th_convol_feature_region'].output.eval({net.input[0]: X.eval()})
		# pdb.set_trace()

		net.layer_opts['num_region'] = 16
		# pdb.set_trace()
		net.content['average_feature_region'] = AverageLayer(net, net.content['4th_convol_feature_region'], 2)

		# Done
		# avg_out = net.content['average_feature_region'].output.eval({net.input[0]:X.eval()})

		net.layer_opts['num_lstm_node'] = 512
		input_shape_h0 =  (1, 512)
		output_shape_h0 = (1 , net.layer_opts['num_lstm_node'])
		n_hidden_h0 = 512
		
		#GENERATING H0
		# net.content['h0_initial'] = MLPLayer(net, net.content['average_feature_region'], 
		# 	input_shape = input_shape_h0, output_shape= output_shape_h0,n_hidden= n_hidden_h0)
		net.layer_opts['num_fc_node'] = n_hidden_h0
		net.content['h0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

		net.layer_opts['num_fc_node'] = output_shape_h0[1]
		hidden_shape = (input_shape_h0[1], n_hidden_h0)
		net.content['h0_initial'] = FCLayer(net, net.content['h0_hidden_layer'], hidden_shape)

		out_shape = net.content['h0_initial'].output.shape
		net.content['h0_initial'].output = net.content['h0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))

		# pdb.set_trace()
		# h0_init_out =net.content['h0_initial'].output.eval({net.input[0]: X.eval()})
		# pdb.set_trace()

		#GENERATING C0
		# net.content['c0_initial'] = MLPLayer(net, net.content['average_feature_region'], 
		# 	input_shape = input_shape_h0, output_shape = output_shape_h0,n_hidden= n_hidden_h0)
		net.layer_opts['num_fc_node'] = n_hidden_h0
		net.content['c0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

		net.layer_opts['num_fc_node'] = output_shape_h0[1]
		net.content['c0_initial'] = FCLayer(net, net.content['c0_hidden_layer'], hidden_shape)

		out_shape = net.content['c0_initial'].output.shape
		net.content['c0_initial'].output = net.content['c0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))

		#Word Embedding Layer
		net.layer_opts['num_emb'] = 512
		net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
		                                 (trainer.opts['batch_size'], max_len-1, n_word, 1))

		# pdb.set_trace()
		# we_out = net.content['we'].output.eval({net.input[1]: Y.eval()})
		# pdb.set_trace()

		net.layer_opts['num_lstm_node'] = 512 #
		net.layer_opts['context_dim']   = 1024
		net.layer_opts['num_dimension_feature'] = 512
		net.layer_opts['num_region'] = 16

		net.content['4th_convol_feature_region'].output = T.transpose(net.content['4th_convol_feature_region'].output, (0,2,1))

		net.content['lstm_attend'] = LSTM_Attend(net, net.content['we'], 
													(trainer.opts['batch_size'], max_len - 1, net.layer_opts['num_emb'], 1), 
													net.content['4th_convol_feature_region'].output, 
													initial_h0 = net.content['h0_initial'].output, initial_c0 = net.content['c0_initial'].output)


		# pdb.set_trace()
					
		# lstm_out = net.content['lstm_attend'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]:Y.eval(), 
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})
		# print(lstm_out[0].shape)
		# print(lstm_out[1].shape)
		# # print(lstm_out[2].shape)
		# pdb.set_trace()

		net.layer_opts['num_deep_out_node'] = 512 #300
		net.layer_opts["n_word"] = n_word
		net.content['deep_out_layer'] = DeepOutputLayer(net, net.content['we'], net.content['lstm_attend'])
		
		# net.layer_opts['num_affine_node'] = n_word
		# net.content['deep_out_layer'] = AffineLayer(net, net.content['lstm_attend'],
		#                                        (trainer.opts['batch_size'],
		#                                         max_len - 1,
		#                                         net.layer_opts['num_lstm_node'],
		#                                         1))

		# pdb.set_trace()
		# deep_out = net.content['deep_out_layer'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]: Y.eval(),
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})

		net.layer_opts['l2_term'] = 0.125
		net.content['l2'] = L2WeightDecay(net, net.content['deep_out_layer'])

		net.layer_opts['softmax_norm_dim'] = 2
		net.content['smloss'] = SoftmaxLogLoss(net, net.content['deep_out_layer'])

		net.content['cost'] = AggregateSumLoss([net.content['l2'], net.content['smloss']]) 

		# pdb.set_trace()
		# print(X.eval().shape)
		# print(Y.eval().shape)
		# print(weight.eval().shape)

		# logloss_out = net.content['cost'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]: input_Y.eval(),
		# 	net.output[0]: expected_Y.eval(),
		# 	net.weight[0]: weight.eval(),
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})

		# print("Done creating layer")
		# pdb.set_trace()

		net.InitLR(0.2) 
		trainer.InitParams(net)
		print("Done init params")
		train_update_rule = trainer.InitUpdateRule(net)
		print("Done init update rule")
		additional_output = ['input_sen', 'deep_out_layer', 'we', 'lstm_attend']

		net.InitTrainFunction(train_update_rule, [train_X, train_Y[:,:-1,:,:]], train_Y[:,1:,:,:], 
			additional_output, train_weight, net.weight[0])
		print("Done init train function")

		net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], additional_output, val_weight)
		print("Done init val function")

		# net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], 
		# 	additional_output, val_weight, net.content['lstm_attend'].output_z)
		e = 0
		last_big_e = 0
	else:
		snapshot_list = sorted(snapshot_list)
		print('Loading latest snapshot at %s' % snapshot_list[-1])

	print("start training")
	trainer.opts['validation'] = False
	trainer.opts['train'] = True
	main_loop = SGDRMainLoop(net, trained_path)
	main_loop.run(net, trainer, e)

def train_Attend_224():
	trained_path = '../../data/trained_model/'
	# LSTM params
	n_word = 2000
	max_len = 40
	
	train_X, train_Y, train_weight, val_X, val_Y, val_weight = CreateDataFlick224(n_word)

        pdb.set_trace()
	#create net
	net = ShowTellNet()
	net.name = "ShowAttendTell"
	snapshot_list = glob.glob(trained_path + net.name + '*.dat')

	X = train_X[0:2,:,:,:]
	Y = train_Y[0:2,:,:,:]
	input_Y = train_Y[:,:-1,:,:]
	expected_Y = train_Y[:,1:,:,:]
	weight = train_weight[0:2,:, :, :]

	num_sample = 6000
	num_big_epoch = 100
	big_batch_size = np.asarray([2000], dtype=theano.config.floatX)
	num_big_batch_iteration = np.ceil(np.asarray(num_sample, dtype=theano.config.floatX)/big_batch_size)

	if(len(snapshot_list) == 0):
		
		# Trainer params
		trainer = Trainer()
		trainer.opts['batch_size'] = 20
		trainer.opts['save'] = False
		trainer.opts['save_freq'] = 20
		trainer.opts['num_sample'] = 2000
		trainer.opts['num_val_sample'] = 1000
		trainer.opts['validation'] = False
		trainer.opts['num_epoch'] = 1
		trainer.opts['dzdw_norm_thres'] = 1
		trainer.opts['dzdb_norm_thres'] = 0.01

		net = LoadVGG_Attend(net)
		net.layer_opts['updatable'] = True

		# Setting params
		net.net_opts['l1_learning_rate'] = np.asarray(0.005, theano.config.floatX)
		net.reset_opts['min_lr'] = np.asarray(0.005, dtype=theano.config.floatX)
		net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

		#Constructing LSTM_ATTEND network from image_feature_region step-by-step
		# step 1: net.content['pool4'] reshape to (N, 196, 512) tensor - image_feature_region
		# step 2: using (N, 196, 512) image_feature_region tensor as input to compute h0, c0 - initial state memory of LSTM_ATTEND
		# step 3: construct LSTM_ATTEND from h0, c0 (kwargs) and (N, 196, 512) image_feature_region tensor
		# step 4: construct DeepOutLayer from h_t, z_t output from LSTM_ATTEND layer
		# step 5: using DeepOutLayer output to compute output vector (instead of h_t), then negative log likelihood calculated by SoftMaxLogLoss Layer

		feature_shape = net.content['relu5_3'].output.shape
		new_shape = (feature_shape[0], feature_shape[1], T.prod(feature_shape[2:]))
		net.content['4th_convol_feature_region'] = ReshapeLayer(net, net.content['relu5_3'], new_shape)#net.content['pool4'].output.reshape()
		
		# Done
		# pdb.set_trace()
		# convol_out = net.content['4th_convol_feature_region'].output.eval({net.input[0]: X.eval()})
		# pdb.set_trace()

		net.layer_opts['num_region'] = 196
		# pdb.set_trace()
		net.content['average_feature_region'] = AverageLayer(net, net.content['4th_convol_feature_region'], 2)

		# Done
		# avg_out = net.content['average_feature_region'].output.eval({net.input[0]:X.eval()})

		net.layer_opts['num_lstm_node'] = 512
		input_shape_h0 =  (1, 512)
		output_shape_h0 = (1 , net.layer_opts['num_lstm_node'])
		n_hidden_h0 = 512
		
		#GENERATING H0
		# net.content['h0_initial'] = MLPLayer(net, net.content['average_feature_region'], 
		# 	input_shape = input_shape_h0, output_shape= output_shape_h0,n_hidden= n_hidden_h0)
		net.layer_opts['num_fc_node'] = n_hidden_h0
		net.content['h0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

		net.layer_opts['num_fc_node'] = output_shape_h0[1]
		hidden_shape = (input_shape_h0[1], n_hidden_h0)
		net.content['h0_initial'] = FCLayer(net, net.content['h0_hidden_layer'], hidden_shape)

		out_shape = net.content['h0_initial'].output.shape
		net.content['h0_initial'].output = net.content['h0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))

		# pdb.set_trace()
		# h0_init_out =net.content['h0_initial'].output.eval({net.input[0]: X.eval()})
		# pdb.set_trace()

		#GENERATING C0
		# net.content['c0_initial'] = MLPLayer(net, net.content['average_feature_region'], 
		# 	input_shape = input_shape_h0, output_shape = output_shape_h0,n_hidden= n_hidden_h0)
		net.layer_opts['num_fc_node'] = n_hidden_h0
		net.content['c0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

		net.layer_opts['num_fc_node'] = output_shape_h0[1]
		net.content['c0_initial'] = FCLayer(net, net.content['c0_hidden_layer'], hidden_shape)

		out_shape = net.content['c0_initial'].output.shape
		net.content['c0_initial'].output = net.content['c0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))

		#Word Embedding Layer
		net.layer_opts['num_emb'] = 512
		net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
		                                 (trainer.opts['batch_size'], max_len-1, n_word, 1))

		# pdb.set_trace()
		# we_out = net.content['we'].output.eval({net.input[1]: Y.eval()})
		# pdb.set_trace()

		net.layer_opts['num_lstm_node'] = 512 #
		net.layer_opts['context_dim']   = 1024
		net.layer_opts['num_dimension_feature'] = 512
		net.layer_opts['num_region'] = 196

		net.content['4th_convol_feature_region'].output = T.transpose(net.content['4th_convol_feature_region'].output, (0,2,1))

		net.content['lstm_attend'] = LSTM_Attend(net, net.content['we'], 
													(trainer.opts['batch_size'], max_len - 1, net.layer_opts['num_emb'], 1), 
													net.content['4th_convol_feature_region'].output, 
													initial_h0 = net.content['h0_initial'].output, initial_c0 = net.content['c0_initial'].output)


		# pdb.set_trace()
					
		# lstm_out = net.content['lstm_attend'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]:Y.eval(), 
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})
		# print(lstm_out[0].shape)
		# print(lstm_out[1].shape)
		# # print(lstm_out[2].shape)
		# pdb.set_trace()

		net.layer_opts['num_deep_out_node'] = 512 #300
		net.layer_opts["n_word"] = n_word
		net.content['deep_out_layer'] = DeepOutputLayer(net, net.content['we'], net.content['lstm_attend'])
		
		# net.layer_opts['num_affine_node'] = n_word
		# net.content['deep_out_layer'] = AffineLayer(net, net.content['lstm_attend'],
		#                                        (trainer.opts['batch_size'],
		#                                         max_len - 1,
		#                                         net.layer_opts['num_lstm_node'],
		#                                         1))

		# pdb.set_trace()
		# deep_out = net.content['deep_out_layer'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]: Y.eval(),
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})

		net.layer_opts['l2_term'] = 0.125
		net.content['l2'] = L2WeightDecay(net, net.content['deep_out_layer'])

		net.layer_opts['softmax_norm_dim'] = 2
		net.content['smloss'] = SoftmaxLogLoss(net, net.content['deep_out_layer'])

		net.content['cost'] = AggregateSumLoss([net.content['l2'], net.content['smloss']]) 

		# pdb.set_trace()
		# print(X.eval().shape)
		# print(Y.eval().shape)
		# print(weight.eval().shape)

		# logloss_out = net.content['cost'].output.eval({net.input[0]: X.eval(), 
		# 	net.input[1]: input_Y.eval(),
		# 	net.output[0]: expected_Y.eval(),
		# 	net.weight[0]: weight.eval(),
		# 	net.content['lstm_attend'].z_m1_sym: np.zeros((1, 2, net.layer_opts['num_dimension_feature']), dtype=theano.config.floatX)})

		# print("Done creating layer")
		# pdb.set_trace()

		net.InitLR(0.2) 
		trainer.InitParams(net)
		print("Done init params")
		train_update_rule = trainer.InitUpdateRule(net)
		print("Done init update rule")
		additional_output = ['input_sen', 'deep_out_layer', 'we', 'lstm_attend']



		# net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], 
		# 	additional_output, val_weight, net.content['lstm_attend'].output_z)
		e = 0
		last_big_e = 0
	else:
		snapshot_list = sorted(snapshot_list)
		print('Loading latest snapshot at %s' % snapshot_list[-1])
	
	for big_e in range(last_big_e, num_big_epoch):
		for j in range(0, num_big_batch_iteration):
			big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
			if ((j+1)*big_batch_size > num_sample):
			    big_batch_range = np.arange(j * big_batch_size, num_sample)
			trainer.opts['num_sample'] = big_batch_range.shape[0]
			big_batch_range = np.asarray(big_batch_range, dtype=np.uint32)            
			memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
			print('Memory: %.2f avail before putting train data to shared' % (memory[0]/1024./1024/1024))
			train_Xj = theano.shared(train_X[big_batch_range, :, :, :])
			train_Yj = theano.shared(train_Y[big_batch_range, :, :, :])
			train_weightj = theano.shared(train_weight[big_batch_range, :, :, :])
			memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
			print('Memory: %.2f avail after' % (memory[0]/1024./1024/1024))

			net.InitTrainFunction(train_update_rule, [train_Xj, train_Yj[:,:-1,:,:]], train_Yj[:,1:,:,:], 
				additional_output, train_weightj, net.weight[0])
			print("Done init train function")

			# net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], additional_output, val_weight)
			# print("Done init val function")

			print("start training")
			trainer.opts['validation'] = False
			trainer.opts['train'] = True
			main_loop = SGDRMainLoop(net, trained_path)
			main_loop.run(net, trainer, e)

			train_Xj = None
			train_Yj = None
			train_weightj = None
			net.train_function = None
			print('Finished iteration %d of big epoch %d' % (j, big_e))

if __name__=='__main__':

	#train_Attend_224()
	train_Attend()