import sys
sys.path.insert(0, '/home/graphicsminer/Projects/image-captioning/data-prepare/coco/PythonAPI')
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import matplotlib.image as mpimg
import matplotlib.pylab as plt
import scipy.misc 
from LayerProvider import *
from NeuralNet import *
import numpy as np
from Utils import LoadList, LoadH5
from Trainer import Trainer
from MainLoop import *
from PrepareCOCOData import VGG_preprocess
import glob
import scipy
import pdb
from coco_utils import *

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

	return net

def train_Attend_224():
    trained_path = '../../data/trained_model/'
    cap_data_path = "../../data/mscoco/MSCOCO_processed/MSCOCO_224_capdata_train_%d.h5"    
    img_data_path = "../../data/mscoco/MSCOCO_processed/MSCOCO_224_imgdata_train_%d.h5"  
    val_cap_data_path = "../../data/mscoco/MSCOCO_processed/MSCOCO_224_capdata_val_%d.h5"
    val_img_data_path = "../../data/mscoco/MSCOCO_processed/MSCOCO_224_imgdata_val_%d.h5"
    fourth_cv_mv = "../../data/mscoco/MSCOCO_processed/4thconvo_meanvar.dat"
    [relu_mean, relu_std] = LoadList(fourth_cv_mv)
    relu_mean = theano.shared(relu_mean.astype(theano.config.floatX))
    relu_std = theano.shared(relu_std.astype(theano.config.floatX))

    # LSTM params
    n_word = 1004
    max_len = 40
    
    memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
    #print('Memory: %.2f avail before putting train data to shared' % (memory[0]/1024./1024/1024))
    
    #create net
    net = ShowTellNet() 
    net = LoadVGG_Attend(net)
    #net.name = "ShowAttendTellCOCO_Re7e-4"
    net.name = "ShowAttendTellBugFind4"
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
    
    num_big_epoch = 5000
    big_batch_size = np.asarray([2000], dtype=theano.config.floatX)
        
    if(len(snapshot_list) == 0):
	
	# Trainer params
	trainer = Trainer()
	trainer.opts['batch_size'] = 20
	trainer.opts['save'] = False
	trainer.opts['save_freq'] = 10
	#trainer.opts['num_sample'] = num_sample
	#trainer.opts['num_val_sample'] = num_val_sample
	trainer.opts['validation'] = False
	trainer.opts['num_epoch'] = 1
	trainer.opts['dzdw_norm_thres'] = 1
	trainer.opts['dzdb_norm_thres'] = 0.01
        #self.opts['update_rule_name'] = 'ADAM'
	net.layer_opts['updatable'] = True

	# Setting params
	net.net_opts['l1_learning_rate'] = np.asarray(0.005, theano.config.floatX)
	net.reset_opts['min_lr'] = np.asarray(0.005, dtype=theano.config.floatX)
	net.reset_opts['max_lr'] = net.net_opts['l1_learning_rate']

	#Constructing LSTM_ATTEND network from image_feature_region step-by-step
	# step 1: net.content['pool4'] reshape to (N, 196, 512) tensor - image_feature_region
	# step 2: using (N, 196, 512) image_feature_region tensor as input to compute h0, c0 - initial state memory of LSTM_ATTEND
	# step 4: construct DeepOutLayer from h_t, z_t output from LSTM_ATTEND layer
	# step 5: using DeepOutLayer output to compute output vector (instead of h_t), then negative log likelihood calculated by SoftMaxLogLoss Layer
        #pdb.set_trace()

	feature_shape = net.content['relu5_3'].output.shape
	new_shape = (feature_shape[0], feature_shape[1], T.prod(feature_shape[2:]))
        #pdb.set_trace() 
        net.content['relu5_3_norm'] = NormLayer(net, net.content['relu5_3'], relu_mean, relu_std)

	net.content['4th_convol_feature_region'] = ReshapeLayer(net, net.content['relu5_3'], new_shape)
       
       	net.layer_opts['num_region'] = 196
	net.content['average_feature_region'] = AverageLayer(net, net.content['4th_convol_feature_region'], 2)

	net.layer_opts['num_lstm_node'] = 512
	input_shape_h0 =  (1, 512)
	output_shape_h0 = (1 , net.layer_opts['num_lstm_node'])
	n_hidden_h0 = 512
	
	net.layer_opts['num_fc_node'] = n_hidden_h0
	net.content['h0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

	net.layer_opts['num_fc_node'] = output_shape_h0[1]
	hidden_shape = (input_shape_h0[1], n_hidden_h0)
	net.content['h0_initial'] = FCLayer(net, net.content['h0_hidden_layer'], hidden_shape)

	out_shape = net.content['h0_initial'].output.shape
	net.content['h0_initial'].output = net.content['h0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))
		
	net.layer_opts['num_fc_node'] = n_hidden_h0
	net.content['c0_hidden_layer'] = FCLayer(net, net.content['average_feature_region'], input_shape_h0, T.tanh)

	net.layer_opts['num_fc_node'] = output_shape_h0[1]
	net.content['c0_initial'] = FCLayer(net, net.content['c0_hidden_layer'], hidden_shape)

	out_shape = net.content['c0_initial'].output.shape
	net.content['c0_initial'].output = net.content['c0_initial'].output.reshape((-1, out_shape[0], out_shape[1]))

	#Word Embedding Layer
	net.layer_opts['num_emb'] = 400
	net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
	                                 (trainer.opts['batch_size'], max_len-1, n_word, 1))
        
	# we_out = net.content['we'].output.eval({net.input[1]: Y.eval()})

	net.layer_opts['num_lstm_node'] = 512 #
	net.layer_opts['context_dim']   = 1024
	net.layer_opts['num_dimension_feature'] = 512
	net.layer_opts['num_region'] = 196

	net.content['4th_convol_feature_region'].output = T.transpose(net.content['4th_convol_feature_region'].output, (0,2,1))
	
        #pdb.set_trace()
	net.content['lstm_attend'] = LSTM_Attend(net, net.content['we'], 
			(trainer.opts['batch_size'], max_len - 1, net.layer_opts['num_emb'], 1), 
			net.content['4th_convol_feature_region'].output, 
			initial_h0 = net.content['h0_initial'].output, initial_c0 = net.content['c0_initial'].output)

	net.layer_opts['num_deep_out_node'] = 512 #300
	net.layer_opts["n_word"] = n_word
	net.content['deep_out_layer'] = DeepOutputLayer(net, net.content['we'], net.content['lstm_attend'])
                
        net.layer_opts['num_affine_node'] = n_word
        #net.content['affine'] = AffineLayer(net, net.content['lstm_attend'],
        #       (1, max_len - 1, net.layer_opts['num_dimension_feature'], 1))

	net.layer_opts['l2_term'] = 0.000014
	net.content['l2'] = L2WeightDecay(net, net.content['deep_out_layer'])

	net.layer_opts['softmax_norm_dim'] = 2
	net.content['smloss'] = SoftmaxLogLoss(net, net.content['deep_out_layer'])

	net.content['cost'] = AggregateSumLoss([net.content['l2'], net.content['smloss']]) 
        
        # SHOW TELL CODE
        #net.layer_opts['num_emb'] = 512
        #num_cnn_features = 4096
        #net.content['dim_swap'] = SwapDim(net, net.content['fc7'], 1, 2)
        #net.content['iwe'] = WordEmbLayer(net, net.content['dim_swap'],
        #                                  (trainer.opts['batch_size'], 1, num_cnn_features, 1))

        #net.content['we'] = WordEmbLayer(net, net.content['input_sen'],
        #                                 (trainer.opts['batch_size'], max_len - 1, n_word, 1))

        #net.content['cat'] = Concat(net, net.content['iwe'], net.content['we'], 1)

        #net.layer_opts['num_lstm_node'] = 1024

        #net.content['lstm_attend'] = LSTM(net, net.content['cat'],
        #                           (trainer.opts['batch_size'], max_len - 1, net.layer_opts['num_emb'], 1))

        #net.layer_opts['num_affine_node'] = n_word
        #net.content['affine'] = AffineLayer(net, net.content['lstm_attend'],
        #                                       (trainer.opts['batch_size'],
        #                                        max_len - 1,
        #                                        net.layer_opts['num_lstm_node'],
        #                                        1))
        #net.content['lstm_r'] = LSTMRemove(net, net.content['affine'], 1)
       
        #net.layer_opts['softmax_norm_dim'] = 2
              
        #net.layer_opts['l2_term'] = 0.000001
        #net.content['l2'] = L2WeightDecay(net, net.content['lstm_r']) 
        #net.content['smloss'] = SoftmaxLogLoss(net, net.content['lstm_r'])
        #net.content['cost'] = AggregateSumLoss([net.content['l2'], net.content['smloss']])
	net.InitLR(0.2) 
	memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        print('Memory: %.2f avail before initialize params' % (memory[0]/1024./1024/1024))

        trainer.InitParams(net)
	print("Done init params")
	train_update_rule = trainer.InitUpdateRule(net)
	print("Done init update rule")
	#additional_output = ['affine', 'l2']
        additional_output = ['deep_out_layer', 'l2']


	# net.InitValFunction([val_X, val_Y[:,:-1,:,:]], val_Y[:,1:,:,:], 
	# 	additional_output, val_weight, net.content['lstm_attend'].output_z)
	e = 0
	last_big_e = 0
    else:
	snapshot_list = sorted(snapshot_list)
	print('Loading latest snapshot at %s' % snapshot_list[-1])
        e = 0
        [net, trainer, last_big_e] = LoadList(snapshot_list[-1])
	
        net.InitLR(0.2)
        memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        print('Memory: %.2f avail before initialize params' % (memory[0]/1024./1024/1024))

        trainer.InitParams(net)
	print("Done init params")
	train_update_rule = trainer.InitUpdateRule(net)
	print("Done init update rule")
	additional_output = ['deep_out_layer', 'l2']
    for big_e in range(last_big_e+1, num_big_epoch):
        # Load train data
        h_list = range(11)
        np.random.shuffle(h_list)
        for h in [0]:
            #break
            #if (not ('train_X' in locals())):
            train_X = LoadH5(img_data_path % h)
            dict_key = train_X.keys()[0]
            train_X = train_X[dict_key]
            num_sample = train_X.shape[0]
            # train_Y has the shape of (num_sample, 5, max_len, n_word, 1)
            train_Y = LoadH5(cap_data_path % h)
            dict_key = train_Y.keys()[0]
            train_Y = train_Y[dict_key]
            Y_shape = train_Y.shape
            
            # For debugging
            #train_X = train_X[0:100,:,:,:]
            #train_Y = train_Y[0:100,:,:,:,:]
            #num_sample = 100

            train_Y = train_Y.reshape(5*num_sample, Y_shape[2], Y_shape[3], 1)
            random_caption_idx = net.net_opts['rng'].randint(0,5,num_sample) + np.asarray([i*5 for i in range(num_sample)])
            
            # Each image has 5 captions, pick one at random
            train_Y = train_Y[random_caption_idx, :, :, :] 
            #train_Y = train_Y[:, 0, :, :, :]
            train_Y = train_Y.astype(theano.config.floatX)   

            # Create weight from train_Y
            train_weight = np.copy(train_Y)
            train_weight = train_weight[:,1:,:,:]
            weight_shape = train_weight.shape
            train_weight = (train_weight[:, :, 0, 0] == 0).reshape(weight_shape[0], weight_shape[1], 1, 1)
            train_weight = np.repeat(train_weight, weight_shape[2], 2)
            train_weight = np.repeat(train_weight, weight_shape[3], 3)
            train_weight = train_weight.astype(theano.config.floatX)

            num_big_batch_iteration = np.ceil(np.asarray(num_sample, dtype=theano.config.floatX)/big_batch_size)
            
            for j in [0]:
            #for j in range(0, num_big_batch_iteration):
	        big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
	        
                if ((j+1)*big_batch_size > num_sample):
	            big_batch_range = np.arange(j * big_batch_size, num_sample)

	        if (net.train_function == None):

                    trainer.opts['num_sample'] = big_batch_range.shape[0]
	            big_batch_range = np.asarray(big_batch_range, dtype=np.uint32)
                    np.random.shuffle(big_batch_range)
	            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
	            print('Memory: %.2f avail before putting train data to shared' % (memory[0]/1024./1024/1024))

	            train_Xj = theano.shared(train_X[big_batch_range, :, :, :])
	            train_Yj = theano.shared(train_Y[big_batch_range, :, :, :])
                    hash_weight = np.asarray([1.3**t for t in range(max_len)])
                    hash_value = np.sum(np.argmax(train_Yj[0,:,:,0].eval(), axis=1)*hash_weight)
                    print(hash_value)
                    #pdb.set_trace()

	            train_weightj = theano.shared(train_weight[big_batch_range, :, :, :])
	            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
	            print('Memory: %.2f avail after' % (memory[0]/1024./1024/1024))
                    val_Xtest = train_Xj.eval()[0:2,:,:,:]
                    val_Ytest = train_Yj.eval()[0:2,:-1,:,:]
                    z_m1_dummy = np.zeros((1, 2, net.content['lstm_attend'].Z_shape[0]), dtype=theano.config.floatX)
       
                    #pdb.set_trace()
                    do0 = net.content['deep_out_layer'].output.eval({ \
                        net.input[0]: val_Xtest, \
                        net.input[1]: val_Ytest, \
                        net.content['lstm_attend'].z_m1_sym: z_m1_dummy \
                    })
               
                    net.InitTrainFunction(train_update_rule, [train_Xj, train_Yj[:,:-1,:,:]], train_Yj[:,1:,:,:], 
	        		additional_output, train_weightj)

                              
	            print("Done init train function")

	        print("start training")
	        trainer.opts['validation'] = False
	        trainer.opts['train'] = True
	        main_loop = SGDRMainLoop(net, trained_path)
	        main_loop.run(net, trainer, e)
                
	        #train_Xj = None
	        #train_Yj = None
	        #train_weightj = None
	        #net.train_function = None
                print('Finished iteration %d, h5 %d, of big epoch %d' % (j, h, big_e))
                
                plt.figure()
                plt.plot(trainer.all_i[-1000:])
                plt.savefig('ST_all_i_last1000_4.png')
                plt.close()
                
                plt.figure()
                plt.plot(trainer.all_i)
                plt.savefig('ST_all_i_4.png')
                plt.close()
            if (big_e%trainer.opts['save_freq']==0):
                net1 = net.NNCopy()
                #SaveList([net1, trainer, big_e], '../../data/trained_model/%s_e-%05d.dat' % (net.name, big_e))

        # Validating frequency is the same with save freq
        #if (big_e % (trainer.opts['save_freq'] * 2) == 0):
        #    for h in range(6):
        #        val_X = LoadH5(val_img_data_path % h)
        #        dict_key = val_X.keys()[0]
        #        val_X = val_X[dict_key]
        #        num_val_sample = val_X.shape[0]
        #        
        #        # val_Y has the shape of (num_val_sample, 5, max_len, n_word, 1)
        #        val_Y = LoadH5(val_cap_data_path % h)
        #        
        #        dict_key = val_Y.keys()[0]
        #        val_Y = val_Y[dict_key]
        #        Y_shape = val_Y.shape
        #        val_Y = val_Y.reshape(5*num_val_sample, Y_shape[2], Y_shape[3], 1)

        #        random_caption_idx = net.net_opts['rng'].randint(0,5,num_val_sample) + np.asarray([i*5 for i in range(num_val_sample)])
        #        # Each image has 5 captions, pick one at random
        #        val_Y = val_Y[random_caption_idx, :, :, :]
        #        val_Y = val_Y.astype(theano.config.floatX) 
        #        # Create weight from val_Y
        #        val_weight = np.copy(val_Y)
        #        val_weight = val_weight[:,1:,:,:]
        #        weight_shape = val_weight.shape
        #        val_weight = (val_weight[:, :, 0, 0] == 0).reshape(weight_shape[0], weight_shape[1], 1, 1)
        #        val_weight = np.repeat(val_weight, weight_shape[2], 2)
        #        val_weight = np.repeat(val_weight, weight_shape[3], 3)
        #        val_weight = val_weight.astype(theano.config.floatX)

        #        num_big_batch_iteration = np.ceil(np.asarray(num_val_sample, dtype=theano.config.floatX)/big_batch_size)

        #        for j in range(0, num_big_batch_iteration):
	#            big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
	#            
        #            if ((j+1)*big_batch_size > num_val_sample):
	#                big_batch_range = np.arange(j * big_batch_size, num_val_sample)
	#            
        #            trainer.opts['num_val_sample'] = big_batch_range.shape[0]
	#            big_batch_range = np.asarray(big_batch_range, dtype=np.uint32)            
	#            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
	#            print('Memory: %.2f avail before putting val data to shared' % (memory[0]/1024./1024/1024))
	#            val_Xj = theano.shared(val_X[big_batch_range, :, :, :])
	#            val_Yj = theano.shared(val_Y[big_batch_range, :, :, :]) 
        #                           
        #            hash_weight = np.asarray([1.3**t for t in range(max_len)])
        #            hash_value = np.sum(np.argmax(val_Yj[0,:,:,0].eval(), axis=1)*hash_weight)
        #            print(hash_value)
	#            val_weightj = theano.shared(val_weight[big_batch_range, :, :, :])

	#            memory = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
	#            print('Memory: %.2f avail after' % (memory[0]/1024./1024/1024))

	#            net.InitValFunction([val_Xj, val_Yj[:,:-1,:,:]], val_Yj[:,1:,:,:], 
	#            		additional_output, val_weightj)
	#            print("Done init val function")
 
	#            print("start validating")
	#            trainer.opts['validation'] = True
	#            trainer.opts['train'] = False
	#            main_loop = SGDRMainLoop(net, trained_path)
	#            main_loop.run(net, trainer, e)
        #            
	#            val_Xj = None
	#            val_Yj = None
	#            val_weightj = None
	#            net.val_function = None
        #            print('Finished validating at iteration %d, h5 %d, of big epoch %d' % (j, h, big_e))


def InferMSCOCO(ID):
    """
    Using a trained Show Attend Tell model to generate caption from an image
    The image is load from MSCOCO (with ground truth caption)
    
    :type ID: int
    :param ID: id of the image in MSCOCO validation set
    """
    max_len = 40
    n_word = 1004

    w = 224
    h = 224
    val_img_path = '../../data/mscoco/val2014/'
    val_json_path = '/home/graphicsminer/data/mscoco/annotations/captions_val2014.json'
    trained_path = '../../data/trained_model/'
    vocab_path = '../../data/mscoco/MSCOCO_processed/vocab.dat'

    file_type = '*.jpg'
    coco = COCO(val_json_path)
    img_keys = sorted(coco.imgs.keys())
    img_key = img_keys[ID]

    print("Infering caption for MSCOCO image with ID %d in validation set" % ID)
    img_info = coco.loadImgs([img_key])
    file_name = val_img_path + img_info[0]['file_name']
    file_url = img_info[0]['flickr_url']
    anns = coco.loadAnns(coco.getAnnIds([img_key]))
    
    # Preprocess image
    X = mpimg.imread(file_name)
    X = scipy.misc.imresize(X, [224, 224], 'bicubic')
    X = np.reshape(X, (1,224,224,3))
    
    # Change RGB to BGR
    X = X[:,:,:,[2,1,0]]

    X = np.transpose(X, (0, 3,2,1))
    X = VGG_preprocess(X)
    X = X.astype(theano.config.floatX)
    
    # Generate a <START> and <END> tokens
    vocab = LoadList(vocab_path)
    vocab = vocab[0]
    start_token = u'<START>'
    end_token = u'<END>'
    start_ID =  np.where(vocab == start_token)
    end_ID = np.where(vocab == end_token)
    
    # Generate fist word of the setence
    X_sen0 = np.zeros((1, 1, n_word, 1), dtype=theano.config.floatX)
    X_sen0[0,0,start_ID,0] = 1.0
    
    net = ShowTellNet() 
    net = LoadVGG_Attend(net)
    net.name = "ShowAttendTellCOCO_Affine"
    snapshot_list = glob.glob(trained_path + net.name + '*.dat')
     
    if(len(snapshot_list) >= 0):
        print('Loading neural network snapshot from %s' % snapshot_list[-1])
        [net, trainer, last_big_e] = LoadList(snapshot_list[-1])

    # Generate z[-1]
    z_m1 = np.zeros((1, 1, net.content['lstm_attend'].Z_shape[0]), dtype=theano.config.floatX)
    

    #pdb.set_trace()
    deep_out0 = net.content['deep_out_layer'].output.eval({ \
        net.input[0]: X, \
        net.input[1]: X_sen0, \
        net.content['lstm_attend'].z_m1_sym: z_m1 \
        })
    #pdb.set_trace()

def four_cv_mv_cal():
    """
    Cal calculate mean and variance of the 4th convolutional layer of VGG on MSCOCO train dataset
    """
    net = ShowTellNet() 
    net = LoadVGG_Attend(net)
    net.name = "dummy"
    
    train_data_path = '../../data/mscoco/MSCOCO_processed/MSCOCO_224_imgdata_train_%d.h5'
    big_batch_size = 100
    all_relu = np.zeros((1, 512, 1, 1), dtype=np.float64)
    all_num_sample = 0
    for h in range(11):
        if (train_X == None):
            train_X = LoadH5(train_data_path % h)
            dict_keys = train_X.keys()
            train_X = train_X[dict_keys[0]]
        
        num_sample = train_X.shape[0]
        num_big_batch_iteration = int(np.ceil(np.asarray(num_sample, dtype=theano.config.floatX)/big_batch_size))
        
        for j in range(0, num_big_batch_iteration):
	    big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
	        
            if ((j+1)*big_batch_size > num_sample):
	        big_batch_range = np.arange(j * big_batch_size, num_sample)
            X = train_X[big_batch_range, :, :, :] 
            relu5_3 = net.content['relu5_3'].output.eval({ \
                net.input[0]: X \
            })
            relu5_3 = np.sum(relu5_3, axis=0, keepdims=True)
            relu5_3 = np.sum(relu5_3, axis=2, keepdims=True)
            relu5_3 = np.sum(relu5_3, axis=3, keepdims=True)

            all_relu = all_relu + relu5_3.astype(np.float64)
            all_num_sample += num_sample
            
            print('Calculate mean at h=%d and j=%d' % (h, j))
     
    mean_relu = all_relu / np.asarray((14.0*14.0*all_num_sample), dtype=np.float64)
    all_relu = np.zeros((1, 512, 1, 1), dtype=np.float64)

    for h in range(11):
       
        train_X = LoadH5(train_data_path % h)
        dict_keys = train_X.keys()
        train_X = train_X[dict_keys[0]]
        
        num_sample = train_X.shape[0]
        num_big_batch_iteration = int(np.ceil(np.asarray(num_sample, dtype=theano.config.floatX)/big_batch_size))
         
        for j in range(0, num_big_batch_iteration):
	    big_batch_range = np.arange(j*big_batch_size, (j+1)*big_batch_size)
	        
            if ((j+1)*big_batch_size > num_sample):
	        big_batch_range = np.arange(j * big_batch_size, num_sample)
            X = train_X[big_batch_range, :, :, :]
            relu5_3 = net.content['relu5_3'].output.eval({ \
                net.input[0]: X \
            })
            relu5_3 = relu5_3.astype(np.float64) - mean_relu
            relu5_3 = np.sum(relu5_3, axis=0, keepdims=True)
            relu5_3 = np.sum(relu5_3, axis=2, keepdims=True)
            relu5_3 = np.sum(relu5_3, axis=3, keepdims=True)
            relu5_3 = relu5_3**2

            all_relu = all_relu + relu5_3.astype(np.float64)
            print('Calculate std at h=%d and j=%d' % (h, j))

        all_num_sample += num_sample

    var_relu = all_relu / np.asarray((14.0*14.0*all_num_sample), dtype=np.float64)
    std_relu = np.sqrt(var_relu)
    pdb.set_trace()
    print('Saving mean and std')
    SaveList([mean_relu, std_relu], '../../data/mscoco/MSCOCO_processed/4thconvo_meanvar.dat')

if __name__=='__main__':
    #import matplotlib
    #matplotlib.use('Agg') 

    #four_cv_mv_cal()
    train_Attend_224()
    #InferMSCOCO(1)
