import numpy as np
import h5py
from LayerProvider import *
from NeuralNet import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc
data_path = '/home/kien/PycharmProjects/data/vgg16.npy'
data_dict = np.load(data_path).item()
VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
VGG_MEAN = np.reshape(VGG_MEAN, (1,3,1,1))
# VGG_MEAN = np.repeat(VGG_MEAN, 244)
# Load a test image
# I = mpimg.imread('/media/kien/DATA/Kien/Car/car140.png')
I = mpimg.imread('scroll0015.jpg')
# I = I[:,200:1401,:]
plt.imshow(I)
I = np.asarray(scipy.misc.imresize(I, (224,224), 'bicubic'), dtype=np.float32)

# Convert RGB to BGR
I = I[:,:,[2,1,0]]
I = np.transpose(I, (2,0,1))
I = np.reshape(I, (1, 3, 224, 224))
I = np.asarray(I - VGG_MEAN, dtype=np.float32)

# Construct the network
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

# net.content['softmax'] = SoftmaxLayer(net, net.content['fc8'])

net.content['softmax'] = LogSoftmaxLayer(net, net.content['fc8'])
a = net.content['softmax'].output.eval({net.input[0]: I})

f = open('synset.txt')
lines = f.readlines()
f.close()

best = np.argmax(a)

print(lines[best])

bp = 1
