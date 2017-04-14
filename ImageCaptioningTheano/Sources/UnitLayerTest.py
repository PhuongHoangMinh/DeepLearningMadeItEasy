import numpy as np
from LayerProvider import *
from NeuralNet import *
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d, relu, elu, sigmoid
from theano.tensor import tanh
import theano.tensor as T
import pdb

def rel_error(x, y):
    """
    function to determine relative error between expected output results from our actual implementation of a layer
    :param x: expected output, arbitrary shape
    :param y: output from our implementation
    :return:  relative error > 1e-2 means that the result is probably wrong
                             <= e-7, you should be happy
    """
    return np.max(np.abs(x - y)/ np.maximum(1e-8, np.abs(x) + np.abs(y)))

def eval_numerical_gradient(f,x, verbose = True, h = 0.00001):
    """
    a naive implementation of numerical gradient of f at x
    :param f: should be a function that takes a single argument x
    :param x: is the point to evaluate the gradient at
    :param verbose:
    :param h:
    :return:
    """
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:

        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph  = f(x) #evaluate f(x+h)
        x[ix] = oldval - h
        fxmh  = f(x)
        x[ix] = oldval

        grad[ix] = (fxph - fxmh)/ (2*h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

def eval_numerical_gradient_array(f,x, df, h = 1e-5):
    """
    Evaluate a numeric gradient for a function with chain rule (df) - best fit purposes for rnn_step_backward(dnext_h, cache)
    :param f:
    :param x:
    :param df: This particular gradient check accounts for the derivative of the outside function
    :param h:
    :return:
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        fxph  = f(x)
        x[ix] = oldval - h
        fxmh  = f(x)
        x[ix] = oldval

        grad[ix] = np.sum((fxph - fxmh)*df) /(2*h)
        it.iternext()
    return grad

def conv_layer_naive(x, w, b, conv_param):

    out = None
    N,C, H,W = x.shape
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    stride = conv_param['stride']
    pad    = conv_param['pad']
    H_out  = 1 + (H+2*pad- HH)/stride
    W_out  = 1 + (W+2*pad- WW)/stride

    out = np.zeros((N,C,H_out, W_out))

    x_pad = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    x_window = x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    out[n,f,i,j] = np.sum(x_window*w[f]) + b[f]
    return out

#def UnitTestConvLayerWithoutBias():

#Checking convolution layer
def UnitTestConvLayer():
    x_shape = (2,3, 4, 4)
    w_shape = (3,3, 4, 4)
    x = np.linspace(-0.1, 0.5, num = np.prod(x_shape),dtype = theano.config.floatX).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num = np.prod(w_shape),dtype = theano.config.floatX).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num = 3, dtype = theano.config.floatX).reshape((1,3, 1,1))
    #print('initialize bias is: {} '.format(b))

    conv_param = {'stride': 2, 'pad': 1}
    #correct_out = np.array([[[[[-0.08759809, -0.10987781],
    #                           [-0.18387192, -0.2109216]],
    #                          [[0.21027089, 0.21661097],
    #                           [0.22847626, 0.23004637]],
    #                          [[0.50813986, 0.54309974],
    #                           [0.64082444, 0.67101435]]],
    #                         [[[-0.98053589, -1.03143541],
    #                           [-1.19128892, -1.24695841]],
    #                          [[0.69108355, 0.66880383],
    #                           [0.59480972, 0.56776003]],
    #                          [[2.36270298, 2.36904306],
    #                           [2.38090835, 2.38247847]]]]])

    b_zero = b.reshape(3,)
    correct_out = conv_layer_naive(x, w,b_zero, conv_param)
    

    input_data = theano.tensor.tensor4(name='x', dtype=theano.config.floatX)
    net = ShowTellNet(test_input_image = input_data)
    net.layer_opts['border_mode'] = (conv_param['pad'], conv_param['pad'])
    net.layer_opts['conv_stride'] = (conv_param['stride'], conv_param['stride'])
    net.layer_opts['filter_shape'] = w_shape
    
    conv_layer = ConvLayer(net, net.content['input_img'])
    conv_layer.W.set_value(w)
    conv_layer.b.set_value(b)
    
    #print('bias is:{} '.format(conv_layer.b.get_value()))
    #print('bias shape is:{} '.format(conv_layer.b.get_value().shape))
    #input_data = theano.tensor.tensor4(name='x', dtype=theano.config.floatX)
   
    out = theano.function([input_data], conv_layer.output)

    result =  out(x)
    
    print(result)
    print(result.shape)
    print(correct_out.shape)
    print(correct_out)
    print('Testing ConvLayer')
    print('difference: {}'.format(rel_error(correct_out, result)))

def UnitTestPoolLayer():
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)

    pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

    pool_size = (pool_param['pool_width'], pool_param['pool_height'])

    correct_out = np.array([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [0.03157895, 0.04631579]]],
                            [[[0.09052632, 0.10526316],
                              [0.14947368, 0.16421053]],
                             [[0.20842105, 0.22315789],
                              [0.26736842, 0.28210526]],
                             [[0.32631579, 0.34105263],
                              [0.38526316, 0.4]]]])

    #net = ShowTellNet()
    #max_pool_layer = Pool2DLayer(net, pre_layer)
    #out = max_pool_layer.output
    input_data = theano.tensor.tensor4(name='x', dtype=theano.config.floatX)
    output = pool_2d(input = input_data,
                     ds = pool_size,
                     ignore_border = True,
                     mode = 'max')

    out = theano.function([input_data], output)
    result = out(np.asarray(x, dtype = np.float32).reshape(x_shape))
    

    print(result)
    print('Testing max pool layer')
    print('difference: {}'.format(rel_error(correct_out, result)))

#Numerical gradient checking for the all parameters from the network 
def NumericalGradientCheckExample():
    N = 4 #number of training samples
    D = 4 #number of features
    M = 20 #number of hidden neurons in layer 1
    H = 10 #number of hidden neurons in layer 2
    
    #USING theano.grad
    x_shape = (N,D)
    #create random data
    dx = np.linspace(-0.1, 0.5, num = np.prod(x_shape),dtype = theano.config.floatX).reshape(x_shape)
    dy = np.zeros(4, dtype = 'int32')
    dy[0:2] = 1
    
    #building symbolic graph
    x = T.fmatrix(name = 'x')
    y = T.ivector(name = 'y')
    w1 = theano.shared(np.asarray(np.random.normal(0,0.1, [D, M]), dtype = theano.config.floatX))
    w2 = theano.shared(np.asarray(np.random.normal(0,0.1, [M, H]), dtype = theano.config.floatX))
    b1 = theano.shared(np.asarray(np.random.normal(0,0.1, [M,]), dtype = theano.config.floatX))
    b2 = theano.shared(np.asarray(np.random.normal(0,0.1, [H,]), dtype= theano.config.floatX))

    #theano grad setup
    z = T.nnet.sigmoid(T.dot(x, w1) + b1)
    h = T.nnet.softmax(T.dot(z, w2) + b2)
    cost = -T.mean(T.log(h)[T.arange(y.shape[0]), y])
    grad = T.grad(cost, w1)
    grad_fn = theano.function([], grad, givens={x:dx[0:4] , y:T.cast(dy[0:4], 'int32') })
    
    tgrad = grad_fn()
    print(tgrad)
    print('shape is: {}'.format(tgrad.shape))
    
    #numerical grad set up
    epsilon = 1e-3

    w1_s = T.fmatrix(name = 'w1_s')
    b1_s = T.fvector(name = 'b1_s')

    z_s = T.nnet.sigmoid(theano.dot(x, w1_s) + b1_s)
    h_s = T.nnet.softmax(theano.dot(z_s, w2) + b2)
    cost_s = -T.mean(T.log(h_s)[T.arange(y.shape[0]),y])
    fn_s = theano.function([w1_s, b1_s], cost_s, givens={x:dx[0:4], y:T.cast(dy[0:4], 'int32')})

    w1c = w1.get_value()
    b1c = b1.get_value()
    n_grad = np.zeros_like(w1c)
    delta  = np.zeros_like(w1c)

    for i in range(w1c.shape[0]):
        for j in range(w1c.shape[1]):
            delta[i][j] = epsilon
            fn_s_pep = fn_s(w1c+delta, b1c)
            fn_s_mep = fn_s(w1c-delta, b1c)
            n_grad[i][j] = (fn_s_pep - fn_s_mep)/(2*epsilon)
            delta[i][j] = 0.0
    
    print('numerical gradient is: ')
    print(n_grad)
    print('shape is {}'.format(n_grad.shape))
    dif = rel_error(n_grad, tgrad)
    print('relative error is: ', dif)

#checking LSTM implementation in Theano
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(x, axisdim = 0):
	return np.exp(x - np.max(x, axisdim, keepdims= True))/ np.sum(np.exp(x - np.max(x, axisdim, keepdims= True)), axis = axisdim)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
	"""
	forward pass for a single timestep of LSTM.
	the input data has dimension D, the hidden state has dimension H and we use a minibatch size of N

	Inputs:
	- x: Input data, shape (N, D)
	- prev_h: previous hidden state, of shape (N, H)
	- prev_c: previous hidden state, shape (N, H)
	- Wx: input2hidden weights, shape (D, 4H)
	- Wh: hidden2hidden weights, shape (H,4H)
	- b: biases, shape (4H)
	Returns a tuple of:
	- next_h: next hidden state, shape (N, H)
	- next_c: next cell state, shape (N, H)
	- cache
	"""
	next_h, next_c, cache = None, None, None

	H = prev_h.shape[1]
	a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
	i = sigmoid(a[:, :H])
	f = sigmoid(a[:, H:2*H])
	o = sigmoid(a[:, 2*H: 3*H])
	g = np.tanh(a[:, 3*H: ])

	next_c = f*prev_c + i*g
	next_h = o*np.tanh(next_c)
	cache  = (x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h)
	return next_h, next_c, cache

def lstm_forward(x, h0, Wx, Wh, b):
	"""
	Inputs:
	- x: Input data of shape (N,T, D)
	- h0: Initial hidden state of shape (N, H) - H - number of hidden node in the network
	- Wx: Weights for input2hidden connections, shape (D, 4H)
	- Wy: Weights for hidden2hidden connections, shpae (H, 4H)
	- b : biases of shape (4H,)
	Returns:
	- h : hidden states for all sequences, shape (N,T, H)
	- cache: values needed for backward 
	"""

	h, cache = None, None
	N, T, D = x.shape
	H = h0.shape[1] #H

	xs, hs, cs, ps = {}, {}, {}, {}
	hs[-1] = h0
	cs[-1] = np.zeros((N,H))
	h = np.zeros((N,T,H))


	for i in range(T):
		xs[i] = x[:, i, :]
		hs[i], cs[i], ps[i] = lstm_step_forward(xs[i], hs[i-1], cs[i-1], Wx, Wh, b)
		h[:, i, :] = hs[i]

	cache = (x, h0, Wx, Wh, hs, cs, ps)

	return (h, cache)

def UnitTestLSTM():
	N = 2 #number of sample
	D = 5 #dimension of input
	H = 4 #dimension of hidden
	T = 3 #length of per each sample
	#context_dim = 3

	x = np.linspace(-0.4, 0.6, num=N*T*D, dtype = theano.config.floatX).reshape(N, T, D)
	h0= np.linspace(-0.4, 0.8, num=N*H, dtype = theano.config.floatX).reshape(N, H)
	Wx= np.linspace(-0.2, 0.9, num=4*D*H, dtype = theano.config.floatX).reshape(D, 4*H)
	Wh= np.linspace(-0.3,0.6, num =4*H*H, dtype = theano.config.floatX).reshape(H,4*H)
	b = np.linspace(0.0, 0.0, num = 4*H, dtype = theano.config.floatX)
	#Wz= np.linspace(-0.3, 0.6, num=4*H*context_dim, dtype = theano.config.floatX).reshape(context_dim, 4*H)

	h, cache =lstm_forward(x, h0, Wx, Wh, b)

	h_print = h.reshape(N, T, H, 1)
	print(h_print)
	print('shape {}'.format(h_print.shape))

	#LSTM by Theano
	#input_data = theano.tensor.tensor4(name='x', dtype=theano.config.floatX)
	#net = ShowTellNet(test_input_image=input_data)
	#net.layer_opts['num_lstm_node']=H
	input_data = theano.tensor.tensor4(name ='x', dtype=theano.config.floatX)
	net_lstm = ShowTellNet(test_input_image=input_data)
	net_lstm.layer_opts['num_lstm_node'] = H
	lstm_theano_layer = LSTM(net_lstm, net_lstm.content['input_img'], (N, T-1, D, 1))

	lstm_theano_layer.W['i'].set_value(Wx[:, :H])
	lstm_theano_layer.W['f'].set_value(Wx[:, H:2*H])
	lstm_theano_layer.W['o'].set_value(Wx[:, 2*H:3*H])
	lstm_theano_layer.W['c'].set_value(Wx[:, 3*H:])

	lstm_theano_layer.U['i'].set_value(Wh[:, :H])
	lstm_theano_layer.U['f'].set_value(Wh[:, H:2*H])
	lstm_theano_layer.U['o'].set_value(Wh[:, 2*H:3*H])
	lstm_theano_layer.U['c'].set_value(Wh[:, 3*H:])

	b_theano= b.reshape(1, 1, 4*H)
	lstm_theano_layer.b['i'].set_value(b_theano[:, :, :H])
	lstm_theano_layer.b['f'].set_value(b_theano[:, :, H:2*H])
	lstm_theano_layer.b['o'].set_value(b_theano[:, :, 2*H:3*H])
	lstm_theano_layer.b['c'].set_value(b_theano[:, :, 3*H:])

	h0_theano = h0.reshape(1, N, H)
	h0_symb   = theano.tensor.ftensor3("h_symb")
	lstm_theano_layer.h_m1.set_value(h0_theano)

	c0_theano = np.zeros((1, N, H), dtype = theano.config.floatX)
	c0_symb   = theano.tensor.ftensor3("c_symb")
	lstm_theano_layer.c_m1.set_value(c0_theano)

	output = theano.function([input_data], lstm_theano_layer.output)

	x_theano = x.reshape(N, T, D, 1)

	result = output(x_theano)

	print(result)
	print('shape {}'.format(result.shape))
	print('difference {}'.format(rel_error(h_print, result)))

def lstm_attend_step_forward(x, prev_h, prev_c,prev_z, Wx, Wh, Wz, b, y_list, Zcontext, Hcontext, Va):
	"""
	forward pass for a single timestep of LSTM.
	the input data has dimension D, the hidden state has dimension H and we use a minibatch size of N

	Inputs:
	- x: Input data, shape (N, D)
	- prev_h: previous hidden state, of shape (N, H)
	- prev_c: previous hidden state, shape (N, H)
	- Wx: input2hidden weights, shape (D, 4H)
	- Wh: hidden2hidden weights, shape (H,4H)
	- b: biases, shape (4H)
	Returns a tuple of:
	- next_h: next hidden state, shape (N, H)
	- next_c: next cell state, shape (N, H)
	- cache
	"""
	next_h, next_c, cache = None, None, None

	H = prev_h.shape[1]
	a = np.dot(x, Wx) + np.dot(prev_h, Wh) + np.dot(prev_z, Wz) + b
	i = sigmoid(a[:, :H])
	f = sigmoid(a[:, H:2*H])
	o = sigmoid(a[:, 2*H: 3*H])
	g = np.tanh(a[:, 3*H: ])

	next_c = f*prev_c + i*g
	next_h = o*np.tanh(next_c)

	s = []
	for y in y_list:
		m_ti = np.dot(next_h, Hcontext) + np.dot(y, Zcontext)
		e_ti = np.dot(m_ti, Va)
		# print(e_ti.shape)
		s.append(e_ti)

	s_np = np.asarray(s).astype("float32")
	print('s_np shape: ')
	print(s_np.shape)
	softmax_s = softmax(s_np,axisdim = 0)
	print(softmax_s.sum(axis = 0))

	#compute z_t
	y_weighted_list = []
	for idx, y in enumerate(y_list):
	    diag_mat = np.diag(softmax_s[idx, :])
	    weighted_y = np.dot(diag_mat, y)
	    y_weighted_list.append(weighted_y)

	# next_z = y_weighted_list[0]
	# for y_weighted in y_weighted_list:
	#     next_z += y_weighted
	# next_z -= y_weighted_list[0]

	next_z = sum(y_weighted_list)

	cache  = (x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h)
	return next_h, next_c, next_z#, cache, next_z

def lstm_attend_forward(x, h0, Wx, Wh,Wz, b, y_list, Zcontext, Hcontext, Va):
	"""
	Inputs:
	- x: Input data of shape (N,T, D)
	- h0: Initial hidden state of shape (N, H) - H - number of hidden node in the network
	- Wx: Weights for input2hidden connections, shape (D, 4H)
	- Wy: Weights for hidden2hidden connections, shpae (H, 4H)
	- b : biases of shape (4H,)
	Returns:
	- h : hidden states for all sequences, shape (N,T, H)
	- cache: values needed for backward 
	"""

	h, cache = None, None
	N, T, D = x.shape
	H = h0.shape[1] #H
	context_dim = y_list[0].shape[1]

	xs, hs, cs, zs = {}, {}, {}, {}
	hs[-1] = h0
	cs[-1] = np.zeros((N,H))
	zs[-1] = np.zeros((N,context_dim))
	h = np.zeros((N,T,H))


	for i in range(T):
		xs[i] = x[:, i, :]
		hs[i], cs[i], zs[i] = lstm_attend_step_forward(xs[i], hs[i-1], cs[i-1], zs[i-1], Wx, Wh, Wz, b, y_list, Zcontext, Hcontext, Va)
		h[:, i, :] = hs[i]

	cache = (x, h0, Wx, Wh, hs, cs, zs)

	return (h, cache)

def UnitTestLSTM_Attend():
	N = 2 #number of sample
	D = 5 #dimension of input
	H = 4 #dimension of hidden
	T = 3 #length of per each sample
	context_dim = 3
	K = 5

	x = np.linspace(-0.4, 0.6, num=N*T*D, dtype = theano.config.floatX).reshape(N, T, D)
	h0= np.linspace(-0.4, 0.8, num=N*H, dtype = theano.config.floatX).reshape(N, H)
	Wx= np.linspace(-0.2, 0.9, num=4*D*H, dtype = theano.config.floatX).reshape(D, 4*H)
	Wh= np.linspace(-0.3,0.6, num =4*H*H, dtype = theano.config.floatX).reshape(H,4*H)
	b = np.linspace(0.0, 0.0, num = 4*H, dtype = theano.config.floatX)
	Wz= np.linspace(-0.3, 0.6, num=4*H*context_dim, dtype = theano.config.floatX).reshape(context_dim, 4*H)
	Hcontext = np.linspace(-0.2, 0.6, num=H*K, dtype = theano.config.floatX).reshape(H, K)
	Zcontext = np.linspace(-0.2, 0.5, num=context_dim*K, dtype= theano.config.floatX).reshape(context_dim, K)
	Va= np.linspace(0.1, 0.4, num=K, dtype = theano.config.floatX)

	image_feature_3D = np.linspace(-0.2, 0.5, num=10*N*context_dim, dtype = theano.config.floatX).reshape(N,10, context_dim)

	y_list = []
	# for i in range(10):
	# 	temp_y = np.linspace(-0.2, 0.5, num=N*context_dim, dtype = theano.config.floatX).reshape(N, context_dim)
	# 	y_list.append(temp_y)
	for i in range(image_feature_3D.shape[1]):
		y_list.append(image_feature_3D[:, i, :].reshape(N, context_dim))


	h, cache =lstm_attend_forward(x, h0, Wx, Wh,Wz, b,y_list, Zcontext, Hcontext, Va)

	h_print = h.reshape(N, T, H, 1)
	print(h_print)
	print('shape {}'.format(h_print.shape))

	#LSTM by Theano
	initial_h0_layer_out = theano.tensor.tensor3(name = 'h0_initial', dtype = theano.config.floatX)
	initial_c0_layer_out = theano.tensor.tensor3(name = 'c0_initial', dtype = theano.config.floatX)
	weight_y_in = theano.tensor.fmatrix("weight_y")

	h0_theano = h0.reshape(1, N, H)
	# h0_symb   = theano.tensor.ftensor3("h_symb")
	# lstm_theano_layer.h_m1.set_value(h0_theano)
	pdb.set_trace()

	c0_theano = np.zeros((1, N, H), dtype = theano.config.floatX)
	# c0_symb   = theano.tensor.ftensor3("c_symb")
	# lstm_theano_layer.c_m1.set_value(c0_theano)
	pdb.set_trace()

	z0_theano = np.zeros((1, N, context_dim), dtype = theano.config.floatX)
	pdb.set_trace()

	x_theano = x.reshape(N, T, D, 1)
	image_feature_input = image_feature_3D
	pdb.set_trace()

	input_data = theano.tensor.tensor4(name='x', dtype=theano.config.floatX)
	net = ShowTellNet(test_input_image=input_data)
	net.layer_opts['num_lstm_node']=H
	input_data = theano.tensor.tensor4(name ='x', dtype=theano.config.floatX)
	image_feature_region = theano.tensor.tensor3(name = 'feature_region', dtype = theano.config.floatX)
	net_lstm = ShowTellNet(test_input_image=input_data)
	net_lstm.layer_opts['num_lstm_node'] = H
	net_lstm.layer_opts['context_dim']   = K
	net.layer_opts['num_dimension_feature'] = context_dim
	net.layer_opts['num_region'] = 10
	
	weight_y_in_value = np.zeros(( net.layer_opts['num_region'], net.layer_opts['num_dimension_feature']) , dtype= theano.config.floatX)
	pdb.set_trace()

	lstm_theano_layer = LSTM_Attend(net_lstm, 
		net_lstm.content['input_img'], 
		(N, T-1, D, 1), image_feature_region, 
		initial_h0 = initial_h0_layer_out, initial_c0 = initial_c0_layer_out, weight_y = weight_y_in)


	lstm_theano_layer.W['i'].set_value(Wx[:, :H])
	lstm_theano_layer.W['f'].set_value(Wx[:, H:2*H])
	lstm_theano_layer.W['o'].set_value(Wx[:, 2*H:3*H])
	lstm_theano_layer.W['c'].set_value(Wx[:, 3*H:])

	lstm_theano_layer.U['i'].set_value(Wh[:, :H])
	lstm_theano_layer.U['f'].set_value(Wh[:, H:2*H])
	lstm_theano_layer.U['o'].set_value(Wh[:, 2*H:3*H])
	lstm_theano_layer.U['c'].set_value(Wh[:, 3*H:])

	lstm_theano_layer.Z['i'].set_value(Wz[:, :H])
	lstm_theano_layer.Z['f'].set_value(Wz[:, H:2*H])
	lstm_theano_layer.Z['o'].set_value(Wz[:, 2*H:3*H])
	lstm_theano_layer.Z['c'].set_value(Wz[:, 3*H:])

	lstm_theano_layer.Wcontext.set_value(Zcontext)
	lstm_theano_layer.Hcontext.set_value(Hcontext)
	Va_reshape = Va.reshape(K,1)
	lstm_theano_layer.Va.set_value(Va_reshape)

	b_theano= b.reshape(1, 1, 4*H)
	lstm_theano_layer.b['i'].set_value(b_theano[:, :, :H])
	lstm_theano_layer.b['f'].set_value(b_theano[:, :, H:2*H])
	lstm_theano_layer.b['o'].set_value(b_theano[:, :, 2*H:3*H])
	lstm_theano_layer.b['c'].set_value(b_theano[:, :, 3*H:])

	lstm_theano_layer.z_m1.set_value(z0_theano)

	output = theano.function([input_data, image_feature_region, initial_h0_layer_out, initial_c0_layer_out, weight_y_in], lstm_theano_layer.output)

	result = output(x_theano, image_feature_input, h0_theano, c0_theano, weight_y_in_value)
	pdb.set_trace()

	print(result)
	print('shape {}'.format(result.shape))
	print('difference {}'.format(rel_error(h_print, result)))
	# print(lstm_theano_layer.output.eval({input_data: x_theano, image_feature_region: image_feature_input}))

	#Theano onestep
	input_basic = theano.tensor.fmatrix('input_basic')

	def step(x, htm1,ztm1):
		print(htm1.type)
		return x+ htm1, ztm1
	h_symb = theano.tensor.fmatrix("out_basic")

	h_out,	_ = theano.scan(fn= step, 
							sequences = input_basic,
							outputs_info= [h_symb, np.array([0., 0., 0., 0., 0.])])

	f_basic = theano.function([input_basic, h_symb], h_out)
	h_t_in = np.array([[0., 0., 0., 0., 0.]]).astype("float32")
	input_in = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).astype("float32")
	print(h_t_in.shape)
	print(input_in.shape)
	out = f_basic(input_in, h_t_in)

	print(out)
	if type(out) is list:
		print(out[0].shape)
		print(out[1].shape)
		print(type(out[0]))
	else:
		print(out.shape)


def BasicTheano():

	#REPEAT vs TILE 
	x = theano.tensor.fmatrix("x")
	z = theano.tensor.repeat(x, 1, axis = 0)
	z_one_more = theano.tensor.repeat(z, 2, axis = 1)

	foo = theano.function([x], z)
	foo_one_more = theano.function([z], z_one_more)

	a = np.array([[1, 2, 3]]).astype("float32")
	print('a.shape: ')
	print(a.shape)

	c = foo(a)
	c_one_more = foo_one_more(c)

	print("applying repeat along axis 0")
	print(c)
	print(c.shape)
	print("applying one more along axis 1")
	print(c_one_more)
	print(c_one_more.shape)

	z_tile = theano.tensor.tile(x, (3,2))
	
	foo_tile = theano.function([x], z_tile)

	c_tile = foo_tile(a)
	print("applying tile along axis 0")
	print(c_tile)


	#TRANSPOSE vs RESHAPE vs DIMSHUFFLE
	v = theano.tensor.ivector("v")
	u = theano.tensor.ivector("u")
	u_dot_v = theano.tensor.dot(u, theano.tensor.transpose(v))
	v_trans = theano.tensor.transpose(v)

	u_dot_v_no_transpose = theano.tensor.dot(u, v)

	foo_dot = theano.function([u, v], u_dot_v)
	foo_trans = theano.function([v], v_trans)
	foo_dot_no_transpose = theano.function([u, v], u_dot_v_no_transpose)

	v_value = np.array([1, 2, 3]).astype("int32")
	u_value = np.array([1, 2, 3]).astype("int32")

	foo_dot_value = foo_dot(u_value, v_value)
	foo_trans_value = foo_trans(v_value)
	foo_dot_no_transpose_value = foo_dot_no_transpose(u_value, v_value)

	print('dot product')
	print(foo_dot_value)

	print('dot product no transpose: ')
	print(foo_dot_no_transpose_value)

	print('transpose: ')
	print(foo_trans_value)
	print(foo_trans_value.shape)

	print('original shape')
	print(v_value.shape)

	print('v reshape')
	v_reshape = v.reshape((v.shape[0], 1))
	print(v_reshape.type)

	foo_reshape = theano.function([v], v_reshape)
	foo_reshape_value = foo_reshape(v_value)
	print(foo_reshape_value.shape)

	#SUM
	v_sum_0 = v.sum(axis = 0)
	foo_sum_0 = theano.function([v], v_sum_0)

	foo_sum_0_value = foo_sum_0(v_value)
	print(foo_sum_0_value)

	#v_sum_1 = v_reshape.sum(axis = 1)
	#foo_sum_1 = theano.function([v], v_sum_1)
	#foo_sum_1_value = foo_sum_0(v_value)
	#print(foo_sum_1_value)

	#test reshape
	y = theano.tensor.ftensor3("y")
	y_shape = y.shape
	y_reshape = y.reshape((y_shape[1], y_shape[2]))#, y_shape[0]

	function_reshape = theano.function([y], y_reshape)

	y_value = np.ones((1,3,2)).astype("float32")
	print(y_value.shape)

	y_reshape_value = function_reshape(y_value)
	print('y_reshape:')
	print(y_reshape_value)

	#reshape matrix to tensor3
	matrix = theano.tensor.fmatrix("matrix")
	print(matrix.type)
	mat_shape = matrix.shape
	mat_reshape = matrix.reshape((-1, mat_shape[0], mat_shape[1]))
	print(mat_reshape.type)

	mat_reshape_func= theano.function([matrix], mat_reshape)

	mat_value = np.ones((3,2)).astype("float32")
	mat_reshape_func_out = mat_reshape_func(mat_value)

	print(mat_value.shape)
	print("matrix to 3D tensor")
	print(mat_reshape_func_out.shape)
	print(mat_reshape_func_out)

	#creating a square matrix with the given vector as its diagonal
	given_vec = theano.tensor.fvector("given_vec")
	diag_mat = theano.tensor.nlinalg.AllocDiag()(given_vec)

	diag_function = theano.function([given_vec], diag_mat)

	given_vec_value = np.array([1, 2, 3]).astype("float32")
	diag_function_value = diag_function(given_vec_value)
	print("diagonal matrix is: ")
	print(diag_function_value)
	print(diag_function_value.shape)

	#multiply an element of vector (1*N) with a row/column of a matrix (N*D*1)
	multiply_vector_matrix = T.dot(diag_mat, y_reshape)

	result_function = theano.function([diag_mat, y_reshape], multiply_vector_matrix)
	output_value = result_function(diag_function_value, y_reshape_value)
	print(output_value.shape)
	print(output_value)
	
	#Reshape to convert tensor from matrix to vector/column/row/3D
	print("matrix to vector/row/column/3D")
	matrix_origin = theano.tensor.fmatrix('Mat')
	mat_2_vector  = matrix_origin.reshape((matrix_origin.shape[0]*matrix_origin.shape[1], ))   
	print(mat_2_vector.type)
	mat_2_row  = matrix_origin.reshape((1, matrix_origin.shape[0]*matrix_origin.shape[1]))   
	print(mat_2_row.type)
	mat_2_column     = matrix_origin.reshape((matrix_origin.shape[0]*matrix_origin.shape[1], 1))  
	print(mat_2_column.type)
	mat_2_3dtensor= matrix_origin.reshape((-1, matrix_origin.shape[0], matrix_origin.shape[1])) 
	print(mat_2_3dtensor.type)

	f = theano.function([matrix_origin], [mat_2_vector, mat_2_column, mat_2_row, mat_2_3dtensor])

	input_value = np.array([[1.,2.], [3.,4.]]).astype("float32")
	print(input_value.shape)

	for output in f(np.array([[1., 2.], [3., 4.]]).astype("float32")):
		
		print(output.shape)
		print(output)

	#REPEAT for 3D tensor
	print("repeat 3D tensor")
	h_t = theano.tensor.tensor3("h_t")
	axis_scalar = theano.tensor.dscalar("axis")
	h_t_repeat = theano.tensor.repeat(h_t, 3,axis= 0)

	repeat_func = theano.function([h_t], h_t_repeat)
	input_value = np.ones((1,3,2)).astype("float32")
	input_value[0, 1, 1] = 5.
	input_value[0, 2, 1] = 3.
	repeat_func_out = repeat_func(input_value)
	print(repeat_func_out.shape)
	print("input value:")
	print(input_value)
	print("element in output: ")
	print(repeat_func_out[0, :, :])
	print("out: ")
	print(repeat_func_out)

	#test (3,1, 2) -> (3, 3, 2)
	h_t_repeat_1 = theano.tensor.repeat(h_t, 3,axis= 1)

	repeat_func_1 = theano.function([h_t], h_t_repeat_1)
	print("repeat (312) to (332)")
	input_values_312 = np.ones((3,1, 2)).astype("float32")
	input_values_312[2, 0, 0] = 9.
	repeat_func_out_332 = repeat_func_1(input_values_312)
	print(repeat_func_out_332.shape)
	print("input value")
	print(input_values_312)
	print("out value")
	print(repeat_func_out_332)


	# print("repeat 2 times: ")
	# #Repeat 3D tensor 2 times with 2 different axes
	# b_value = np.ones((1, 1 ,5)).astype("float32")
	# b_value[0, 0, 2] = 3.
	# b_value[0, 0, 4] = 5.
	# print("1st time: ")
	# print(repeat_func(b_value))
	# print(repeat_func(b_value).shape)

	# h_t_repeat_2x = theano.tensor.repeat(h_t_repeat, 2, axis = 1)
	# repeat_func_2x = theano.function([h_t_repeat], h_t_repeat_2x)
	# print("2nd time")
	# print(repeat_func_2x(repeat_func(b_value)))
	# print(repeat_func_2x(repeat_func(b_value)).shape)

	#Theano tensor concatenate
	z_t = theano.tensor.tensor3("z_t")
	concat = theano.tensor.concatenate([h_t, z_t], axis = 0)

	concat_func = theano.function([h_t, z_t], concat)
	z_t_input = np.ones((3,2,1)).astype("float32")
	h_t_input = np.ones((3,2,1)).astype("float32")
	print("concat : ")
	print(concat_func(h_t_input, z_t_input))
	print(concat_func(h_t_input, z_t_input).shape)

	#T.arange, T.mean, T.log, T.neq
	print("T.arange function:")
	mat_y = theano.tensor.fmatrix("mat_y")
	colum_vector = mat_y[theano.tensor.arange(mat_y.shape[0]), :]

	t_arange_function = theano.function([mat_y], colum_vector)

	mat_y_value = np.random.randn(3,2).astype("float32")
	t_arange_out = t_arange_function(mat_y_value)
	print('input value:')
	print(mat_y_value)
	print('output value:')
	print(t_arange_out.shape)
	print(t_arange_out)


	#NUMPY example  A(N, M, K)  B(N, M)  -> C(N, M) = A[arange(N), arange(M), B] using B as indexing matrix
	A = np.arange(4*2*5).reshape(4,2,5)
	B = np.arange(4*2).reshape(4,2)%5

	# print('arange: ')
	# print(np.arange(A.shape[0])[:, np.newaxis])
	# print(np.arange(A.shape[1]))

	C = A[np.arange(A.shape[0])[:, np.newaxis], np.arange(A.shape[1]), B] #
	print(A)
	print(B)
	print(C)
	print(C.shape)

	#Theano tensor slicing and assigning
	print("slicing theano")
	x_vector = theano.tensor.vector()
	y_slicing = x_vector[0::2]
	print(y_slicing.eval({x_vector: np.array([1,2, 3, 4]).astype("float32")}))

	#Theano split----------------------------------------
	# print("split theano")
	# def split_half(x, axis = 0):
	# 	if theano.tensor.le(x.shape[axis], 1):
	# 		return x
	# 	size1 = x.shape[axis]/2
	# 	size2 = x.shape[axis] - size1
	# 	split_out = theano.tensor.split(x, [size1, size2], 2, axis = axis)
	# 	first_part= split_out[0]
	# 	second_part = split_out[1]
	# 	return (split_half(first_part), split_half(second_part))
	
	# def split_6_along_axis(x, axis = 0):
	# 	size = []
	# 	for i in range(6):
	# 		size.append(1)
	# 	return theano.tensor.split(x, size, 6, axis = axis)

	# split_x = theano.tensor.matrix("split_x")
	# axis_split = theano.tensor.lscalar()
	# split_y_first, split_y_second = split_half(split_x, axis= axis_split)
	# f_split = theano.function([split_x, axis_split], split_y_first, split_y_second)
	# print(f_split(np.arange(12).reshape(6, 2).astype("float32"), 0))
	# # print(split_y.eval({split_x: np.arange(12).reshape(6, 2).astype("float32"), axis_split: 0}))


	# split_y_individual = split_6_along_axis(split_x, axis = axis_split)
	# f_split_individual = theano.function([split_x, axis_split], split_y_individual)
	# print(f_split_individual(np.arange(12).reshape(6, 2).astype("float32"), 0))

	#T.dot between two 3D tensors-------------------------
	tensor_1 = theano.tensor.tensor3("tensor_1")
	tensor_2 = theano.tensor.tensor3("tensor_2")

	dot_2_tensors = theano.tensor.dot(tensor_1, tensor_2)
	dot_2_tensor_func = theano.function([tensor_1, tensor_2], dot_2_tensors)

	tensor_1_in = np.ones((3,2,2)).astype("float32")
	tensor_2_in = np.ones((2,2,3)).astype("float32") #2,1,3 -wrong

	out_dot_2_tensors = dot_2_tensor_func(tensor_1_in, tensor_2_in)

	print("dot between two 3D tensors")
	print(out_dot_2_tensors.shape)
	# print(out_dot_2_tensors)

	#Theano tensor identity_like
	print('tensor identity like')
	identity_3D = T.identity_like(tensor_1)

	identity_out = identity_3D.eval({tensor_1: tensor_1_in})
	print(identity_out.shape)
	print(identity_out)
	print(tensor_1_in.shape)
	print(tensor_1_in)

	#T.repeat itself
	# bi = T.tensor3("bi")

	# bi = T.repeat(bi, 3, axis = 0)
	# out = bi.eval({bi: np.ones((1,3,2)).astype("float32")})
	# print(out.shape)

	# out_func = theano.function([bi], bi)
	# print(out_func(np.ones((1,3,2)).astype("float32")).shape)

	#i_t = i_t + a_t - Add itself --------------------------
	print("adding itself")
	a_t = T.fmatrix("a_t")
	h_t = T.fmatrix("h_t")

	i_t = h_t + a_t
	i_t = i_t + a_t

	# function_itself = i_t.eval({i_t: np.ones((2,2)).astype("float32"), a_t: np.ones((2,2)).astype("float32")})
	function_itself = theano.function([h_t, a_t], i_t)
	function_itself_out = function_itself(np.zeros((2,2)).astype("float32"), np.ones((2,2)).astype("float32"))
	print(function_itself_out)
	# ------------------------------------------------------

	#shared variable repeat - It works
	shared_var = theano.shared(name = "shared",
		value = np.ones((1,3, 2)).astype("float32"),
		borrow = True)

	shared_var = T.repeat(shared_var, 3, axis = 0)

	shared_var_reshape_out = shared_var.eval()
	print(shared_var_reshape_out.shape)

	#Test Max, Min, and along axis
	print("Test max, min")
	value_mat = np.asarray([[1.0, 2.0],[3.0, 4.0]]).astype("float32")
	test_tensor = theano.tensor.fmatrix("tensor")
	c = test_tensor.min()
	function_max = theano.function([test_tensor], c)
	out = function_max(value_mat)
	print(out)
	c_along = test_tensor.min(axis = 1)
	function_max_along = theano.function([test_tensor], c_along)
	out = function_max_along(value_mat)
	print(out)

	#rescale 3D tensor values to range [0, 1]
	print("scaling value of a tensor to range [0, 1]")
	def rescale_step(input_tensor):
		min_value = input_tensor.min()
		max_value = input_tensor.max()
		out_rescale = (input_tensor - min_value)/(max_value- min_value)
		return out_rescale

	input_rescale = theano.tensor.tensor3("in_rescale", dtype = theano.config.floatX)
	output_rescale, updates = theano.scan(fn=rescale_step,
	                                   outputs_info=[],
	                                   sequences=[input_rescale],
	                                   non_sequences=[])

	rescale_func = theano.function([input_rescale], output_rescale)

	input_rescale_value = np.linspace(1, 30, num = 2*5*3, dtype = theano.config.floatX).reshape(2, 5, 3)
	out_rescale = rescale_func(input_rescale_value)
	print(out_rescale)
	print(out_rescale.shape)
	print("input value")
	print(input_rescale_value)



def UnitTest_OnestepAttend():
	N = 2 #number of sample
	D = 5 #dimension of input
	H = 4 #dimension of hidden
	T_new = 1 #length of per each sample
	context_dim = 3
	K = 5

	x = np.linspace(-0.4, 0.6, num=N*T_new*D, dtype = theano.config.floatX).reshape(T_new, N, D)
	h0= np.linspace(-0.4, 0.8, num=N*H, dtype = theano.config.floatX).reshape(N, H)
	Wx= np.linspace(-0.2, 0.9, num=4*D*H, dtype = theano.config.floatX).reshape(D, 4*H)
	Wh= np.linspace(-0.3,0.6, num =4*H*H, dtype = theano.config.floatX).reshape(H,4*H)
	b = np.linspace(0.0, 0.0, num = 4*H, dtype = theano.config.floatX)
	Wz= np.linspace(-0.3, 0.6, num=4*H*context_dim, dtype = theano.config.floatX).reshape(context_dim, 4*H)
	Hcontext = np.linspace(-0.2, 0.6, num=H*K, dtype = theano.config.floatX).reshape(H, K)
	Zcontext = np.linspace(-0.2, 0.5, num=context_dim*K, dtype= theano.config.floatX).reshape(context_dim, K)
	Va= np.linspace(0.1, 0.4, num=K, dtype = theano.config.floatX)
	Va_reshape = Va.reshape(K,1)

	image_feature_3D = np.linspace(-0.2, 0.5, num=10*N*context_dim, dtype = theano.config.floatX).reshape(N,10, context_dim)

	h0_theano = h0.reshape(1, N, H)
	# h0_symb   = theano.tensor.ftensor3("h_symb")
	# lstm_theano_layer.h_m1.set_value(h0_theano)

	c0_theano = np.zeros((1, N, H), dtype = theano.config.floatX)
	# c0_symb   = theano.tensor.ftensor3("c_symb")
	# lstm_theano_layer.c_m1.set_value(c0_theano)

	z0_theano = np.zeros((1, N, context_dim), dtype = theano.config.floatX)

	x_theano = x.reshape(T_new, N, D, 1)
	image_feature_input = image_feature_3D

	weight_y_in_value = np.zeros(( 10, context_dim) , dtype= theano.config.floatX)
	b_theano= b.reshape(1, 1, 4*H)
	pdb.set_trace()

	#symbolic variables
	initial_h0_layer_out = theano.tensor.tensor3(name = 'h0_initial', dtype = theano.config.floatX)
	initial_c0_layer_out = theano.tensor.tensor3(name = 'c0_initial', dtype = theano.config.floatX)
	initial_z0			 = T.tensor3(name= 'z0_initial', dtype = theano.config.floatX)
	weight_y_in = theano.tensor.fmatrix("weight_y")	
	input_data = theano.tensor.tensor3(name ='x', dtype=theano.config.floatX)
	image_feature_region = theano.tensor.tensor3(name = 'feature_region', dtype = theano.config.floatX)

	Wi_sym, Wf_sym, Wc_sym, Wo_sym, Ui_sym, Uf_sym, Uc_sym, Uo_sym, Zi_sym, Zf_sym, Zc_sym, Zo_sym = T.fmatrices(12)
	Zcontext_sym, Hcontext_sym = T.fmatrices(2)
	bi  = T.ftensor3("bi")
	bf  = T.ftensor3("bf")
	bc  = T.ftensor3("bc")
	bo  = T.ftensor3("bo")
	Va_sym = T.fcol("Va")


	out_sym = onestep_attend_tell(input_data, initial_h0_layer_out, initial_c0_layer_out, initial_z0, 
		Wi_sym, Wf_sym, Wc_sym, Wo_sym, Ui_sym, Uf_sym, Uc_sym, Uo_sym, Zi_sym, Zf_sym, Zc_sym, Zo_sym,
		Zcontext_sym, Hcontext_sym, Va_sym,
		bi, bf, bc, bo, image_feature_region, weight_y_in)

	onestep_func = theano.function([input_data, initial_h0_layer_out, initial_c0_layer_out, initial_z0, 
		Wi_sym, Wf_sym, Wc_sym, Wo_sym, Ui_sym, Uf_sym, Uc_sym, Uo_sym, Zi_sym, Zf_sym, Zc_sym, Zo_sym,
		Zcontext_sym, Hcontext_sym, Va_sym,
		bi, bf, bc, bo, image_feature_region, weight_y_in], out_sym)

	list_output = onestep_func(x, h0_theano, c0_theano, z0_theano,
		Wx[:, :H], Wx[:, H:2*H], Wx[:, 2*H:3*H], Wx[:, 3*H:],
		Wh[:, :H], Wh[:, H:2*H], Wh[:, 2*H:3*H], Wh[:, 3*H:],
		Wz[:, :H], Wz[:, H:2*H], Wz[:, 2*H:3*H], Wz[:, 3*H:],
		Zcontext,Hcontext,
		Va_reshape,
		b_theano[:,: , :H], b_theano[:, :, H:2*H], b_theano[:, :, 2*H:3*H], b_theano[:, :, 3*H:], 
		image_feature_input, weight_y_in_value)


	pdb.set_trace()

	print(list_output[0].shape)
	print(list_output[1].shape)
	print(list_output[2].shape)

	pdb.set_trace()

def onestep_attend_tell(x_t, pre_h, pre_c, pre_z, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, Zi, Zf, Zc, Zo, Zcontext, Hcontext, Va, bi, bf, bc, bo, image_feature_region, weight_y):
	#-------------------------------------------------
	# pre_h = T.tensor3(name = 'h0_initial', dtype = theano.config.floatX)
	# x_t   = T.tensor3(name ='x', dtype=theano.config.floatX)
	# pre_z = T.tensor3(name= 'z0_initial', dtype = theano.config.floatX)
	# Wi, Ui, Zi   =  T.fmatrices(3)
	# bi    = T.ftensor3("bi")
	#-------------------------------------------------

	i_t = T.dot(x_t, Wi) + T.dot(pre_h, Ui) + T.dot(pre_z, Zi)
	i_t_shape = T.shape(i_t)

	#------------------------------------------------------------------
	# i_t_test = i_t.eval({x_t: x_theano, pre_h: h0_theano, pre_z: z0_theano, Wi: Wx[:, :H], Ui: Wh[:, :H], Zi: Wz[:, :H]})
	# print(i_t_test.shape)
	# pdb.set_trace()    
	#------------------------------------------------------------------

	bi_reshape = T.repeat(bi, i_t_shape[0], 0)
	bi_reshape_2x = T.repeat(bi_reshape, i_t_shape[1], 1)

	# -----------------------------------------------------------------
	# bi_test = bi_reshape_2x.eval({bi: b_theano[:,:,:H], i_t: i_t.eval({i_t: np.zeros((1,2,4)).astype("float32")})})
	# print(bi_test.shape)
	# pdb.set_trace()
	# ------------------------------------------------------------------

	bf_reshape = T.repeat(bf, i_t_shape[0], 0)
	bf_reshape_2x = T.repeat(bf_reshape, i_t_shape[1], 1)

	bc_reshape = T.repeat(bc, i_t_shape[0], 0)
	bc_reshape_2x = T.repeat(bc_reshape, i_t_shape[1], 1)

	bo_reshape = T.repeat(bo, i_t_shape[0], 0)
	bo_reshape_2x = T.repeat(bo_reshape, i_t_shape[1], 1)

	i_t_new= sigmoid(i_t + bi_reshape_2x)
	

	# ------------------------------------------------------------------
	# i_t_new_eval = i_t_new.eval({i_t: np.zeros((1,2,4)).astype("float32"), bi: b_theano[:, : , :H]})
	# print(i_t_new_eval.shape)
	# pdb.set_trace()
	# --------------------------------------------------------------------
	

	f_t= sigmoid(T.dot(x_t, Wf) + T.dot(pre_h, Uf) + T.dot(pre_z, Zf) + bf_reshape_2x)
	# --------------------------------------------------------------------
	# f_t_eval = f_t.eval({x_t:x_theano, pre_h: h0_theano, pre_z: z0_theano,
	# 	Wf: Wx[:, H:2*H],
	# 	Uf: Wh[:, H:2*H],
	# 	Zf: Wz[:, H:2*H],
	# 	bf: b_theano[:, :, H:2*H],
	# 	i_t: np.zeros((1,2,4)).astype("float32")})

	# print(f_t_eval.shape)
	# pdb.set_trace()
	# --------------------------------------------------------------------

	o_t= sigmoid(T.dot(x_t, Wo) + T.dot(pre_h, Uo) + T.dot(pre_z, Zo) + bo_reshape_2x)

	c_th = tanh(T.dot(x_t, Wc)  + T.dot(pre_h, Uc) + T.dot(pre_z, Zc) + bc_reshape_2x)

	c_t = f_t*pre_c + i_t_new*c_th

	h_t = o_t*T.tanh(c_t) #shape (1, N, h_dim)

	# ------------------------------------------------------------------
	# ht_test = h_t.eval({x_t:x_theano, pre_h: h0_theano, pre_c: c0_theano, pre_z: z0_theano, 
	# 	Wi: Wx[:, :H], Wf: Wx[:, H:2*H], Wo: Wx[:, 2*H:3*H], Wc: Wx[:, 3*H:],
	# 	Ui: Wh[:, :H], Uf: Wh[:, H:2*H], Uo: Wh[:, 2*H:3*H], Uc: Wh[:, 3*H:],
	# 	Zi: Wz[:, :H], Zf: Wz[:, H:2*H], Zo: Wz[:, 2*H:3*H], Zc: Wz[:, 3*H:], 
	# 	bi: b_theano[:,:,:H], bf: b_theano[:, :, H:2*H], bo: b_theano[ :, :, 2*H:3*H], bc: b_theano[:,:, 3*H:]})
	# print(ht_test.shape)
	# pdb.set_trace()
	# ------------------------------------------------------------------

	h_t_context = T.repeat(h_t, image_feature_region.shape[1], axis = 0) #new shape (No_region, N, h_dim)
	image_feature_reshape = T.transpose(image_feature_region, (1, 0, 2))
	#compute non-linear correlation between h_t(current text) to image_feature_region (64 for 128*128 and 196 for 224*224)
	# pdb.set_trace()
	m_t = T.tanh(T.dot(h_t_context, Hcontext) + T.dot(image_feature_reshape, Zcontext)) #shape (No_region, N, context_dim)	

	# ------------------------------------------------------------------
	# N = 2 #number of sample
	# D = 5 #dimension of input
	# H = 4 #dimension of hidden
	# T_new = 1 #length of per each sample
	# context_dim = 3
	# K = 5

	# x = np.linspace(-0.4, 0.6, num=N*T_new*D, dtype = theano.config.floatX).reshape(T_new, N, D)
	# h0= np.linspace(-0.4, 0.8, num=N*H, dtype = theano.config.floatX).reshape(N, H)
	# Wx= np.linspace(-0.2, 0.9, num=4*D*H, dtype = theano.config.floatX).reshape(D, 4*H)
	# Wh= np.linspace(-0.3,0.6, num =4*H*H, dtype = theano.config.floatX).reshape(H,4*H)
	# b = np.linspace(0.0, 0.0, num = 4*H, dtype = theano.config.floatX)
	# Wz= np.linspace(-0.3, 0.6, num=4*H*context_dim, dtype = theano.config.floatX).reshape(context_dim, 4*H)
	# Hcontext_in = np.linspace(-0.2, 0.6, num=H*K, dtype = theano.config.floatX).reshape(H, K)
	# Zcontext_in = np.linspace(-0.2, 0.5, num=context_dim*K, dtype= theano.config.floatX).reshape(context_dim, K)
	# Va= np.linspace(0.1, 0.4, num=K, dtype = theano.config.floatX)
	# Va_reshape = Va.reshape(K,1)

	# image_feature_3D = np.linspace(-0.2, 0.5, num=10*N*context_dim, dtype = theano.config.floatX).reshape(N,10, context_dim)

	# h0_theano = h0.reshape(1, N, H)
	# # h0_symb   = theano.tensor.ftensor3("h_symb")
	# # lstm_theano_layer.h_m1.set_value(h0_theano)

	# c0_theano = np.zeros((1, N, H), dtype = theano.config.floatX)
	# # c0_symb   = theano.tensor.ftensor3("c_symb")
	# # lstm_theano_layer.c_m1.set_value(c0_theano)

	# z0_theano = np.zeros((1, N, context_dim), dtype = theano.config.floatX)

	# x_theano = x.reshape(T_new, N, D)
	# image_feature_input = image_feature_3D

	# weight_y_in_value = np.zeros(( 10, context_dim) , dtype= theano.config.floatX)
	# b_theano= b.reshape(1, 1, 4*H)

	# h_t_context_eval = m_t.eval({h_t: np.ones((1,2,4)).astype("float32"), image_feature_region: image_feature_input, Hcontext: Hcontext_in, Zcontext: Zcontext_in})
	# print(h_t_context_eval.shape)
	# pdb.set_trace()
	# ------------------------------------------------------------------

	
	e = T.dot(m_t, Va) #No_region, N, 1
	e_reshape = e.reshape((e.shape[0], T.prod(e.shape[1:])))
	
	# ------------------------------------------------------------------
	# Va_in= np.linspace(0.1, 0.4, num=5*1, dtype = theano.config.floatX).reshape(5,1)
	# Va_reshape = Va_in.reshape(5,1).astype("float32")
	# # print(Va_reshape)
	# e_val = e_reshape.eval({m_t: np.ones((10,2,5)).astype("float32"), Va: Va_reshape}) #np.ones((10,2,5)).astype("float32")
	# print(e_val.shape)
	# ------------------------------------------------------------------
	


	e_softmax = softmax_along_axis(e_reshape, axis = 0) #shape No_region, N
	
	# -------------------------------------------------------------------
	# pdb.set_trace()
	# e_softmax_eval = e_softmax.eval({e_reshape: np.random.randn(10,2).astype("float32")})
	# print(e_softmax_eval.shape)
	# -------------------------------------------------------------------

	e_t = T.transpose(e_softmax, (1,0)) #shape N, No_region
	e_t_r = e_t.reshape([-1, e_softmax.shape[0], e_softmax.shape[1]]) #3D tensor 1, N, No_region
	e_t_r_t = T.transpose(e_t_r, (1,0, 2)) # shape N, 1, No_region
	e_3D = T.repeat(e_t_r_t, e_t_r_t.shape[2], axis = 1) #shape N, No_region, No_region  image_feature_region.shape[1]
	e_3D_t = T.transpose(e_3D, (1,2,0)) #No_region, No_region, N

	# ---------------------------------------------------------------------
	# image_feature_3D = np.linspace(-0.2, 0.5, num=10*2*3, dtype = theano.config.floatX).reshape(2,10, 3)
	# e_3D_t_eval = e_3D_t.eval({e_softmax: np.random.randn(10,2).astype("float32")})
	# print(e_3D_t_eval.shape)
	# pdb.set_trace()
	# ---------------------------------------------------------------------

	identity_2D = T.identity_like(e_3D_t)# shape No_region, No_region
	identity_3D = identity_2D.reshape([-1, identity_2D.shape[0], identity_2D.shape[1]]) # shape 1, No_region, No_region
	identity_3D_t = T.repeat(identity_3D,  image_feature_region.shape[0], axis = 0)
	e_3D_diagonal = e_3D*identity_3D_t #diagonal tensor 3D  (N, No_region, No_region)

	# ----------------------------------------------------------------------
	# image_feature_3D = np.linspace(-0.2, 0.5, num=10*2*3, dtype = theano.config.floatX).reshape(2,10, 3)
	
	# e_3D_diagonal_eval = e_3D_diagonal.eval({e_3D_t: np.ones((10, 10, 2)).astype("float32"), 
	# 	image_feature_region: image_feature_3D, 
	# 	e_3D: np.ones((2, 10, 10)).astype("float32")})
	
	# print(e_3D_diagonal_eval)
	# pdb.set_trace()
	# ----------------------------------------------------------------------

	# weight_y = T.fmatrix("weight_y")

	out_weight_y, updates = theano.scan(fn=onestep_weight_feature_multiply,
	                                   outputs_info=[weight_y],
	                                   sequences=[e_3D_diagonal, image_feature_region],
	                                   non_sequences=[])

	#out_weight_y shape (N, No_region, feature_dim)
	z_t = T.sum(out_weight_y, axis = 1) #shape (N, feature_dim)

	z_t_r = z_t.reshape((-1,z_t.shape[0],z_t.shape[1]))

    #------------------------------------------------------------------------ 
	pdb.set_trace()
	image_feature_3D = np.linspace(-0.2, 0.5, num=10*2*3, dtype = theano.config.floatX).reshape(2,10, 3)
	z_t_r_eval = z_t_r.eval({e_3D_diagonal: np.ones((2,10,10)).astype("float32"), 
		image_feature_region: image_feature_3D, 
		weight_y: np.zeros((10,3)).astype("float32")})
	
	print(z_t_r_eval.shape)

	pdb.set_trace()
	# -----------------------------------------------------------------------

	return [h_t, c_t, z_t_r]




def onestep_weight_feature_multiply(e, feature, out):
    # e_mat = e.reshape((e.shape[1], e.shape[2]))
    # feature_mat = feature.reshape((feature.shape[1], feature.shape[2]))
    out = T.dot(e, feature)
    return out

def onestep_attend_copy():

	i_t = T.dot(x_t, Wi) + T.dot(pre_h, Ui) + T.dot(pre_z, Zi)
	i_t_shape = T.shape(i_t)

	bi_reshape = T.repeat(bi, i_t_shape[0], 0)
	bi_reshape_2x = T.repeat(bi_reshape, i_t_shape[1], 1)

	bf_reshape = T.repeat(bf, i_t_shape[0], 0)
	bf_reshape_2x = T.repeat(bf_reshape, i_t_shape[1], 1)

	bc_reshape = T.repeat(bc, i_t_shape[0], 0)
	bc_reshape_2x = T.repeat(bc_reshape, i_t_shape[1], 1)

	bo_reshape = T.repeat(bo, i_t_shape[0], 0)
	bo_reshape_2x = T.repeat(bo_reshape, i_t_shape[1], 1)

	i_t_new= sigmoid(i_t + bi_reshape_2x)
	f_t= sigmoid(T.dot(x_t, Wf) + T.dot(pre_h, Uf) + T.dot(pre_z, Zf) + bf_reshape_2x)
	o_t= sigmoid(T.dot(x_t, Wo) + T.dot(pre_h, Uo) + T.dot(pre_z, Zo) + bo_reshape_2x)
	c_th = tanh(T.dot(x_t, Wc)  + T.dot(pre_h, Uc) + T.dot(pre_z, Zc) + bc_reshape_2x)

	c_t = f_t*pre_c + i_t_new*c_th

	h_t = o_t*T.tanh(c_t) #shape (1, N, h_dim)

	h_t_context = T.repeat(h_t, image_feature_region.shape[1], axis = 0) #new shape (No_region, N, h_dim)
	image_feature_reshape = T.transpose(image_feature_region, (1, 0, 2))
	#compute non-linear correlation between h_t(current text) to image_feature_region (64 for 128*128 and 196 for 224*224)
	# pdb.set_trace()
	m_t = T.tanh(T.dot(h_t_context, Hcontext) + T.dot(image_feature_reshape, Zcontext)) #shape (No_region, N, context_dim)

	e = T.dot(m_t, Va) #No_region, N, 1
	e_reshape = e.reshape((e.shape[0], T.prod(e.shape[1:])))

	e_softmax = softmax_along_axis(e_reshape, axis = 0) #shape No_region, N

	e_t = T.transpose(e_softmax, (1,0)) #shape N, No_region
	e_t_r = e_t.reshape([-1, e_softmax.shape[0], e_softmax.shape[1]]) #3D tensor 1, N, No_region
	e_t_r_t = T.transpose(e_t_r, (1,0, 2)) # shape N, 1, No_region
	e_3D = T.repeat(e_t_r_t, e_t_r_t.shape[2], axis = 1) #shape N, No_region, No_region  image_feature_region.shape[1]
	e_3D_t = T.transpose(e_3D, (1,2,0)) #No_region, No_region, N

	identity_2D = T.identity_like(e_3D_t)# shape No_region, No_region
	identity_3D = identity_2D.reshape([-1, identity_2D.shape[0], identity_2D.shape[1]]) # shape 1, No_region, No_region
	identity_3D_t = T.repeat(identity_3D,  image_feature_region.shape[0], axis = 0)
	e_3D_diagonal = e_3D*identity_3D_t #diagonal tensor 3D  (N, No_region, No_region)

	out_weight_y, updates = theano.scan(fn=onestep_weight_feature_multiply,
	                                   outputs_info=[weight_y],
	                                   sequences=[e_3D_diagonal, image_feature_region],
	                                   non_sequences=[])

	z_t = T.sum(out_weight_y, axis = 1) #shape (N, feature_dim)

	z_t_r = z_t.reshape((-1,z_t.shape[0],z_t.shape[1]))


	return [h_t, c_t, z_t_r]

def DimShuffleTest(axis = -1):
	# input_shape = (4,3, 2)
	# broadcast = ['x']*len(input_shape)
	# broadcast[axis] = 0

	# tensor3_in = np.ones((4,)).astype("float32")
	# print(tensor3_in)
	# print(tensor3_in.shape)

	# tensor3 = theano.tensor.ftensor3('g')
	# out = tensor3.dimshuffle(*broadcast)
	# out_func = theano.function([tensor3], out)

	
	# out_value= out_func(tensor3_in)
	# print("output shape is: ")
	# print(out_value.shape)

	#convert from N*M to N*1*M tensor
	tensor_N_M = T.fmatrix('tensor_N_M')
	out_dimshuffle = tensor_N_M.dimshuffle(1, 'x', 0)
	out_tensor = theano.function([tensor_N_M], out_dimshuffle)

	tensor_N_M_in = np.ones((4,3)).astype("float32")
	out_value = out_tensor(tensor_N_M_in)
	print(out_value.shape)
	print(out_value)

def basic_shuffle_test():
	batch_range = np.arange(0, 100, 10, dtype=np.int64)
	print('batch_range:')
	print(batch_range)

	np.random.shuffle(batch_range)
	print("shuffle batch range")
	print(batch_range)

def reshape_test():
	tensor2 = theano.tensor.fmatrix('tensor2')
	reshape_tensor = tensor2.reshape((theano.tensor.prod(tensor2.shape[:]), 1))
	function_reshape = theano.function([tensor2], reshape_tensor)

	input_mat = np.ones((4,3)).astype("float32")
	out_mat = function_reshape(input_mat)

	print(out_mat.shape)
	print(input_mat.shape)

	input_mat[1, 1] = 10.0
	print(out_mat)
	print(input_mat)

	out_mat[0,0] = 200.0
	print(out_mat)
	print(input_mat)

if __name__=='__main__':
	#basic_shuffle_test()
	# UnitTest_OnestepAttend()
	#UnitTestLSTM_Attend() #not in debug mode now
	# BasicTheano()
	#UnitTestLSTM()	
   # UnitTestPoolLayer()
   # UnitTestConvLayer()
   #NumericalGradientCheckExample()
   #checkingLSTM()
   DimShuffleTest(0)
	# reshape_test()