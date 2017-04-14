from __future__ import print_function
from LayerProvider import *
import copy

class NeuralNet(object):
    """
    Class that stores network & layer
    :type test_input: 4D tensor
    :param test_input: Real input which will be used to calculate outputs of each layer

    :type test_output: 4D tensor
    :param test_output: Real output we expect to come out at the last layer of the network
    net_opts: global options such as start learning rate, rng
                ['rng']: random number generator, used for filter init
                ['l1_learning_rate']: learning rate for the first 'updatable' layer

    layer_opts: layer-specific options
                Specific options like conv filter_shape should be decided when constructing the network
                ['']
    content: dict that contains actual layers
    """

    def __init__(self, test_input=None, test_output=None):
        """
        :type input: theano.tensor.tensor4
        :param input: input of the CNN, if it is not provided then the network will use a symbolic input instead
        """

        self.net_opts = {}
        self.layer_opts = {}

        # Setting default options for net_opts
        self.net_opts['rng_seed'] = 1231
        self.net_opts['rng'] = np.random.RandomState(self.net_opts['rng_seed'])
        self.net_opts['l1_learning_rate'] = np.asarray(0.001, theano.config.floatX)


        # Default options for softmax layers
        self.layer_opts['softmax_norm_dim'] = 1

        # Default options for relu layers
        self.layer_opts['relu_alpha'] = 0.01

        # Deafult options for elu layers
        self.layer_opts['elu_alpha'] = 1

        # Default l2 term
        self.layer_opts['l2_term'] = 0.0005

        # Default dropping rate for dropout
        self.layer_opts['drop_rate'] = 0

        # Network name for saving
        self.name = 'netnet'
        
        # Train mode or not, used for layers like DropOut
        self.mode = theano.shared(1)

        # The content dictionary will store actual layers (LayerProvider)
        self.content = {}
        self.input = []
        if (test_input != None):
            self.input.append(test_input)
        else:
            self.input.append(T.tensor4('x', dtype=theano.config.floatX))
        self.content['input'] = InputLayer(self.input[0])

        self.output = []
        self.weight = []
        if(test_output != None):
            self.output.append(test_output)
        else:
            self.output.append(T.tensor4('y', dtype=theano.config.floatX))
        self.weight.append(T.tensor4('weight', dtype=theano.config.floatX))

        self.index = T.lvector('index')
        self.batch_size = T.scalar('batch_size')

        # For SGDR
        self.reset_opts = {}
        self.reset_opts['min_lr'] = np.asarray(0.0001, dtype=theano.config.floatX)
        self.reset_opts['max_lr'] = self.net_opts['l1_learning_rate']
        self.reset_opts['t_mult'] = 2
        
        self.t_cur = 0
        self.t_i = 1

        self.train_function = None
        self.test_function = None
        self.val_function = None

    #TODO: Fix this function since it will be stuck if there is no loss (?)
    def simpleprint(self):
        """ Print a net

        """
        l = 1
        stop_print = False
        print('input', end = "")
        while not stop_print:
            for key, value in self.content.iteritems():
                if (hasattr(value, 'topo_order')):
                    if (value.topo_order == l):
                        if (l == len(self.content)):
                            print('', end="")
                            stop_print = True
                        l += 1
                        print (' -> %s' % key, end="")
        print('')

    def InitLR(self, factor):
        """ Generate varied learning rates for different layers
        Only layers with .updatable attribute = true are included
        :type net: NeuralNet.NeuralNet
        :param net: The network

        :type factor: float
        :param factor: next layer's lr = current layer's lr * factor
                    Note that only layer that is 'updatable' count toward this list
        """
        self.net_opts['lr'] = []
        for key, value in self.content.iteritems():
            if (hasattr(value,'updatable')):
                if (value.updatable == True):

                    # Note that the same lr applies to all params (W, b) of a layer
                    l_factor = np.asarray(factor**(value.update_order-1),dtype=theano.config.floatX)
                    self.net_opts['lr'] += [theano.shared(self.net_opts['l1_learning_rate'] * l_factor)
                                            for i in range(0, len(value.param))]
        # This won't be affected by LR restart
        self.const_lr = copy.deepcopy(self.net_opts['lr']) 
    def InitTrainFunction(self, update_rule, real_input, expected_output, 
    	additional_output=None, weight=None, additional_output_obj=None):
        """ Generate the feed forward function for both train processes
        :type update_rule: List
        :param update_rule: List of tuples for updating params

        :type real_input: 4D Tensor or list of 4D Tensor
        :param real_input: entire training dataset, shaped as a 4D theano shared tensor
            If the network take in multiple inputs, real_input should be a list of 4D Tensor

        :type expected_output: 4D tensor
        :param expected_output: label, for now the label dim should be 1

        :type additional_output: list of basestring
        :param additional_output: tell the function to return additional output at specific layers

        :type weight: 4D tensor
        :param weight: sample weight
        """
        self.train_function = None

        # Check input consistency
        if (type(real_input) != list):
            real_input = [real_input]

        if (type(expected_output) != list):
            expected_output = [expected_output]

        assert len(real_input) == len(self.input), "The network require %d inputs, function argument provided %d" % (
            len(self.input), len(real_input))
        assert len(expected_output) == len(self.output), "The network produce %d output, function argument provided %d" % (
            len(self.output), len(expected_output))

        # For layers that depends on batch size
        givens = {}
        for key, value in self.content.iteritems():
            # self.index.shape[0] is batch size
            if (type(value) == LSTM):
                givens = {
                    value.h_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX),
                    value.c_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX)
                }
            if (type(value) == LSTM_Attend):
            	givens = {
            		value.z_m1_sym: T.zeros((1, self.index.shape[0], value.Z_shape[0]), dtype=theano.config.floatX)
            		
            	}

        function_output = [self.content['cost'].output]
        if (additional_output != None):
            function_output += [self.content[l].output for l in additional_output]
      	
      	if (additional_output_obj):
      		if (type(additional_output_obj) != list):
      			additional_output_obj = [additional_output_obj]
      		function_output += additional_output_obj
      	print("DONE function_output")

        for i in range(len(self.input)):
            givens.update({
                self.input[i]: real_input[i][self.index,:,:,:]
            })
        print("DONE input")

        for i in range(len(self.output)):
            givens.update({
                self.output[i]: expected_output[i][self.index,:,:,:]
            })
        print("DONE output")

        if (weight == None):
            weight = []
            for i in range(len(expected_output)):
                weight += [theano.shared(np.ones_like(expected_output[i].eval()))]
        elif (type(weight) != list):
            weight = [weight]
        

        for i in range(len(self.weight)):
            givens.update({
                self.weight[i]: weight[i][self.index, :, :, :]
            })
        print("DONE weight")
        
        self.train_function = theano.function(
            [self.index],
            outputs=function_output + self.output,
            updates=update_rule,
            givens=givens
        )

    def InitTestFunction(self, test_input, test_output):
        """ Generate the feed forward function for both test/validation processes
        It's simply a train_function without the updating part
        :type update_rule: List
        :param update_rule: List of tuples for updating params

        :type test_input: 4D Tensor or list of 4D Tensor
        :param test_input: entire testing dataset, shaped as a 4D theano shared tensor
            If the network take in multiple inputs, test_input should be a list of 4D Tensor

        :type test_output: 4D tensor
        :param test_output: label, for now the label dim should be 1
        """

        # Check input consistency
        if (type(test_input) != list):
            test_input = [test_input]

        if (type(test_output) != list):
            test_output = [test_output]

        assert len(test_input) == len(self.input), "The network require %d inputs, function argument provided %d" % (
            len(self.input), len(test_input))
        assert len(test_output) == len(
            self.output), "The network produce %d output, function argument provided %d" % (
            len(self.output), len(test_output))

        # Special inputs that depend on batch size, therefore we need them to be given before training/testing
        givens = {}
        for key, value in self.content.iteritems():
            # self.index.shape[0] is batch size
            if (type(value) == LSTM):
                givens = {
                    value.h_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX),
                    value.c_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX)
                }

        for i in range(len(self.input)):
            givens.update({
                self.input[i]: test_input[i][self.index,:,:,:]
            })

        for i in range(len(self.output)):
            givens.update({
                self.output[i]: test_output[i][self.index,:,:,:]
            })
        

        self.test_function = theano.function(
            [self.index],
            outputs=self.content['cost'].output,
            givens=givens
        )

    # Duplicated codes
    def InitValFunction(self, val_input, val_output, additional_output=None, weight=None, additional_output_obj=None):
        """ Generate the feed forward function for both test/validation processes
        It's simply a train_function without the updating part
        :type update_rule: List
        :param update_rule: List of tuples for updating params

        :type val_input: 4D Tensor or list of 4D Tensor
        :param val_input: entire validation dataset, shaped as a 4D theano shared tensor.
            If the network take in multiple inputs, test_input should be a list of 4D Tensor

        :type val_output: 4D tensor
        :param val_output: label, for now the label dim should be 1

        :type additional_output: list of basestring
        :param additional_output: tell the function to return additional output at specific layers

        :type weight: 4D tensor
        :param weight: sample weight
        """
        # Check input consistency
        if (type(val_input) != list):
            val_input = [val_input]

        if (type(val_output) != list):
            val_output = [val_output]

        assert len(val_input) == len(self.input), "The network require %d inputs, function argument provided %d" % (
            len(self.input), len(val_input))
        assert len(val_output) == len(
            self.output), "The network produce %d output, function argument provided %d" % (
            len(self.output), len(val_output))

        # Special inputs that depend on batch size, therefore we need them to be given before training/testing
        givens = {}
        for key, value in self.content.iteritems():
            # self.index.shape[0] is batch size
            if (type(value) == LSTM):
                givens = {
                    value.h_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX),
                    value.c_m1_sym: T.zeros((1, self.index.shape[0], value.W_shape[1]), dtype=theano.config.floatX)
                }
            if (type(value) == LSTM_Attend):
            	givens = {
            		value.z_m1_sym: T.zeros((1, self.index.shape[0], value.Z_shape[0]), dtype=theano.config.floatX)
            	}


        for i in range(len(self.input)):
            givens.update({
                self.input[i]: val_input[i][self.index,:,:,:]
            })

        for i in range(len(self.output)):
            givens.update({
                self.output[i]: val_output[i][self.index,:,:,:]
            })

        function_output = [self.content['cost'].output]
        if (additional_output != None):
            function_output += [self.content[l].output for l in additional_output]

        if (additional_output_obj):
      		if (type(additional_output_obj) != list):
      			additional_output_obj = [additional_output_obj]
      		function_output += additional_output_obj

        if (weight == None):
            weight = []
            for i in range(len(val_output)):
                weight += [theano.shared(np.ones_like(val_output[i].eval()))]
        elif (type(weight) != list):
            weight = [weight]

        for i in range(len(self.weight)):
            givens.update({
                self.weight[i]: weight[i][self.index, :, :, :]
            })

        self.val_function = theano.function(
            [self.index],
            outputs=function_output + [val_output[i][self.index,:,:,:] for i in range(len(self.output))],
            givens=givens
        )


    def FirstParamIndex(self):
        """
        Return the index of the param of the first layer in the network
        This need to be done because while creating params using dictionary, the order of the params are not kept accordingly
        to the network
        TODO: Find a better way to handle this
        This is just plain stupid
        """

        dict_index = 0
        for key, value in self.content.iteritems():
            if (hasattr(value,'updatable')):
                if (value.updatable == True):
                    # The first updatable layer
                    if (value.update_order == 1):
                        return dict_index
                    dict_index += len(value.param)
        return -1

    #TODO: These are still fairly simplistic, if a network has branches then it would not work
    def GetOutputShape(self, input_shape):
        """ Get output shape up until the last layer of the network given input_shape
        :type input_shape: tuple
        :param input_shape: shape of the input
        """
        topo_index = 1
        output_shape = list(input_shape)
        while True:
            for key, value in self.content.iteritems():
                if (value.topo_order == topo_index):
                    if (type(value) == ConvLayer):
                        w_factor = 0
                        h_factor = 0
                        if (type(value.border_mode) == int):
                            h_factor = value.border_mode
                            w_factor = value.border_mode

                        if (type(value.border_mode) == tuple):
                            h_factor = value.border_mode[0]
                            w_factor = value.border_mode[1]

                        output_shape[1] = value.filter_shape[0]
                        output_shape[2] = output_shape[2] - value.filter_shape[2] + 1 + h_factor*2
                        output_shape[3] = output_shape[3] - value.filter_shape[3] + 1 + w_factor*2
                    if (type(value) == Pool2DLayer):
                        output_shape[2] = np.int64(
                            np.ceil(
                                float(output_shape[2])/float(value.pool_size[0])
                            )
                        )
                        output_shape[3] = np.int64(
                            np.ceil(
                                float(output_shape[3])/float(value.pool_size[1])
                            )
                        )
                        breakpoint = 1
                    if (type(value) == FCLayer):
                        output_shape[1] = value.layer_shape[1]
                        output_shape[2] = 1
                        output_shape[3] = 1

                    topo_index += 1
                    if (topo_index > len(self.content)):
                        return tuple(output_shape)


    def NNCopy(self):
        """
        Copy src Neural Network to a dest Neural network without the train_function and val_function
        This is to save memory when we have to save a network/load a pretrained network
        """
        dest = ShowTellNet()
        dest.input = copy.copy(self.input)
        dest.index = copy.copy(self.index)
        dest.output = copy.copy(self.output)
        dest.content = copy.copy(self.content)
        dest.net_opts = copy.copy(self.net_opts)
        dest.layer_opts = copy.copy(self.layer_opts)
        dest.weight = copy.copy(self.weight)
        dest.reset_opts = copy.copy(self.reset_opts)
        dest.train_function = None
        dest.val_function = None
        dest.test_function = None
        dest.name = copy.copy(self.name)
        return dest

class ConvNeuralNet(NeuralNet):
    """
    Class that stores network & layer
    :type test_input: 4D tensor
    :param test_input: Real input which will be used to calculate outputs of each layer

    :type test_output: 4D tensor
    :param test_output: Real output we expect to come out at the last layer of the network
    net_opts: global options such as start learning rate, rng
                ['rng']: random number generator, used for filter init
                ['l1_learning_rate']: learning rate for the first 'updatable' layer

    layer_opts: layer-specific options
                Specific options like conv filter_shape should be decided when constructing the network
                ['']
    content: dict that contains actual layers
    """

    def __init__(self, test_input=None, test_output=None):
        """
        :type input: theano.tensor.tensor4
        :param input: input of the CNN, if it is not provided then the network will use a symbolic input instead
        """
        NeuralNet.__init__(self, test_input, test_output)

        # Setting default options for layer_opts
        # Default options for conv layers
        self.layer_opts['border_mode'] = 'valid'
        self.layer_opts['conv_stride'] = (1,1)
        self.layer_opts['updatable'] = True

        # Default options for pooling layers
        self.layer_opts['pool_stride'] = (2,2)
        self.layer_opts['pool_padding'] = (0,0)
        self.layer_opts['pool_mode'] = 'max'
        self.layer_opts['pool_size'] = (2,2)
        self.layer_opts['ignore_border'] = False

        # Network name for saving
        self.name = 'convnetnet'

        self.index = T.lvector('index')
        
        # Related functions
        self.train_function = None
        self.val_function = None

class ShowTellNet(ConvNeuralNet):
    """
    Class that stores network & layer. This Network was modified so that it can take two types of inputs (images and
    sentences)
    :type test_input_image: 4D tensor
    :param test_input_image: Real input which will be used to calculate outputs of each layer

    :type test_input_sentence: 4D tensory
    :param test_input_sentence: Sentences

    :type test_output: 4D tensor
    :param test_output: Real output we expect to come out at the last layer of the network
    net_opts: global options such as start learning rate, rng
                ['rng']: random number generator, used for filter init
                ['l1_learning_rate']: learning rate for the first 'updatable' layer

    layer_opts: layer-specific options
                Specific options like conv filter_shape should be decided when constructing the network
                ['']
    content: dict that contains actual layers
    """
    def __init__(self, test_input_image=None, test_input_sentence=None, test_output=None):
        ConvNeuralNet.__init__(self, test_input_image, test_output)

        # Clear input
        self.input = []

        if (test_input_image != None):
            self.input.append(test_input_image)
        else:
            self.input.append(T.tensor4('x_img', dtype=theano.config.floatX))
        self.content['input_img'] = InputLayer(self.input[0])

        if (test_input_sentence != None):
            self.input.append(test_input_sentence)
        else:
            self.input.append(T.tensor4('x_sen', dtype=theano.config.floatX))
        self.content['input_sen'] = InputLayer(self.input[1])

        # Set word embedding params
        self.layer_opts['num_emb'] = 350

        # Set LSTM params
        self.layer_opts['num_lstm_node'] = 128

        #Set LSTM Attend params
        self.layer_opts['num_region'] = 196
        self.layer_opts['num_dimension_feature'] = 512
        self.layer_opts['context_dim'] = 300
        self.layer_opts["num_hidden_node"] = 1024
        self.layer_opts["num_deep_out_node"] = 1024
        self.layer_opts["n_word"] = 2000
#class ShowAttendTellNet(ConvNeuralNet):
	"""
	class that stores network and layer. This network was modified so that it can take three types of inputs (images and sentences)
	"""
