from __future__ import print_function
from MainLoop import  *
from LayerProvider import  *

import numpy as np
from Trainer import Trainer
from NeuralNet import ConvNeuralNet

if __name__ == '__main__':
    trainer = Trainer()
    trainer.opts['num_sample'] = 100

    net = ConvNeuralNet()
    image_shape = (1,1,8,8)

    # Generate 4D label tensor
    one_hot = np.eye(3,3)

    # Generate expected output (aka label)
    label = np.ones(trainer.opts['num_sample'], dtype=np.int16)
    one_hot = one_hot[label]
    expected_output = theano.shared(
        np.asarray(
            np.reshape(one_hot,(trainer.opts['num_sample'],3,1,1)),
            dtype=theano.config.floatX
        )
    )
    input = theano.shared(
        np.repeat(
            np.asarray(
                net.net_opts['rng'].uniform(low=-1, high=1, size=image_shape),
                dtype=theano.config.floatX
            ),
            trainer.opts['num_sample'],
            0
        )
    )

    # Construct the network
    net.layer_opts['filter_shape'] = (3,1,8,8)
    net.content['l1'] = ConvLayer(net, net.content['input'])

    net.layer_opts['filter_shape'] = (3,3,1,1)
    net.content['l2'] = ConvLayer(net, net.content['l1'])

    net.layer_opts['softmax_norm_dim'] = 1
    net.content['l3']  = SoftmaxLayer(net, net.content['l2'])
    net.content['cost'] = CategoricalCrossEntropy(net, net.content['l3'])

    # Print the network architecture
    net.simpleprint()

    # Initialize learning rate for each updatable layer
    net.InitLR(0.5)

    # Create params list, grad list, momentum list for the theano function to update
    trainer.InitParams(net)
    trainer.opts['validation'] = False
    trainer.opts['test_emp'] = False
    # Update rule
    train_update_rule = trainer.InitUpdateRule(net)
    net.InitTrainFunction(train_update_rule, input, expected_output, ['l3'])
    main_loop = SGDRMainLoop(net)
    main_loop.run(net, trainer)
