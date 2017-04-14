from theano.ifelse import ifelse
import theano.tensor as T
from NeuralNet import *
import numpy as np
import theano
class Trainer(object):
    """
    This is the class that stores attributes and methods for training a network
    You can find things like learning rate, training routine, update rules here

    opts: dict
    ['rule_name']:          string
                            decide update rule
    ['dzdw_norm_thres']:    float
                            normalization threshold of W grad, decide how to clip W
    ['dzdb_norm_thres']:    float
                            normalization threshold of b grad, decide how to clip b

    param_list: list
                store list of params (w,b) of "updatable" layers in the network
    momentum_list: list
                    store respective momentum of params
    grad_list: list
                    store respective gradient of params in each iteration
    """
    def __init__(self):
        """ Set default attributes
        """
        self.opts = {}
        self.opts['num_epoch'] = 100
        self.opts['num_sample'] = -1
        self.opts['batch_size'] = 10
        self.opts['dzdw_norm_thres'] = np.inf
        self.opts['dzdb_norm_thres'] = np.inf
        self.opts['train'] = True

        self.opts['num_val_sample'] = -1
        self.opts['validation'] = True

        self.opts['num_test_sample'] = -1
        self.opts['test'] = False

        self.opts['test_emp'] = False

        # Save everything every 5 epochs
        self.opts['save_freq'] = 10
        self.opts['save'] = False

        # update rule name can be SGD, SGDM, ADAM or none
        self.opts['update_rule_name'] = 'ADAM'
        self.opts['momentum_rate'] = theano.shared(np.asarray(0.9, dtype=theano.config.floatX))

        self.current_epoch = 0

        self.param_list = []
        self.momentum_list = []
        self.grad_list = []
        self.config_list = []

        # Meta training data, for ploting and such
        self.all_i = []
        self.all_e = []
        self.all_val = []
        self.all_emp = []
        self.all_v_emp = []

    def InitParams(self, net):
        """ Iterate over the network and concat params & init gradients for user
        Only layers with .updatable attribute = true are included
        :type net: NeuralNet.*
        :param net: store layer objects from LayerProvider
        """
        self.grad_list = []
        self.param_list = []
        self.momentum_list = []

        for key, value in net.content.iteritems():
            if (hasattr(value,'updatable')):
                if (value.updatable == True):
                    self.param_list.append(value.param)

                    zero_momentum_list = []
                    print(type(value))
                    for i in range(len(value.param)):
                        print(value.param[i]['val'].shape.eval())
                        zero_filter = theano.shared(np.zeros(value.param[i]['val'].shape.eval(), theano.config.floatX))
                        zero_momentum_list.append(zero_filter)

                    self.momentum_list.append(zero_momentum_list)
                    self.grad_list.append(T.grad(net.content['cost'].output, [p['val'] for p in value.param]))
                    
                    config_temp_list = []
                    
                    for i in range(len(value.param)):
                        config = {}
                        config.setdefault('learning_rate', 1e-3)
                        config.setdefault('beta1', 0.9)
                        config.setdefault('beta2', 0.999)
                        config.setdefault('epsilon', 1e-8)
                        config.setdefault('t', 0)
                        m = theano.shared(np.zeros(value.param[i]['val'].shape.eval(), theano.config.floatX))
                        config.setdefault('m', m)

                        v = theano.shared(value= np.zeros(value.param[i]['val'].shape.eval(), theano.config.floatX))
                        config.setdefault('v', v)
                        config_temp_list.append(config)

                    self.config_list.append(config_temp_list)


        breakpoint=1

    def InitUpdateRule(self, net):
        """ Iterate over param list to create update rule for the main loop
        :type param_list: list
        :param param_list: store params of several layers in a list

        :type grad_list: list
        :param grad_list: store gradients of respective params

        :type lr: list of float numbers
        :param lr: learning rate, can be either list or float
        """
        rule_name = self.opts['update_rule_name']

        if (rule_name.upper == 'NONE'):
            return []

        momentum_rate = self.opts['momentum_rate']

        dzdw_norm_thres = self.opts['dzdw_norm_thres']
        dzdb_norm_thres = self.opts['dzdb_norm_thres']
        update_rule = []
        for params_i, grads_i, momentums_i, lr_i, config_i in zip(self.param_list, self.grad_list, self.momentum_list, net.net_opts['lr'], self.config_list):

            # Make another loop to iterate over the sub-list of params for each updatable layer
            for j in range(len(params_i)):
                param_j = params_i[j]
                grads_j = grads_i[j]
                momentum_j = momentums_i[j]
                config_j = config_i[j]


                grad_norm = T.sqrt(T.sum(grads_j**2))
                # if (param_counter % 2 ==0 ):
                #     grad_norm_thres = dzdw_norm_thres
                # else:
                #     grad_norm_thres = dzdb_norm_thres

                # Assuming that the last param in a param list is b (bias)
                if (param_j['type'] == 'b'):
                    grad_norm_thres = dzdb_norm_thres
                elif (param_j['type'] == 'w'):
                    grad_norm_thres = dzdw_norm_thres
                else:
                    raise ValueError('Neural network param type not known.')
                # If pure SGD
                if (rule_name == 'SGD'):
                    update_rule += [(
                        param_j['val'],
                        ifelse(
                            T.gt(grad_norm, grad_norm_thres),
                            param_j['val'] - grads_j*lr_i*grad_norm_thres/grad_norm,
                            param_j['val'] - grads_j*lr_i
                        )
                    )]

                # If SGD with momentum
                elif (rule_name == 'SGDM'):
                    update_rule += [(
                        param_j['val'],
                        ifelse(
                            T.gt(grad_norm, grad_norm_thres),
                            param_j['val'] - grads_j*lr_i*grad_norm_thres/grad_norm - momentum_rate*momentum_j,
                            param_j['val'] - grads_j*lr_i - momentum_rate*momentum_j
                        )
                    )]

                    update_rule += [(
                        momentum_j,
                        ifelse(
                            T.gt(grad_norm, grad_norm_thres),
                            grads_j*lr_i*grad_norm_thres/grad_norm + momentum_rate*momentum_j,
                            grads_j*lr_i - momentum_rate*momentum_j
                        )
                    )]
                elif (rule_name == 'ADAM'):
                    config_j['t'] += 1
                    update_rule += [(config_j['m'], config_j['beta1']*config_j['m'] + (1 - config_j['beta1'])*grads_j)]
                    update_rule += [(config_j['v'], config_j['beta2']*config_j['v'] + (1 - config_j['beta2'])*(grads_j**2))]

                    #config_j['m'] = config_j['beta1']*config_j['m'] + (1 - config_j['beta1'])*grads_j.eval()
                    #config_j['v'] = config_j['beta2']*config_j['v'] + (1 - config_j['beta2'])*(grads_j.eval()**2)

                    mb = config_j['m']/(1 - config_j['beta1']**config_j['t'])
                    mv = config_j['v']/(1 - config_j['beta2']**config_j['t'])
                    #mv = np.sqrt(mv)
                    #anpha = config_j['learning_rate']*(1-config_j['beta2']**config_j['t'])
                    mv = T.sqrt(mv)
                    update_rule += [(param_j['val'], param_j['val'] - config_j['learning_rate']*mb/ (mv + config_j['epsilon']))]

        return update_rule
