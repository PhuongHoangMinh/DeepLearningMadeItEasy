from __future__ import print_function

import numpy as np
import theano
import copy
import time
import pdb
from EvaluationHelper import EmpError
from Utils import SaveList

theano_pi = theano.shared(np.pi)

class SGDMainLoop(object):
    def run(self, net, trainer,last_e=-1):
        """ Run SGD over trainer.opts['num_epoch'] epochs
        :type net: NeuralNet.NeuralNet
        :param net: a NeuralNet object

        :type trainer: LayerProvider.*
        :param pre_layer: The layer that outputs to this layer
        """
        batch_size = 1
        num_sample = 1
        num_epoch = 1
        num_ite = 0

        num_val_sample = 1
        num_val_ite = 0
        
        

        if(trainer.opts['train']):
            batch_size = trainer.opts['batch_size']
            num_sample = trainer.opts['num_sample']
            num_epoch = trainer.opts['num_epoch']
            num_ite = int(np.ceil(num_sample/batch_size))

        if(trainer.opts['validation']):
            num_val_sample = trainer.opts['num_val_sample']
            num_val_ite = int(num_val_sample/batch_size)

        print('Function compilation done. Start training...')

        for e in np.arange(last_e+1, num_epoch):
            e_cost = 0
            v_cost = 0
            for lr_i in net.net_opts['lr']:
                lr_i.set_value(lr_i.get_value()*1)

            t_start = time.time()
            empirical_err = 0
            if (trainer.opts['train']):
                for i in np.arange(0, num_ite):
                    batch_range = np.arange(i, trainer.opts['num_sample'], num_ite, dtype=np.int64)
                    function_output = net.train_function(batch_range)
                    e_cost += function_output[0]
                    print('Finished iteration %d of epoch %d: %f' % (i, e, function_output[0]))

                    if (trainer.opts['test_emp']):
                        empirical_err += EmpError(function_output[1], function_output[2])

                # Saving
                if (trainer.opts['save'] and e % trainer.opts['save_freq'] == 0):
                    SaveList([net, trainer, e], '../../data/trained_model/%s_e-%05d.dat' % (net.name, e))

            t_end = time.time()

            if (trainer.opts['validation']):
                for i in np.arange(0, num_val_ite):
                    batch_range = np.arange(i, trainer.opts['num_val_sample'], num_val_ite, dtype=np.int64)
                    v_cost += net.val_function(batch_range)

            print('Epoch %d done in %fs' % (e, t_end - t_start))

            trainer.all_e.append(e_cost / num_sample)
            trainer.all_val.append(v_cost / num_val_sample)
            print('Train cost: %f' % (e_cost / num_sample))
            print('Valid cost: %f' % (v_cost / num_val_sample))
            if (trainer.opts['test_emp']):
                print('Empirical error: %f' % (float(empirical_err) / float(num_sample)))
                trainer.all_emp.append(float(empirical_err) / float(num_sample))

class SGDRMainLoop(object):
    """
    SGD with learning rate restarts
    Implemented based on the paper Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic Gradient Descent with
    Restarts." arXiv preprint arXiv:1608.03983 (2016).

    t_cur: current epoch in a restart session
    t_i: the number of epochs need to be run before we reset the learning rate
    t_mult: (next t_i) = (current t_i)*t_mult
    """

    def __init__(self, net, save_path):
        """ Init the loop
        :type net: NeuralNet.NeuralNet
        :param net: a NeuralNet object
        """
        self.reset_opts = net.reset_opts
        self.sym_const_lr = net.const_lr
        self.const_lr = []
        for lr_i in self.sym_const_lr:
            self.const_lr += [lr_i.eval()]

        self.min_lr = self.reset_opts['min_lr']
        self.max_lr = self.reset_opts['max_lr']
        net.t_mult = self.reset_opts['t_mult']
        #net.t_cur = 0
        #net.t_i = 1

        self.first_param_index = net.FirstParamIndex()
        self.name = net.name
        self.save_path = save_path

    def WriteLog(self, line):
        f = open(self.save_path + self.name + '.log', 'a')
        f.write(line + '\n')
        f.close()

    def LRRestart(self, net):
        """
        Modify learning rates of the network each iteration according to the equation provided in the original paper
        :type net: NeuralNet.NeuralNet
        :param net: the network to be trained
        """
        if (net.t_cur >= net.t_i):
            net.t_cur = 0
            net.t_i *= net.t_mult
            for lr_i, sym_const_lr_i in zip(net.net_opts['lr'], self.sym_const_lr):
                lr_i.set_value(sym_const_lr_i.eval())
        else:
            net.t_cur += 1
            new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(net.t_cur * np.pi / net.t_i))
            reduce_rate = new_lr / net.net_opts['l1_learning_rate']
            reduce_rate = reduce_rate.astype(theano.config.floatX)
            for lr_i, const_lr_i in zip(net.net_opts['lr'], self.const_lr):
                lr_i.set_value(const_lr_i * reduce_rate)

    def run(self, net, trainer, last_e=0):
        """ Run SGD with learning rate restarts over trainer.opts['num_epoch'] epochs
        :type net: NeuralNet.NeuralNet
        :param net: a NeuralNet object

        :type trainer: Trainer.*
        :param trainer: Trainer object
        """
        batch_size = 1
        num_sample = 1
        num_epoch = 1
        num_ite = 0

        num_val_sample = 1
        num_val_ite = 0

        if (trainer.opts['train']):
            batch_size = trainer.opts['batch_size']
            num_sample = trainer.opts['num_sample']
            num_epoch = trainer.opts['num_epoch']
            num_ite = int(np.ceil(num_sample / batch_size))

            #Mix train and validation
            num_val_sample = trainer.opts['num_val_sample']
            
            num_val_ite = int(np.ceil(float(num_val_sample) / float(batch_size)))

            print(num_sample)
            print(batch_size)
            print(num_ite)
        if (trainer.opts['validation']):
            if (trainer.opts.has_key('batch_size')):
                if (trainer.opts['batch_size'] > 1):
                    batch_size = trainer.opts['batch_size']

            num_val_sample = trainer.opts['num_val_sample']
            assert num_val_sample > 0, ("Validation while training is turned on but trainer.opts['num_val_sample']"
                                        + " < 1. Please specify the number of validation samples.")

            num_val_ite = int(np.ceil(float(num_val_sample) / float(batch_size)))

        print('Function compilation done. Start training...')

        for e in np.arange(last_e, num_epoch):
            e_cost = 0
            v_cost = 0
            t_start = time.time()
            empirical_err = 0
            v_empirical_err = 0

            # If we are training the network
            if (trainer.opts['train']):
                for i in np.arange(0, num_ite):
                    # Compute a feed forward pass and add result to e_cost
                    batch_range = np.arange(i, trainer.opts['num_sample'], num_ite, dtype=np.int64)
                    #np.random.shuffle(batch_range)
                    # batch_range = np.random.choice(trainer.opts['num_sample'], batch_size).astype(np.int64)
                    #iwe_W0 = np.asarray(net.content['iwe'].We.eval())
                    #we_W0 = np.asarray(net.content['we'].We.eval())
                    lstm_Wi0 = np.asarray(net.content['lstm_attend'].W['i'].eval())
                    lstm_Wo0 =  np.asarray(net.content['lstm_attend'].W['o'].eval())
                    lstm_Wf0 =  np.asarray(net.content['lstm_attend'].W['f'].eval())
                    lstm_Wc0 =  np.asarray(net.content['lstm_attend'].W['c'].eval())
                    #lstm_Ui0 = np.asarray(net.content['lstm_attend'].U['i'].eval())
                    #lstm_Uo0 =  np.asarray(net.content['lstm_attend'].U['o'].eval())
                    #lstm_Uf0 =  np.asarray(net.content['lstm_attend'].U['f'].eval())
                    #lstm_Uc0 =  np.asarray(net.content['lstm_attend'].U['c'].eval())

                    #lstm_Zi0 = np.asarray(net.content['lstm_attend'].Z['i'].eval())
                    #lstm_Zo0 =  np.asarray(net.content['lstm_attend'].Z['o'].eval())
                    #lstm_Zf0 =  np.asarray(net.content['lstm_attend'].Z['f'].eval())
                    #lstm_Zc0 =  np.asarray(net.content['lstm_attend'].Z['c'].eval())

                    #lstm_Zcon0 = np.asarray(net.content['lstm_attend'].Zcontext.eval())
                    #lstm_Hcon0 = np.asarray(net.content['lstm_attend'].Hcontext.eval())
                    #lstm_Xcon0 = np.asarray(net.content['lstm_attend'].Xcontext.eval())
                    #lstm_Va0 = np.asarray(net.content['lstm_attend'].Va.eval())



                    # lstm_bi0 = np.asarray(net.content['lstm'].b['i'].eval())
                    # lstm_bo0 =  np.asarray(net.content['lstm'].b['o'].eval())
                    # lstm_bf0 =  np.asarray(net.content['lstm'].b['f'].eval())
                    # lstm_bc0 =  np.asarray(net.content['lstm'].b['c'].eval())
                    #do_Lz0 = np.asarray(net.content['deep_out_layer'].Lz.eval())
                    #do_Lh0 = np.asarray(net.content['deep_out_layer'].Lh.eval())
                    #do_We0 = np.asarray(net.content['deep_out_layer'].WordEmb.eval())
                    #do_Lo0 = np.asarray(net.content['deep_out_layer'].Lo.eval())
                    #
                    #h0_hidden_w0 = np.asarray(net.content['h0_hidden_layer'].W.eval())
                    #h0_init_w0 = np.asarray(net.content['h0_initial'].W.eval())
                    #
                    #c0_hidden_w0 = np.asarray(net.content['c0_hidden_layer'].W.eval())
                    #c0_init_w0 = np.asarray(net.content['c0_initial'].W.eval())

                    #we0 = np.asarray(net.content['h0_hidden_layer'].W.eval())
                    
                    function_output = net.train_function(batch_range)
                    e_cost += function_output[0]
                    #do_Lz1 = np.asarray(net.content['deep_out_layer'].Lz.eval())
                    #do_Lh1 = np.asarray(net.content['deep_out_layer'].Lh.eval())
                    #do_We1 = np.asarray(net.content['deep_out_layer'].WordEmb.eval())
                    #do_Lo1 = np.asarray(net.content['deep_out_layer'].Lo.eval())

                    #h0_hidden_w1 = np.asarray(net.content['h0_hidden_layer'].W.eval())
                    #h0_init_w1 = np.asarray(net.content['h0_initial'].W.eval())
                    #
                    #c0_hidden_w1 = np.asarray(net.content['c0_hidden_layer'].W.eval())
                    #c0_init_w1 = np.asarray(net.content['c0_initial'].W.eval())

                    #we1 = np.asarray(net.content['h0_hidden_layer'].W.eval())

                    #iwe_W1 = np.asarray(np.asarray(net.content['iwe'].We.eval()))
                    #we_W1 = np.asarray(net.content['we'].We.eval())
                    #lstm_Wi1 = np.asarray(net.content['lstm'].W['i'].eval())
                    #lstm_Wo1 =  np.asarray(net.content['lstm'].W['o'].eval())
                    #lstm_Wf1 =  np.asarray(net.content['lstm'].W['f'].eval())
                    #lstm_Wc1 =  np.asarray(net.content['lstm'].W['c'].eval())
                    #lstm_Ui1 = np.asarray(net.content['lstm'].U['i'].eval())
                    #lstm_Uo1 =  np.asarray(net.content['lstm'].U['o'].eval())
                    #lstm_Uf1 =  np.asarray(net.content['lstm'].U['f'].eval())
                    # lstm_Uc1 =  np.asarray(net.content['lstm'].U['c'].eval())
                    # lstm_bi1 = np.asarray(net.content['lstm'].b['i'].eval())
                    # lstm_bo1 =  np.asarray(net.content['lstm'].b['o'].eval())
                    # lstm_bf1 =  np.asarray(net.content['lstm'].b['f'].eval())
                    # lstm_bc1 =  np.asarray(net.content['lstm'].b['c'].eval())
                    lstm_Wi1 = np.asarray(net.content['lstm_attend'].W['i'].eval())
                    lstm_Wo1 =  np.asarray(net.content['lstm_attend'].W['o'].eval())
                    lstm_Wf1 =  np.asarray(net.content['lstm_attend'].W['f'].eval())
                    lstm_Wc1 =  np.asarray(net.content['lstm_attend'].W['c'].eval())
                    #lstm_Ui1 = np.asarray(net.content['lstm_attend'].U['i'].eval())
                    #lstm_Uo1 =  np.asarray(net.content['lstm_attend'].U['o'].eval())
                    #lstm_Uf1 =  np.asarray(net.content['lstm_attend'].U['f'].eval())
                    #lstm_Uc1 =  np.asarray(net.content['lstm_attend'].U['c'].eval())

                    #lstm_Zi1 = np.asarray(net.content['lstm_attend'].Z['i'].eval())
                    #lstm_Zo1 =  np.asarray(net.content['lstm_attend'].Z['o'].eval())
                    #lstm_Zf1 =  np.asarray(net.content['lstm_attend'].Z['f'].eval())
                    #lstm_Zc1 =  np.asarray(net.content['lstm_attend'].Z['c'].eval())

                    #lstm_Zcon1 = np.asarray(net.content['lstm_attend'].Zcontext.eval())
                    #lstm_Hcon1 = np.asarray(net.content['lstm_attend'].Hcontext.eval())
                    #lstm_Xcon1 = np.asarray(net.content['lstm_attend'].Xcontext.eval())
                    #lstm_Va1 = np.asarray(net.content['lstm_attend'].Va.eval())

                    trainer.all_i.append(function_output[0] / trainer.opts['batch_size'])

                    if (i % 20 == 0):
                        log = ('Finished iteration %d of epoch %d: %f' %

                               (i, e, function_output[0] / trainer.opts['batch_size']))
                        # pdb.set_trace()
                        #print(np.argmax(function_output[2][0,:,:,0], 1))
                        print("Guess:")
                        print(np.argmax(function_output[1][0,0:10,:,0], 1))
                        print("Ground truth:")
                        print(np.argmax(function_output[-1][0,0:10,:,0], 1))
                        print("l2: %f" % function_output[2])

                        print('LSTM wi param diff %f ' % np.mean(np.abs(lstm_Wi0-lstm_Wi1)/np.abs(lstm_Wi0)))
                        print('LSTM wc param diff %f ' % np.mean(np.abs(lstm_Wc0-lstm_Wc1)/np.abs(lstm_Wc0)))
                        print('LSTM wo param diff %f ' % np.mean(np.abs(lstm_Wo0-lstm_Wo1)/np.abs(lstm_Wo0)))
                        print('LSTM wf param diff %f ' % np.mean(np.abs(lstm_Wf0-lstm_Wf1)/np.abs(lstm_Wf0)))
                        #              
                        #print('LSTM wi param diff min %f ' % np.max(np.abs(lstm_Wi0-lstm_Wi1)/np.minimum(1e-8, np.abs(lstm_Wi0, lstm_Wi1))))
                        #print('LSTM wc param diff min %f ' % np.max(np.abs(lstm_Wc0-lstm_Wc1)/np.minimum(1e-8, np.abs(lstm_Wc0, lstm_Wc1))))
                        #print('LSTM wo param diff min %f ' % np.max(np.abs(lstm_Wo0-lstm_Wo1)/np.minimum(1e-8, np.abs(lstm_Wo0, lstm_Wo1))))
                        #print('LSTM wf param diff min %f ' % np.max(np.abs(lstm_Wf0-lstm_Wf1)/np.minimum(1e-8, np.abs(lstm_Wf0, lstm_Wf1))))

                        print('LSTM wi param diff max %f ' % np.max(np.abs(lstm_Wi0-lstm_Wi1)/np.maximum(1e-8, np.abs(lstm_Wi0, lstm_Wi1))))
                        print('LSTM wc param diff max %f ' % np.max(np.abs(lstm_Wc0-lstm_Wc1)/np.maximum(1e-8, np.abs(lstm_Wc0, lstm_Wc1))))
                        print('LSTM wo param diff max %f ' % np.max(np.abs(lstm_Wo0-lstm_Wo1)/np.maximum(1e-8, np.abs(lstm_Wo0, lstm_Wo1))))
                        print('LSTM wf param diff max %f ' % np.max(np.abs(lstm_Wf0-lstm_Wf1)/np.maximum(1e-8, np.abs(lstm_Wf0, lstm_Wf1))))

                        #print('LSTM ui param diff %f ' % np.mean(np.abs(lstm_Ui0-lstm_Ui1)/np.abs(lstm_Ui0)))
                        #print('LSTM uc param diff %f ' % np.mean(np.abs(lstm_Uc0-lstm_Uc1)/np.abs(lstm_Uc0)))
                        #print('LSTM uo param diff %f ' % np.mean(np.abs(lstm_Uo0-lstm_Uo1)/np.abs(lstm_Uo0)))
                        #print('LSTM uf param diff %f ' % np.mean(np.abs(lstm_Uf0-lstm_Uf1)/np.abs(lstm_Uf0)))
                        #
                        #print('LSTM zi param diff %f ' % np.mean(np.abs(lstm_Zi0-lstm_Zi1)/np.abs(lstm_Zi0)))
                        #print('LSTM zc param diff %f ' % np.mean(np.abs(lstm_Zc0-lstm_Zc1)/np.abs(lstm_Zc0)))
                        #print('LSTM zo param diff %f ' % np.mean(np.abs(lstm_Zo0-lstm_Zo1)/np.abs(lstm_Zo0)))
                        #print('LSTM zf param diff %f ' % np.mean(np.abs(lstm_Zf0-lstm_Zf1)/np.abs(lstm_Zf0)))

                        #print('LSTM Zcon param diff %f ' % np.mean(np.abs(lstm_Zcon0-lstm_Zcon1)/np.abs(lstm_Zcon0)))
                        #print('LSTM Hcon param diff %f ' % np.mean(np.abs(lstm_Hcon0-lstm_Hcon1)/np.abs(lstm_Hcon0)))
                        #print('LSTM Xcon param diff %f ' % np.mean(np.abs(lstm_Xcon0-lstm_Xcon1)/np.abs(lstm_Xcon0)))
                        #print('LSTM Va param diff %f ' % np.mean(np.abs(lstm_Va0-lstm_Va1)/np.abs(lstm_Va0)))
                        #print('deep out Lz param diff %f ' % np.mean(np.abs(do_Lz0-do_Lz1)/np.abs(do_Lz0)))
                        #print('deep out Lo param diff %f ' % np.mean(np.abs(do_Lo0-do_Lo1)/np.abs(do_Lo0)))
                        #print('deep out Lh param diff %f ' % np.mean(np.abs(do_Lh0-do_Lh1)/np.abs(do_Lh0)))
                        #print('deep out Lz param diff %f ' % np.mean(np.abs(do_We0-do_We1)/np.abs(do_We0)))

                        #print('h0 hidden layer param diff %f ' % np.mean(np.abs(h0_hidden_w0-h0_hidden_w1)/np.abs(h0_hidden_w0)))
                        #print('h0 init layer param diff %f ' % np.mean(np.abs(h0_init_w0-h0_init_w1)/np.abs(h0_init_w0)))
                        #print('c0 hidden layer param diff %f ' % np.mean(np.abs(c0_hidden_w0-c0_hidden_w1)/np.abs(c0_hidden_w0)))
                        #print('c0 init layer param diff %f ' % np.mean(np.abs(c0_init_w0-c0_init_w1)/np.abs(c0_init_w0)))

                        #print('word emb param diff %f ' % np.mean(np.abs(we0-we1)/np.abs(we0)))




                        #pdb.set_trace()
                        #print('softmax log loss %f | l2 loss %f' % (function_output[3], function_output[4]))
                        #print(function_output[2][0,0:5,0:5,0], 1)
                        #print(np.sum(function_output[-2][0:4,0:4]))
                        # print('lstm_uc param diff %f ' % np.mean(np.abs(lstm_Uc0-lstm_Uc1)/np.abs(lstm_Uc0)))
                        # print('lstm_bi param diff %f ' % np.mean(np.abs(lstm_bi0-lstm_bi1)/np.abs(lstm_bi0)))
                        # print('lstm_bo param diff %f ' % np.mean(np.abs(lstm_bo0-lstm_bo1)/np.abs(lstm_bo0)))
                        # print('lstm_bf param diff %f ' % np.mean(np.abs(lstm_bf0-lstm_bf1)/np.abs(lstm_bf0)))
                        # print('lstm_bc param diff %f ' % np.mean(np.abs(lstm_bc0-lstm_bc1)/np.abs(lstm_bc0)))

                        print(log)
                        self.WriteLog(log)

                    if (trainer.opts['test_emp']):
                        empirical_err += EmpError(function_output[-2], function_output[-1],
                                                  net.layer_opts['softmax_norm_dim'])
                # After one epoch : switching mode to validation
                # trainer.opts['validation'] = True
                # trainer.opts['train'] = False

                self.LRRestart(net)
                # Saving
                if (trainer.opts['save'] and ((e + 1) % trainer.opts['save_freq'] == 0)):
                    net1 = net.NNCopy()
                    SaveList([net1, trainer, e], '../../data/trained_model/%s_e-%05d.dat' % (net1.name, e))

                trainer.all_e.append(e_cost/num_ite)
                print(num_ite)
                log = 'Train cost: %f' % (e_cost/num_ite)
                print(log)
                self.WriteLog(log)

            t_end = time.time()

            # If we are validating the network
            if (trainer.opts['validation']):
                log = 'number validation iteration : {num_val}'.format(num_val = num_val_ite)
                self.WriteLog(log)
                
                for i in np.arange(0, num_val_ite):
                    batch_range = np.arange(i, trainer.opts['num_val_sample'], num_val_ite, dtype=np.int64)
                    function_output = net.val_function(batch_range)
                    v_cost += function_output[0]
                    if (trainer.opts['test_emp']):
                        v_empirical_err += EmpError(function_output[-2], function_output[-1],
                                                    net.layer_opts['softmax_norm_dim'])
                    if (i%20 == 0):
                        log = ('Finished validation iteration %d of epoch %d: %f' %

                               (i, e, function_output[0] / trainer.opts['batch_size']))
                        #print(np.argmax(function_output[2][0,:,:,0], 1))
                        print("Guess:")
                        print(np.argmax(function_output[1][0,0:10,:,0], 1))
                        print("Ground truth:")
                        print(np.argmax(function_output[-1][0,0:10,:,0], 1))
                        #pdb.set_trace()
                        #print(np.argmax(function_output[-1][0,:,:,0], 1))
                        print(log)
                        self.WriteLog(log)

                #after one epoch: switch to train mode
                trainer.opts['train'] = True
                trainer.opts['validation'] = False

                trainer.all_val.append(v_cost/num_val_ite)
                log = 'Valid cost: %f' % (v_cost/num_val_ite)
                print(log)
                self.WriteLog(log)

            log = ('Epoch %d done in %fs' % (e, t_end - t_start))
            print(log)
            self.WriteLog(log)

            if (trainer.opts['test_emp']):
                log = ('Empirical error: %f' % (float(empirical_err) / float(num_sample)))
                trainer.all_emp.append(float(empirical_err) / float(num_sample))
                print(log)
                self.WriteLog(log)
                log = ('Val empirical error: %f' % (float(v_empirical_err) / float(num_val_sample)))
                trainer.all_v_emp.append(float(v_empirical_err) / float(num_val_sample))
                print(log)
                self.WriteLog(log)


                # print(net.net_opts['lr'][2].eval())
