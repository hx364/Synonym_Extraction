#Build a basic MLP for synonym detection

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import cPickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

class HiddenLayer(object):
    def __init__(self, n_in, n_out, W_init=None, b_init=None,
                 activation=T.tanh):
        self.activation = activation
        if W_init is None:
            rng = numpy.random.RandomState(1234)
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W_init = theano.shared(value=W_values, name='W', borrow=True)

        if b_init is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_init = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W_init
        self.b = b_init
        # parameters of the model
        self.params = [self.W, self.b]

    def output(self, input):
        #output of the layer
        lin_output = T.dot(input, self.W) + self.b
        output = lin_output if self.activation is None else self.activation(lin_output)
        return output



class MLP(object):
    def __init__(self, layer_size):
        """
        layer_size: list of integers, indicating the number of neurons in each layer
        """
        #Build the layer
        size_list = zip(layer_size[:-1], layer_size[1:])
        self.layers = []
        for input, output in size_list:
            self.layers.append(HiddenLayer(n_in=input, n_out=output))

        #L1 and L2 constraints
        self.L1 = numpy.sum([abs(self.layers[i].W).sum() for i in range(len(self.layers))])
        self.L2_sqr = numpy.sum([(self.layers[i].W ** 2).sum() for i in range(len(self.layers))])

        #params of the mlp
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        #Output of the MLP
        for layer in self.layers:
            x = layer.output(x)
        return x

    def modified_square_error(self, x, y):
        #Compute the squared euclidean error of the network output against the "true" output y
        error = T.flatten(self.output(x)) - y
        error = error*(error*y<0)
        return T.mean(error**2)

    def predict(self, x):
        #The label predicted by MLP
        output = T.flatten(self.output(x)).eval()
        output[output>0]=1
        output[output<0]=-1
        return output

    def f1_score(self, x, y_true):
        pred = self.predict(x)
        return f1_score(y_true, pred, average='micro')


def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


def mlp_train(X_train_np, X_test_np, y_train_np, y_test_np,
             learning_rate=0.01, L1_reg=0.00, L2_reg=0.00, n_epochs=1000,batch_size=10000, n_hidden=100):
    #X_train, y_train = shared_dataset((X_train_np, y_train_np))
    #X_test, y_test = shared_dataset((X_test_np, y_test_np))
    # compute number of minibatches for training, validation and testing
    #Get the balanced data
    """
    pos_index = numpy.asarray(range(len(y_train_np)))[y_train_np>0]
    neg_index = numpy.asarray(range(len(y_train_np)))[y_train_np<0]
    n_train_batches = int(2*min(len(pos_index), len(neg_index))/batch_size)
    """
    n_train_batches = int(y_train_np.shape[0])/batch_size
    val_batch_size = X_test_np.shape[0]/n_train_batches

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    # construct the MLP class
    mlp = MLP([250, n_hidden, 1])
    cost = (mlp.modified_square_error(x, y) + L1_reg * mlp.L1+ L2_reg * mlp.L2_sqr)

    validate_model = theano.function(inputs=[x, y],outputs=cost)

    gparams = [T.grad(cost, param) for param in mlp.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(mlp.params, gparams)
    ]

    train_model = theano.function(inputs=[x, y],outputs=cost,updates=updates)

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    #neg_index = shuffle(neg_index)
    #index = shuffle(numpy.concatenate((pos_index, neg_index[:len(pos_index)])))

    while (epoch < n_epochs) and (not done_looping):
        #get balanced subsample data
        """
        neg_index = shuffle(neg_index)
        index = shuffle(numpy.concatenate((pos_index, neg_index[:len(pos_index)])))
        x_train_sub = X_train_np[index]
        y_train_sub = y_train_np[index]
        """
        x_train_sub = X_train_np
        y_train_sub = y_train_np

        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            train_cost = train_model(x_train_sub[minibatch_index*batch_size: (minibatch_index + 1) * batch_size],
                                    y_train_sub[minibatch_index*batch_size: (minibatch_index + 1) * batch_size])
            validate_cost = validate_model(X_test_np[minibatch_index*val_batch_size:(minibatch_index + 1) * val_batch_size],
                                  y_test_np[minibatch_index*val_batch_size: (minibatch_index + 1)* val_batch_size])

            iter = (epoch - 1) * n_train_batches + minibatch_index
            train_f1 = mlp.f1_score(X_train_np, y_train_np)
            val_f1 = mlp.f1_score(X_test_np, y_test_np)
            print 'epoch %r, minibatch %r/%r, train cost %.3f , val cost: %.3f, train f1: %.3f, val f1: %.3f' \
                      %(epoch,minibatch_index + 1,n_train_batches, train_cost, validate_cost, train_f1, val_f1)
            #print train_f1, val_f1


    end_time = time.clock()
    print 'Optimization complete. Best validation score of %f obtained at iteration %i' %(best_validation_loss, best_iter + 1)
    return mlp

def main():
    X_train, X_test, y_train, y_test = cPickle.load(open('word_mat_min.bin', 'rb'))
    print X_train.shape, X_test.shape
    mlp = mlp_train(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()