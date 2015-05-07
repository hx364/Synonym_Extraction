#Build a deeper neural network
import cPickle
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class HiddenLayer(object):
    #The class for normal hidden layer: shape(n_input, n_output), with b enabled
    def __init__(self, n_input, n_output, W_init=None, b_init=None, activation=T.tanh):

        #Initialize W
        if W_init is None:
            rng = np.random.RandomState(1234)
            W_init = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_input + n_output)),
                    high=np.sqrt(6. / (n_input + n_output)),
                    size=(n_input, n_output)))
        if b_init is None:
            b_init = np.zeros((n_output,))

        self.W = theano.shared(value=W_init.astype(theano.config.floatX),name='W',borrow=True)
        self.b = theano.shared(value=b_init.astype(theano.config.floatX),name='b',borrow=True)
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(x, self.W) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))

class HiddenLayerWithoutB(object):
    #The class without B hidden layer
    def __init__(self, n_input, n_output, W_init=None, activation=T.tanh):
        #Initialize W_init
        if W_init is None:
            rng = np.random.RandomState(12)
            W_init = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_input + n_output)),
                    high=np.sqrt(6. / (n_input + n_output)),
                    size=(n_input, n_output)))

        self.W = theano.shared(value=W_init.astype(theano.config.floatX),name='W',borrow=True)
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W]

    def output(self, x):
        lin_output = T.dot(x, self.W)
        return (lin_output if self.activation is None else self.activation(lin_output))


class FlattenLayer(object):
    #The flatten layer, change tensor size from n to n-1
    def __init__(self, flattendim=2):
        self.dim=flattendim
        self.params=None
    def output(self, x):
        return T.flatten(x, outdim=self.dim)


class DeepMLP(object):
    """
    Input of MLP
    x: A 3d tensor: (n_batch, word_embedding_dim, 3), 3 represents 3channels: w1, w2, w1*w2
    y: A 1d tensor: (n_batch)
    """
    def __init__(self):
        #For a quick implementation
        #Build MLP within __ini__, this is ugly though

        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        self.layers.append(HiddenLayerWithoutB(3, 10))
        self.layers.append(FlattenLayer(flattendim=2))
        self.layers.append(HiddenLayer(1000, 100, activation=T.tanh))
        self.layers.append(HiddenLayer(100, 1, activation=None))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            if layer.params is not None:
                self.params += layer.params

    def output(self, x):
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def modified_square_error(self, x, y):
        #Compute the squared euclidean error of the network output against the "true" output y
        error = T.flatten(self.output(x)) - y
        error = error*(error*y<0)
        return T.sum(error)

    def f1_score(self, x, y_true):
        pred = T.flatten(self.output(x))
        pred = pred.eval()
        pred[pred>=0]=1
        pred[pred<0]=-1
        return f1_score(y_true, pred)


def gradient_updates_momentum(cost, params, learning_rate, momentum):

    #Compute updates for gradient descent with momentum
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates


def mlp_train(x_train, x_test, y_train, y_test):
    print x_train.shape, x_test.shape
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'

    mlp = DeepMLP()
    # Create Theano variables for the MLP input
    mlp_input = T.matrix('mlp_input')
    # ... and the desired output
    mlp_target = T.vector('mlp_target')
    n_iter = T.scalar('n_iter')


    learning_rate = 0.005
    momentum = 0.0
    batch_size = 100
    n_train_batches = x_train.shape[0]/batch_size
    val_batch_size = x_test.shape[1]/n_train_batches


    # Create a function for computing the cost of the network given an input
    cost = mlp.modified_square_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target, n_iter], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    iteration = 0
    max_iteration = 1000

    while iteration < max_iteration:
        for i in range(n_train_batches):
            x_train_sample = x_train[i*batch_size, (i+1)*batch_size]
            y_train_sample = y_train[i*batch_size, (i+1)*batch_size]
            train_cost = train(x_train_sample, y_train_sample)

            x_val_sample = x_test[i*val_batch_size, (i+1)*val_batch_size]
            y_val_sample = y_test[i*val_batch_size, (i+1)*val_batch_size]
            val_cost = mlp.modified_square_error(x_val_sample, y_val_sample).eval()
            print "Epoch: %r, Iter: %r, Train_cost: %.3f, Val_cost: %.3f" %(iteration, i, train_cost, val_cost)

        full_val_cost =mlp.modified_square_error(x_test, y_test).eval()
        full_f1_score = mlp.f1_score(x_test, y_test).eval()
        print "Epoch: %r , Val cost: %.3f" %(iteration, val_cost)
        iteration += 1

    return mlp


def main():
    X_train, X_test, y_train, y_test = cPickle.load(open('word_mat_min.bin', 'rb'))
    mlp = mlp_train(X_train, X_test, y_train, y_test)

main()