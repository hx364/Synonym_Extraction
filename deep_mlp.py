#Build a deeper neural network
import cPickle
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

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
        self.layers.append(HiddenLayer(500, 100, activation=T.tanh))
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
        #print error
        error = error*(error*y<0)
        return T.mean(error**2)


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
    mlp_input = T.tensor3('mlp_input')
    # ... and the desired output
    mlp_target = T.vector('mlp_target')
    n_iter = T.scalar('n_iter')


    learning_rate = 0.1
    momentum = 0.0
    batch_size = 39805

    #Get the balanced data
    #pos_index = np.asarray(range(len(y_train)))[y_train>0]
    #neg_index = np.asarray(range(len(y_train)))[y_train<0]
    n_train_batches = int(y_train.shape[0])/batch_size
    val_batch_size = x_test.shape[0]/n_train_batches


    # Create a function for computing the cost of the network given an input
    cost = mlp.modified_square_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target, n_iter], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate/n_iter, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    iteration = 1
    max_iteration = 100

    print "...begin training"
    #neg_index = shuffle(neg_index, random_state=1)
    #index = shuffle(np.concatenate((pos_index, neg_index[:len(pos_index)])), random_state=1)

    while iteration < max_iteration:

        #get subsample in each epoch
        #neg_index = shuffle(neg_index, random_state=1)
        #index = shuffle(np.concatenate((pos_index, neg_index[:len(pos_index)])), random_state=1)
        #x_train_sub = x_train[index]
        #y_train_sub = y_train[index]
        x_train_sub = x_train
        y_train_sub = y_train

        #full_val_cost =mlp.modified_square_error(x_test, y_test).eval()
        #full_f1_score = mlp.f1_score(x_test, y_test)
        #print "Epoch: %r , Val cost: %.6f, Val F1: %.6f" %(iteration, full_val_cost, full_f1_score)

        for i in range(n_train_batches):
            x_train_sample = x_train_sub[i*batch_size:(i+1)*batch_size]
            y_train_sample = y_train_sub[i*batch_size:(i+1)*batch_size]
            train_cost = train(x_train_sample, y_train_sample, iteration)

            x_val_sample = x_test[i*val_batch_size:(i+1)*val_batch_size]
            y_val_sample = y_test[i*val_batch_size:(i+1)*val_batch_size]
            val_cost = mlp.modified_square_error(x_val_sample, y_val_sample).eval()

            train_f1 = mlp.f1_score(x_train, y_train)
            test_f1 = mlp.f1_score(x_test, y_test)
            print "Epoch: %r, Iter: %r, Train_cost: %.6f, Val_cost: %.6f, Train f1: %.3f, Test f1: %.3f" %(iteration, i+1, train_cost, val_cost, train_f1, test_f1)

        iteration += 1

    return mlp


def main():
    X_train, X_test, y_train, y_test = cPickle.load(open('word_mat_min.bin', 'rb'))
    X_train, X_test = X_train[:, :150].reshape((X_train.shape[0], 3, 50)), X_test[:,:150].reshape((X_test.shape[0],3,50))
    X_train, X_test = X_train.transpose([0,2,1]), X_test.transpose([0,2,1])

    print X_train.shape, X_test.shape

    #What's the problem of predicting all to -1
    #The extremely unbalanced data
    #So we do resample in every epoch of training, transform to balanced data
    theano.config.optimizer='None'
    mlp = mlp_train(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
