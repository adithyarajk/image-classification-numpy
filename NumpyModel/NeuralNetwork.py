import numpy as np
import loaddataset

def sigmoid(x):
    """
    calcutes the sigmoid activation for 
    @param X The input after the linear transofrmation

    @return sigmoid the sigoid of the input ( 1/(1+exp(-x)) )

    """

    x = np.clip(x, -100, 100)     
    print('sigmoid values{}', format(x))
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    return sigmoid

def convertToOneHot(Y, n_labels=10):
    """
    Converts the input labels into one-hot vectors
    @params Y the input labels 
            n_labels the number of classes
    @return ohm The one hot vectors of dimension [Num_examples, classes]
    """
    n_labels = 10
    ohm = np.zeros((Y.shape[0], n_labels))
    # empty one-hot matrix

    for i in range(Y.shape[0]):
        ohm[i, Y[i]] = 1
    return ohm


def softmax(x):
    """
    Calculates the softmax of batch of inputs in the output layer
    @param x Values after the linear transformation 

    @return output The softmax output
    """
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu)/sigma
    expo = np.exp(x)
    total = expo.sum(axis=0, keepdims=True)
    output = expo / total
    return output

class FeedForwardLayer():

    def __init__(self, inputdata, hyperparameters):
        self.layer_size = hyperparameters.layer_size
        self.parameters = initializeLayerParameters()


    def initializeLayerParameters(curr_layer_size, prev_layer_size):
        """
        Initialize the parameter in a layer based on the size 
        @param curr_layer_size The size of the current layer
               prev_layer_size the size of the previous layer
        @return 
            parameters A dictionary containing the weights and bias matrices
        """

        parameters['W'] = np.random.randn(curr_layer_size, last_layer_size) * 0.001
        parameters['b'] = np.random.randn(curr_layer_size, 1) * 0.001

        return parameters

    def forward(self, input_data):
        """
        Forward propogation of the input data
        """
        Z = self.parameters['W'] * input_data + self.parameters['b']
        A = sigmoid(Z)

        return Z, A

class FeedForwardNetworkModel():

    def __init__(self,input_data, layer_sizes,):

        for i, _ in layer_sizes:
            initializeLayerParameters()

    def forward(self, input_data,)

def forward(input_image, parameters):

    prev_input = input_image.T
    for W, b in zip(parameters[]
        cache['Z'] = np.dot(W, prev_input) + b
        A = sigmoid(Z)
        prev_input = A

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, }

    return cache, A2


def lossfx(Y, Y_pred):
    """
    The defenition for Cross Entropy loss
    loss = Y * log(Y_pred)
    """
    m = Y.shape[1]

    y = np.reshape(np.argmax(Y_pred, axis=0), (10000, 1))
    y = convertToOneHot(y)
    print('Accuracy : {}'.format(np.sum(Y.T * y) / 100))
    # 10 000 * 10Tepe
    loss = -np.sum(Y * np.log(Y_pred), axis=1, keepdims=True)#
    #print(loss.sum() / m)
    return loss


def meanSquaredError(Y, Y_pred):
    """
    Y is 10,10000
    """

    m = Y.shape[1]


def calculateGradient(
        inputimage,
        parameters,
        cache,
        loss,
        learning_rate=0.001):

    X = loss
    dZ2 = cache['Z2']
    dZ2 = dZ2 * (1 - dZ2)
    X = X * dZ2

    dW2 = np.dot(X, cache['A1'].T)
    db2 = X.sum(axis=1, keepdims=True) / 10000
    X = np.dot(parameters['W2'].T, X)

    parameters['W2'] = parameters['W2'] - learning_rate * dW2

    parameters['b2'] = parameters['b2'] - learning_rate * db2

    dZ1 = cache['Z1']
    X = X * dZ1 * (1 - dZ1)

    dW1 = np.dot(X, inputimage)
    db1 = X.sum(axis=1, keepdims=True) / 10000

    assert dW1.shape == parameters['W1'].shape

    parameters['W1'] = parameters['W1'] - learning_rate * dW1
    parameters['b1'] = parameters['b1'] - learning_rate * db1

    return parameters


batch_num = 1
# +str(batch_num)+'.bin'
filename = './cifar-10-batches-py/data_batch_2'

