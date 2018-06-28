import numpy as np
import loaddataset

dt = np.dtype('f8')


def sigmoid(X):

    return 1.0 / (1.0 + np.exp(-X))


def convertToOneHot(Y):

    n_labels = 10
    ohm = np.zeros((Y.shape[0], n_labels))
    # empty one-hot matrix

    for i in range(Y.shape[0]):
        ohm[i, Y[i]] = 1
    return ohm


def softmax(X):

    expo = np.exp(X).astype(dt)
    total = expo.sum(axis=0, keepdims=True)
    output = expo / total
    return output


def initializeParameters(h_size):

    W1 = np.random.randn(h_size, 32 * 32 * 3).astype(dt) * 0.1
    b1 = np.random.randn(h_size, 1).astype(dt) * 0.1
    W2 = np.random.randn(10, h_size).astype(dt)
    b2 = np.random.randn(10, 1).astype(dt)

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }
    return parameters


def forward(inputImage, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, inputImage.T) + b1
    A1 = sigmoid(Z1)

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
    print(np.sum(Y.T * y) / 100)
    # 10 000 * 10Tepe
    loss = -np.sum(Y * np.log(Y_pred), axis=1, keepdims=True)
    print(loss.sum() / m)
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

if __name__ == '__main__':

    parameters = initializeParameters(64)
    for j in range(100):

        closs = 0
        for i in range(1, 6):

            # +str(batch_num)+'.bin'
            filename = './cifar-10-batches-py/data_batch_' + str(i)
            dataset = loaddataset.main()

            imput = np.asarray(dataset['data'])
            imput = np.divide(imput, 256)
            Y = np.asarray(dataset['labels'])
            Y = np.reshape(Y, (10000, 1))

            cache, fin = forward(imput, parameters)

            y_pred = convertToOneHot(Y)
            y_pred = y_pred.T
            loss = lossfx(y_pred, fin)
            parameters = calculateGradient(
                imput, parameters, cache, loss, learning_rate=0.001)
            totalloss = np.sum(loss)
            closs += totalloss
            print(totalloss)
        if j % 1 == 0:
            print("loss at epoc {} is {} ".format(j, closs))
