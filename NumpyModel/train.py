import numpy as np

import loaddataset


def training(datasetfile,):


if __name__ == '__main__':

    layer_size = [32 * 32 * 3, 128, 32, 10]
    parameters = initializeParameters(layer_size)
    for j in range(5):

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
