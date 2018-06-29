import numpy as np
import matplotlib.pyplot as plt
import tkinter

from PIL import Image


def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


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

file_name = '/home/adithyaraj/Projects/image-classification-numpy/cifar-10-batches-py/data_batch_'
def load_dataset(batch_num):
	
	dataset = unpickle(file_name+str(batch_num))
	# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

	labels = np.asarray(dataset["labels".encode('utf-8')])
	data = np.asarray(dataset["data".encode('utf-8')])

	images = np.reshape(data, (10000,3*32*32))
	labels = convertToOneHot(labels, n_labels=10)

	datas = {'images': images, 'labels': labels}
	

	print("Hello why no call me {}".format(images[0].shape))

	return datas
