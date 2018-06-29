import numpy as np

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def main():
	dataset = unpickle(filename)
	# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

	labels = dataset["labels".encode('utf-8')]
	data = dataset["data".encode('utf-8')]
	
	np.reshape(data, (10000,32,32,3))
	datas = {'data':data, 'labels':labels}
	return datas

if __name__ == '__main__':
	main() 
