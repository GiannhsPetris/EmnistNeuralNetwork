import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History
from PyQt5 import QtCore, QtGui, QtWidgets
from gui1 import *



def data_from_CSV():
	#read train, test and 40 S and P letters data from csv
	train = pd.read_csv('emnist-letters-train.csv')
	test = pd.read_csv('emnist-letters-test.csv')
	s_ds = pd.read_csv('emnist_s_letter.csv')
	p_ds = pd.read_csv('emnist_p_letter.csv')

	return(train, test, s_ds, p_ds)

def panda_dataframe_processing(train, test, s_ds, p_ds):
	#split the dataframes into labels and data for each category
	train_data = train.iloc[:, 1:]
	train_labels = train.iloc[:, 0]

	test_data = test.iloc[:, 1:]
	test_labels = test.iloc[:, 0]

	s_data = s_ds.iloc[:, 1:]
	p_data = p_ds.iloc[:, 1:]

	#transform the labels into one hot representation.
	#test code
	train_labels = pd.get_dummies(train_labels)
	test_labels = pd.get_dummies(test_labels)

	return(train_data, train_labels, test_data, test_labels, s_data, p_data)


def numpy_array_processing(train_data, train_labels, test_data, test_labels, s_data, p_data):

	#transform every panda dataframe into a numpy array
	data_train = train_data.to_numpy()
	labels_train = train_labels.to_numpy()
	data_test = test_data.to_numpy()
	labels_test = test_labels.to_numpy()
	data_s = s_data.to_numpy()
	data_p = p_data.to_numpy()

	#undo the one hot encoding
	#test code
	labels_train = np.argmax(labels_train, axis=1)
	labels_test = np.argmax(labels_test, axis=1)

	#del train, test

	return(data_train, data_test, data_s, data_p, labels_test, labels_train)


#rotates and reshapes the images from the datasets
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])


def image_processing(data_train, data_test, data_s, data_p, labels_test, labels_train):
	#normalises the images
	data_train = np.apply_along_axis(rotate, 1, data_train)/255
	data_test = np.apply_along_axis(rotate, 1, data_test)/255
	data_s =  np.apply_along_axis(rotate, 1, data_s)/255
	data_p =  np.apply_along_axis(rotate, 1, data_p)/255

	#reshapes the numpy arrays
	data_train = np.array(data_train).reshape(-1,28,28)
	data_test = np.array(data_test).reshape(-1,28,28)
	data_s = np.array(data_s).reshape(-1,28,28)
	data_p = np.array(data_p).reshape(-1,28,28)

	return(data_train, data_test, data_s, data_p, labels_test, labels_train)



def create_model(data_train, labels_train, data_test, labels_test):
	#creates a neural network with 3 layers.
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),  # input layer 28x28 neurons
		keras.layers.Dense(128, activation='relu'),  # hidden layer 128 neurons using the rectified linear unit function for the activation
		keras.layers.Dense(26, activation='softmax') # output layer 26 neurons one for each class using the softmax function
	])

	#pass the desired optimizer and loss calculating functions to the network
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def neural_network_train(model, data, labels, mode):
	#training the network with 10 epochs and 32 batch size
	if mode == False :
		hist = History()
		model.fit(data, labels, epochs=10, verbose=1, callbacks=[hist])

		return hist
	#evaluating the network
	if mode == True :
		test_loss, test_acc = model.evaluate(data,  labels, verbose=1)

		print('Test accuracy:', test_acc)
		return test_loss, test_acc

#takes as parameters an image and with the help of the previous network trie to predict what it is
def predict(model, image, correct_label):
  class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  real_class = class_names[correct_label]

  # shows the pyplot graph of the desired letter
  show_image(image)

  return(predicted_class, real_class)


def show_image(img):
	plt.figure()
	plt.imshow(img, cmap=plt.cm.binary)
	plt.show()

#three modes. Predict a letter from the first 1000 rows of the test data or from 20 rows from the k letter data or from 20 rows of the p letter data
def predict_mode(model, data_test, labels_test, data_p, data_s, inp, num):
	if inp == 'test':
		image = data_test[num]
		label = labels_test[num]
	elif inp == 's':
		image = data_s[num]
		label = 18
	elif inp == 'p':
		image = data_p[num]
		label = 15

	predicted_class , real_class = predict(model, image, label)

	return predicted_class , real_class
