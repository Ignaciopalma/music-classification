import numpy as np
import pandas as pd
from keras.utils import Sequence, to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from keras import optimizers
import os
import re
from PIL import Image
import sys

TRAIN_FILES_DIR = '../flat-files/'
CSV_FILE_PATH = '../sample_train_labels.csv'
UNIFORM_IMAGE_SIZE = (1000, 1000)
NUMBER_OF_GENRES = 8

train_labels = pd.read_csv(CSV_FILE_PATH)
all_genres = train_labels['genre'].unique()
genre_to_index = {genre: index for index, genre in enumerate(all_genres)}

def index_to_file_name(index):
  index = str(index)
  file_name_len = 6
  if len(index) == file_name_len:
    return index

  for x in range(file_name_len - len(index)):
    index = '0' + index
  return index

def load_and_prepare_image(file_index):
	raw_image = Image.open('../flat-files/' + file_index + '.png')
	resized_image = raw_image.resize(UNIFORM_IMAGE_SIZE)
	image_tensor = np.asarray(resized_image)
	return image_tensor / 255

# class DataGenerator(Sequence):
# 	def __init__(self, data_frame, batch_size):
# 	  self.data_frame = data_frame
# 	  self.batch_size = batch_size

# 	def __len__(self):
# 	  return int(np.ceil(len(self.data_frame) / float(self.batch_size)))

# 	def __getitem__(self, idx):
# 		start = idx * self.batch_size
# 		end = (idx + 1) * self.batch_size

# 		batch = self.data_frame[start:end]
# 		labels = np.zeros((len(batch), len(all_genres)), np.float32)

# 		for index, row in batch.iterrows():
# 			position = genre_to_index[row['genre']]
# 			labels[index - start][position] = 1

# 		image_tensors = [
# 			load_and_prepare_image(index_to_file_name(row['track_id']))
# 			for index, row in batch.iterrows()
# 		]

# 		return np.array(image_tensors), labels

def one_hot_encoding(values, length, width):
  encoded_labels = np.zeros((length, width), np.float32)
  for index, label in enumerate(values):
    encoded_labels[index][label] = 1
  return encoded_labels

def build_train_and_test_data(index, batch_size):
	images = []
	labels = []
	start = index * batch_size
	end = (index + 1) * batch_size

	list_of_files = os.listdir(TRAIN_FILES_DIR)[0:10]
	for file in list_of_files:
		raw_image = Image.open(TRAIN_FILES_DIR + file)
		resized_image = raw_image.resize(UNIFORM_IMAGE_SIZE)
		image_tensor = np.array(np.asarray(resized_image), np.float32)
		images.append(image_tensor)

		image_tensor /= 255
		file_as_str_number = re.sub('([a-z])|(\.)', '', file)
		label_name = train_labels[train_labels['track_id'] == int(file_as_str_number)]['genre'].values[0]
		labels.append(genre_to_index[label_name])
	return (np.array(images), one_hot_encoding(labels, len(list_of_files), NUMBER_OF_GENRES))


def generate_arrays_from_file(batch_size):
	print('GENERATE ARRAYS FROM FILE')
	while True:
		for index in range(batch_size):
			print('BUILD TRAIN LOOP: {}'.format(index))
			train_images, train_labels = build_train_and_test_data(index, batch_size)
		yield (train_images, train_labels)


# iterator = generate_arrays_from_file('../sample_train_labels.csv', 3)

# for batch_images, batch_labels in iterator:
# 	print('BATCH LABELS: {}'.format(batch_labels))



		# for index, row in train_labels.iterrows():
		# 	file_name = index_to_file_name(row['track_id'])
		# 	raw_image = Image.open('../flat-files/' + file_name + '.png')
		# 	resized_image = raw_image.resize(arguments["UNIFORM_IMAGE_SIZE"])
		# 	image_tensor = np.array(np.asarray(resized_image), np.float32)
		# 	image_tensor /= 255
		# 	images.append(image_tensor)
		# 	file_as_str_number = re.sub('([a-z])|(\.)', '', file)
		# 	label_name = csv_file[csv_file['track_id'] == int(file_as_str_number)]['genre'].values[0]
		# 	train_labels.append(genre_to_index[label_name])

		# 	label = np.zeros((8,1), np.float32)
		# 	label[genre_to_index[row['genre']]]



# def custom_generator():
#   train_data = []
#   train_labels = []
#   for file_index, file_name in enumerate(os.listdir('../flat-files')):
#     raw_image = Image.open('../flat-files/' + file_name + '.png')
#     resized_image = raw_image.resize(UNIFORM_IMAGE_SIZE)
#     image_tensor = np.asarray(resized_image)
#     image_tensor /= 255
#     yield image_tensor, file_index




# generator = DataGenerator(train_labels, 10)

# features, labels = generator.__getitem__(0)
# # print('FEATS {}'.format(features))
# print('DICT: {}'.format(genre_to_index))
# print('LABELS {}'.format(labels))

model = Sequential()

model.add(
  Conv2D(6, (5, 5),
         data_format="channels_last",
         padding="same",
         strides=1,
         input_shape=(1000, 1000, 4)
        )
)
model.add(MaxPooling2D(pool_size=(5, 5), strides=5))
model.add(
  Conv2D(12, (5, 5),
         data_format="channels_last",
         padding="same",
         strides=1,
         input_shape=(500, 500, 6)
        )
)
model.add(MaxPooling2D(pool_size=(5, 5), strides=5))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(len(all_genres), activation="softmax"))

model.compile(optimizer=optimizers.Adadelta(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generate_arrays_from_file(1),
                    steps_per_epoch=, epochs=1)

# model.fit_generator(
# 	generator,
# 	epochs=3,
# 	verbose=1,
# 	callbacks=None,
# 	validation_data=None,
# 	validation_steps=None,
# 	class_weight=None,
# 	max_queue_size=10,
# 	workers=4,
# 	use_multiprocessing=True,
# 	shuffle=True,
# 	initial_epoch=0
# )









# def generator(features, labels, batch_size):
# 	# Create empty arrays to contain batch of features and labels#
# 	batch_features = np.zeros((batch_size, 64, 64, 3))
# 	batch_labels = np.zeros((batch_size,1))
# 	while True:
# 		for i in range(batch_size):
# 			# choose random index in features
# 			index= random.choice(len(features),1)
# 			batch_features[i] = some_processing(features[index])
# 			batch_labels[i] = labels[index]

# 		yield batch_features, batch_labels










# class DataGenerator(Sequence):
# 	def __init__(self, x_set, y_set, batch_size):
# 	  self.x, self.y = x_set, y_set
# 	  self.batch_size = batch_size

# 	def __len__(self):
# 	  return int(np.ceil(len(self.x) / float(self.batch_size)))

# 	def __getitem__(self, idx):
# 		start = idx * self.batch_size
# 		end = (idx + 1) * self.batch_size

# 		batch_x = self.x[start:end]
# 		batch_y = self.y[start:end]

# 		training_array = [
# 			resize(imread(file_name), (200, 200))
# 			for file_name in batch_x
# 		]

# 		return np.array(training_array), np.array(batch_y)

# training_generator = DataGenerator()
# print("DataGenerator: {}".format(training_generator))









# class DataGenerator(Sequence):
# 	def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
# 		self.list_IDs = list_IDs
# 		self.labels = labels
# 		self.batch_size = batch_size
# 		self.dim = dim
# 		self.n_channels = n_channels
# 		self.n_classes = n_classes
# 		self.shuffle = shuffle
# 		self.on_epoch_end()

# 	def __getitem__(self, index):
# 		"""Generate one batch of data"""

# 		# Generate indexes of the batch
# 		#   batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
# 		# batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
# 		start = index * self.batch_size
# 		end = (indexl + 1) * self.batch_size
# 		indexes = self.indexes[start:end]

# 		# Find list of IDs
# 		list_IDs_temp = [self.list_IDs[k] for k in indexes]

# 		# Generate data
# 		X, y = self.__data_generation(list_IDs_temp)

# 		return X, y

# 	def __len__(self):
# 		"""Denotes the number of batches per epoch"""
# 		return int(np.floor(len(self.list_IDs) / self.batch_size))

# 	def on_epoch_end(self):
# 		"""Updates indexes after each epoch"""
# 		print("Epoch has ended")
# 		# self.indexes = np.arange(len(self.list_IDs))
# 		# if self.shuffle == True:
# 		# 	np.random.shuffle(self.indexes)


# 	def __data_generation(self, list_IDs_temp):
# 	  """Generates data containing batch_size samples""" # X : (n_samples, *dim, n_channels)

# 	  # Initialization
# 	  X = np.empty((self.batch_size, *self.dim, self.n_channels))
# 	  y = np.empty((self.batch_size), dtype=int)

# 	  # Generate data
# 	  for i, ID in enumerate(list_IDs_temp):
# 	      # Store sample
# 	      X[i,] = np.load('data/' + ID + '.npy')

# 	      # Store class
# 	      y[i] = self.labels[ID]

# 	  return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
