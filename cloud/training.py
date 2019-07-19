import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from keras import optimizers
from keras.callbacks import TensorBoard
import sys

line_args = sys.argv[1:]
NUMBER_OF_GENRES = 8
TRAIN_FILES_DIR = './train_files/'
arguments = {
  "EPOCHS": 3,
  "BATCH_SIZE": 10,
  "LEARNING_RATE": 0.01,
  "CAP_BY_LABEL": 10
}

csv_file = pd.read_csv('./small_train_labels.csv')
all_genres = csv_file['genre'].unique()
genre_to_index = {genre: index for index, genre in enumerate(all_genres)}
arguments['START'] = 0
arguments['FINISH'] = len(csv_file)

for arg_index, arg in enumerate(line_args):
  if arg == '--epochs':
    arguments['EPOCHS'] = int(line_args[arg_index + 1])
  if arg == '--cap-by-label':
    arguments['CAP_BY_LABEL'] = int(line_args[arg_index + 1])
  elif arg == '--batch-size':
    arguments['BATCH_SIZE'] = int(line_args[arg_index + 1])
  elif arg == '--learning-rate':
    arguments['LEARNING_RATE'] = float(line_args[arg_index + 1])
  elif arg == '--start':
    arguments['START'] = float(line_args[arg_index + 1])
  elif arg == '--finish':
    arguments['FINISH'] = float(line_args[arg_index + 1])

print('HYPER PARAMS: {}'.format(arguments))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

UNIFORM_IMAGE_SIZE = (1000, 1000)

def build_train_and_test_data():
  train_data = []
  train_labels = []

  list_of_files = os.listdir(TRAIN_FILES_DIR)[arguments['START']:arguments['FINISH']]
  for file_index, file in enumerate(list_of_files):
    raw_image = Image.open(TRAIN_FILES_DIR + file)
    resized_image = raw_image.resize(UNIFORM_IMAGE_SIZE)
    image_tensor = np.asarray(resized_image)
    train_data.append(image_tensor)

    file_as_str_number = re.sub('([a-z])|(\.)', '', file)
    label_name = csv_file[csv_file['track_id'] == int(file_as_str_number)]['genre'].values[0]
    train_labels.append(genre_to_index[label_name])
  return (
    np.array(train_data).astype(np.float32),
    np.array(train_labels, np.int32)
  )

tensorboard = TensorBoard(
  log_dir='./logs',
  histogram_freq=0,
  batch_size=arguments["BATCH_SIZE"],
  write_graph=True,
  write_grads=False,
  write_images=False,
  embeddings_freq=0,
  embeddings_layer_names=None,
  embeddings_metadata=None,
  embeddings_data=None,
  update_freq='epoch'
)

train_data, train_labels = build_train_and_test_data()
train_data /= 255

dummy_labels = np.zeros((len(train_labels), NUMBER_OF_GENRES), np.float32)

for index, label in enumerate(train_labels):
  dummy_labels[index][label] = 1

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
model.add(Dense(NUMBER_OF_GENRES, activation="softmax"))

model.compile(optimizer=optimizers.Adadelta(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
  train_data,
  dummy_labels,
  batch_size=arguments["BATCH_SIZE"],
  epochs=arguments["EPOCHS"],
  verbose=1,
  callbacks=[tensorboard]
)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
