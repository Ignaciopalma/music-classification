import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import random
from keras.models import model_from_json
from keras import optimizers
from keras.callbacks import TensorBoard
import sys

json_file = open('model.json')

model = model_from_json(json_file.read())
model.load_weights('model.h5')

model.compile(optimizer=optimizers.Adadelta(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

line_args = sys.argv[1:]
NUMBER_OF_GENRES = 8
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
    arguments['START'] = int(line_args[arg_index + 1])
  elif arg == '--finish':
    arguments['FINISH'] = int(line_args[arg_index + 1])

print('HYPER PARAMS: {}'.format(arguments))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

UNIFORM_IMAGE_SIZE = (1000, 1000)

def build_train_and_test_data():
  train_data = []
  train_labels = []

  dir_name = './small-melspec-tracks/'
  list_of_files = os.listdir(dir_name)[arguments['START']:arguments['FINISH']]
  for file_index, file in enumerate(list_of_files):
    raw_image = Image.open(dir_name + file)
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




# def build_train_and_test_data():
#   train_data = []
#   train_labels = []
#   for dir_index, directory_name in enumerate(os.listdir('./tmp')):
#     print('Loading ' + directory_name + '...')
#     for file_index, file in enumerate(os.listdir('./tmp/' + directory_name + '/')):
#       if file_index < arguments["CAP_BY_LABEL"]:
#         raw_image = Image.open('./tmp/' + directory_name + '/' + file)
#         resized_image = raw_image.resize(UNIFORM_IMAGE_SIZE)
#         image_tensor = np.asarray(resized_image)
#         train_data.append(image_tensor)
#         train_labels.append(dir_index)
#         print('file {} loaded. shape: {}'.format(file_index, image_tensor.shape))
#   return (
#     np.array(train_data).astype(np.float32),
#     np.array(train_labels, np.int32)
#   )

