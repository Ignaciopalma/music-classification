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
from sklearn.metrics import confusion_matrix
import sys

TEST_FILES_DIR = '../test_files/'
CSV_PATH = '../sample_train_labels.csv'
JSON_MODEL_PATH = '../re_model.json'
WEIGHTS_PATH = '../re_model.h5'

csv_file = pd.read_csv(CSV_PATH)
all_genres = csv_file['genre'].unique()
genre_to_index = {genre: index for index, genre in enumerate(all_genres)}
index_to_genre = {index: genre for index, genre in enumerate(all_genres)}

def get_real_label(file):
  file_numbers = re.sub('([a-z])|(\.)', '', file)
  file_as_int = int(file_numbers)
  return csv_file[csv_file['track_id'] == file_as_int]['genre'].values[0]

def get_predicted_label(file):
  raw_image = Image.open(TEST_FILES_DIR + file)
  resized_image = raw_image.resize((1000, 1000))
  sample_tensor = np.array([np.asarray(resized_image)], np.float32)
  sample_tensor /= 255
  prediction_proba = model.predict(sample_tensor)
  max_index = np.argmax(prediction_proba)
  return index_to_genre[max_index]

json_architecture = open(JSON_MODEL_PATH)
model = model_from_json(json_architecture.read())
model.load_weights(WEIGHTS_PATH)

real_labels = []
predicted_labels = []

for file in os.listdir(TEST_FILES_DIR):
  real_labels.append(get_real_label(file))
  predicted_labels.append(get_predicted_label(file))

confusion = confusion_matrix(real_labels, predicted_labels, all_genres)
print('real labels: {}'.format(real_labels))
print('predicted labels: {}'.format(predicted_labels))
print('confusion: {}'.format(confusion))
