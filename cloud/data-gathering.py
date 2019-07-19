import numpy as np
import pandas as pd
import os
import re
import sys

UNIFORM_IMAGE_SIZE = (1000, 1000)

line_args = sys.argv[1:]
arguments = {
  "CAP_BY_LABEL": 10
}

for arg_index, arg in enumerate(line_args):
  if arg == '--cap-by-label':
    arguments['CAP_BY_LABEL'] = int(line_args[arg_index + 1])

def index_to_file_name(index):
  index = str(index)
  file_name_len = 6
  if len(index) == file_name_len:
      return index

  for x in range(file_name_len - len(index)):
      index = '0' + index
  return index

def download_from_gcloud(ids_by_label):
  for genre, ids in ids_by_label.items():
    print('Downloading genre {}'.format(genre))
    for song_id in ids:
      os.system('gsutil cp gs://small-melspec-tracks/'+ index_to_file_name(song_id)+'.png ./tmp/'+genre+'/'+index_to_file_name(song_id)+'.png')

def parse_label_title(title):
  title = re.sub('/', '', title)
  parsed_title = title.lower().split(' ')
  return '-'.join(parsed_title)

labels = pd.read_csv('./small_train_labels.csv')
all_genres = labels['genre'].unique()

ids_by_label = {
  parse_label_title(genre): [
    row['track_id']
    for index, row in labels.iterrows()
    if row['genre'] == genre
  ][:arguments["CAP_BY_LABEL"]]
  for genre in all_genres
}

download_from_gcloud(ids_by_label)

print('training data shape: {}'.format(train_data.shape))
