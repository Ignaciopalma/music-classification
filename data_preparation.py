import pandas as pd
import re
import os
from pprint import pprint

CSV_FILE_PATH = 'sample_train_labels.csv'
ALL_FILES_PATH = './flat-files/'
TRAIN_FILES_PATH = './train_files/'
TEST_FILES_PATH = './test_files/'

csv_file = pd.read_csv(CSV_FILE_PATH)
all_genres = csv_file['genre'].unique()

def index_to_file_name(index):
  index = str(index)
  file_name_len = 6
  if len(index) == file_name_len:
      return index

  for x in range(file_name_len - len(index)):
      index = '0' + index
  return index


def parse_label_title(title):
  title = re.sub('/', '', title)
  parsed_title = title.lower().split(' ')
  return '-'.join(parsed_title)

ids_by_label = {
  parse_label_title(genre): [
    index_to_file_name(row['track_id'])
    for index, row in csv_file.iterrows()
    if row['genre'] == genre
  ]
  for genre in all_genres
}

train_ratio, test_ratio = [0.8, 0.2]
training_files = []
testing_files = []

for genre in ids_by_label.keys():
	current_list = ids_by_label[genre]
	total_length = len(current_list)
	train_chunk = current_list[:int(total_length * train_ratio)]
	test_chunk = current_list[int(total_length * train_ratio):]

	for file in train_chunk:
		print('Moving File ' + file + ' to training folder.')
		os.system('mv ' + ALL_FILES_PATH + file + '.png ' + TRAIN_FILES_PATH)
		training_files.append(file)

	for file in test_chunk:
		print('Moving file ' + file + ' to test folder.')
		os.system('mv ' + ALL_FILES_PATH + file + '.png ' + TEST_FILES_PATH)
		testing_files.append(file)

print('Training files: {}'.format(len(training_files)))
print('Testing files: {}'.format(len(testing_files)))
print('Finish.')