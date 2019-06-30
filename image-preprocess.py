import librosa
import librosa.display
import numpy as np
import pandas as pd
import imageio
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from PIL import Image
import sys

arguments = sys.argv[1:]

if len(arguments) == 0:
  raise Exception('Please include start and end index')

train_labels = pd.read_csv('./train_labels.csv')
start_index, end_index = sys.argv[1:]

def index_to_file_name(index):
  index = str(index)
  file_name_len = 6
  if len(index) == file_name_len:
    return index
  
  for x in range(file_name_len - len(index)):
    index = '0' + index
  return index


def save_mel_spectogram(file_name, wave_data, sampling_rate):
  fig, ax = pyplot.subplots(1, 1)
  ax.axis('off')
  S = librosa.feature.melspectrogram(y=wave_data, sr=sampling_rate, n_mels=128, fmax=8000)
  librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
  local_file_path = './tmp/' + file_name + '.png'
  fig.savefig(local_file_path, dpi=300, frameon='false', bbox_inches='tight', pad_inches=0.0)
  return local_file_path

def delete_padding(image_data, padding_color):
  # delete row wise
  new_image = []
  for row_index, row in enumerate(image_data):
    for cell_index, cell in enumerate(row):
      pixel_is_white = True

      for color_index, color in enumerate(cell):
        if color.item() is not padding_color and color_index is not 3:
          pixel_is_white = False

      if not pixel_is_white:
        new_image.append(row)
        break
      
          
  # delete column wise
  final_new_image = []
  image_width = len(new_image[0])
  image_height = len(new_image)
    
  i = 0
  while i < image_width:
    column_is_same_color = True
    new_column = []

    for j in range(0, image_height):
      cell = new_image[j][i]
      cell_is_white = True
      for color_index, color in enumerate(cell):
        if color.item() is not padding_color and color_index is not 3:
          cell_is_white = False    
      new_column.append(cell)
      if not cell_is_white:
        column_is_same_color = False
    if not column_is_same_color:
      if len(final_new_image) > 0:
        for col_index, pixel in enumerate(new_column):
          final_new_image[col_index].append(pixel)
      else:
        for col_index, pixel in enumerate(new_column):
          final_new_image.append([pixel])
    i += 1

  return np.array(final_new_image, np.uint8)


def generate_melspec_and_save_in_gs(file_index):
  track_gs_path = 'gs://music-tracks/' + file_index + '.mp3'
  track_destination_path = './tmp/' + file_index + '.mp3'
  print('Downloading {}'.format(track_gs_path))
  os.system('gsutil cp ' + track_gs_path + ' ' + track_destination_path)
  wave_data , sampling_rate = librosa.load(track_destination_path)  
  print('Generatin Spectogram..'.format(track_gs_path))
  local_image_path = save_mel_spectogram(file_index, wave_data, sampling_rate)
  image_data = np.array(Image.open(local_image_path))
  print('Removing Padding..')
  padless_image = delete_padding(image_data, 255)
  imageio.imwrite(local_image_path, padless_image)
  gs_image_path = 'gs://melspec-tracks/' + file_index + '.png'
  print('Copying from {} to {}'.format(local_image_path, gs_image_path))
  os.system('gsutil cp ' + local_image_path + ' ' + gs_image_path)
  os.system('rm ' + local_image_path)
  os.system('rm ' + track_destination_path)
  print('Finish')

for index, row in train_labels[int(start_index):int(end_index)].iterrows():
  print("Processing index {}".format(index))
  file_index = index_to_file_name(row['track_id'])
  generate_melspec_and_save_in_gs(file_index)

print("End of file {} {}".format(start_index, end_index))