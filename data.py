import os
import pickle
import itertools
from pydub import AudioSegment

#this creates the dataset to be used for training

# assumes the segments folder exists within the same dirctory as this file
input_folder = 'segments'
output_folder = 'dataset'
output_of_list_of_datasets='dataset.pkl'
# change pitch with the following numbers
speed_factors = [0,1.1, 1.25, 1.4]
pitch_factors = [.8,0,1.25,1.9]

def change_audio(full_audio_path):
  #read the audio file
  audio = AudioSegment.from_file(full_audio_path)
  file_name=full_audio_path.split('/')[-1]
  all_modifications=[]
  
  # Export the modified audio files for each pitch factor
  label_path=os.path.join('segments', full_audio_path)   

  for pitch,speed in itertools.product(pitch_factors,speed_factors):
    if pitch!=0:
      try:
        audio = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * pitch)})
      except:
        continue
    if speed!=0:
      try:
        audio = audio.speedup(playback_speed=speed) 
      except:
        continue
      
    output_file_path = os.path.join(output_folder, file_name[:-4]+f'_p_{pitch}_s_{speed}.wav')
    
    audio.export(output_file_path, format='wav')
    all_modifications.append(output_file_path)
  
  return list(zip(all_modifications,len(all_modifications)*[label_path]))

# Loop through each audio file in the input folder
dataset_list=[]
dataset_list=[change_audio(os.path.join(input_folder, f)) for f in os.listdir(input_folder) if f.endswith('.wav') ]
dataset_list=list(itertools.chain.from_iterable(dataset_list))

# save the dataset list
with open(output_of_list_of_datasets, 'wb') as f:
    pickle.dump(dataset_list, f)

