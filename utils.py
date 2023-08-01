import librosa
from glob import glob
from pathlib import Path
from random import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def shuffle_split(alist, first_half_ratio):
    """Shufle and split a list in two. Choose the size of the parts."""
    shuffle(alist)
    k = int(first_half_ratio * len(alist))
    return alist[:k], alist[k:]


def audio_sampler(filenames, batch_size, return_max_power=False):
    """Returns a generator that iterates through the given audio file names
    and returns a batch of pairs of (source, target) spectograms.
    """
    sample_rate = 44100
    n_fft = 2048
    hop_length = 518
    n_mels = 256
    
    a = np.zeros((batch_size, 1, 256, 256)) 
    b = np.zeros((batch_size, 1, 256, 256))
    
    max_power_a = []
    
    while True:
        indices = np.random.randint(len(filenames), size=batch_size)
        # generate batch_size  integers with max value len(files_names)
        
        for i, idx in enumerate(indices):
            audio_a, _ = librosa.load(filenames[idx][0], sr=sample_rate)
            audio_b, _ = librosa.load(filenames[idx][1], sr=sample_rate)
            
            len_a = audio_a.size
            len_b = audio_b.size
            
            if len_a < 3*sample_rate:
                diff = 3*sample_rate - len_a
                audio_a = np.concatenate((audio_a, np.zeros(diff)))
                len_a = audio_a.size
                
            if len_b < 3*sample_rate:
                diff = 3*sample_rate - len_b
                audio_b = np.concatenate((audio_b, np.zeros(diff)))
                len_b = audio_b.size
                
            if len_a < len_b:
                diff = len_b - len_a
                audio_a = np.concatenate((audio_a, np.zeros(diff)))
            else:
                diff = len_a - len_b
                audio_b = np.concatenate((audio_b, np.zeros(diff)))
            
            assert audio_a.size == audio_b.size
            size = audio_a.size
            
            r = np.random.randint(size - 3*sample_rate + 1)
            
            audio_a = audio_a[r:r+3*sample_rate]
            audio_b = audio_b[r:r+3*sample_rate]            

            S_a = librosa.feature.melspectrogram(
                y=audio_a,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            S_b = librosa.feature.melspectrogram(
                y=audio_b,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )

            S_db_a = librosa.power_to_db(S_a, ref=np.max)
            S_db_b = librosa.power_to_db(S_b, ref=np.max)

            a[i, 0] = S_db_a
            b[i, 0] = S_db_b
                
            max_power_a.append(np.max(S_a))
        
        if return_max_power:
            yield (a, b), max_power_a
        else:
            yield (a, b)
