from glob import glob
from pathlib import Path
from random import shuffle


from IPython.display import Audio
from tqdm import tqdm
import numpy as np

import pickle, os, torch, torchaudio , librosa 


import utils as util
import models as model

checkpoint_path = 'checkpoints/'  # Your models will be saved here
checkpoint_name = 'checkpoint'    # Base name of a model

dataset_list='dataset_list.pkl'
# hyperparameters
batch_size = 4
epochs = 10
checkpoint_freq = 100
criterion_adversarial = torch.nn.MSELoss()

with open(dataset_list, 'rb') as f:
    filenames = pickle.load(f)

# Split the data
train_filenames, test_filenames = util.shuffle_split(filenames, 0.8)

# Create a training sampler so we can ask for random batches
sampler = util.audio_sampler(train_filenames, batch_size)

Tensor = torch.FloatTensor

# models
Generator=model.AudioAutoencoder()
Discriminator=model.PatchGANDiscriminator()

#train
for e in tqdm(range(1, epochs+1)):
    # Give me a batch
    a, b = next(sampler)
    
    # Model inputs
    real_a = Tensor(a)
    real_b = Tensor(b)
    
    # Adversarial ground truths
    real = torch.ones(batch_size, 1, 30, 30).type(Tensor)
    real = torch.ones(batch_size,1, 22, 23).type(Tensor)
    
    fake = torch.zeros(batch_size, 1, 30, 30).type(Tensor)
    fake = torch.zeros(batch_size, 1, 22, 23).type(Tensor)
    
    # Train generator
    optim_G.zero_grad()
    
    fake_b = Generator(real_a)
    pred_fake =Discriminator(fake_b)
    loss_G = criterion_adversarial(pred_fake, real)
    
    loss_G.backward()
    optim_G.step()
    ###################################################################################################
    # Train discriminator
    optim_D.zero_grad()
    
    pred_real = Discriminator(real_b)
    loss_real = criterion_adversarial(pred_real, real)
    
    pred_fake = Discriminator(fake_b.detach())
    loss_fake = criterion_adversarial(pred_fake, fake)
    
    loss_D = 0.5 * (loss_real + loss_fake)
    
    loss_D.backward()

