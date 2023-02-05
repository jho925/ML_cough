from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import pad_trunc, spectro_gram, open_audio, resample, rechannel, time_shift, spectro_augment
import librosa
import librosa.display
import matplotlib.pyplot as plt


class cough_dataloader(Dataset):
  def __init__(self, df):
    self.df = df
    self.duration = 5000
    self.channel = 2
    self.sr = 44100
    self.shift_pct = 0
  
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
    audio_file =  self.df.loc[idx, 'path']
    is_cough = self.df.loc[idx, 'is_cough']

    aud = open_audio(audio_file)

  
    reaud = resample(aud, self.sr)
    rechan = rechannel(reaud, self.channel)
    aud_dur = pad_trunc(rechan, self.duration)
    aud_shift = time_shift(aud_dur, self.shift_pct)
    aud_sgram = spectro_gram(aud_shift, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = spectro_augment(aud_sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram, is_cough